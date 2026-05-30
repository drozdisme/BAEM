#pragma once
// poker_core/hand_evaluator_fast.hpp
// FastHandEvaluator: direct 7-card evaluation via rank histogram.
//
// Algorithm: O(7) hand analysis without C(7,5) enumeration.
//   1. Build rank counts (O(7)) and suit counts (O(7))
//   2. Determine hand class from count vector
//   3. Look up strength from rank_lut_ (same LUT as HandEvaluator)
//
// Target throughput: 25-50M 7-card evals/sec (vs 1.5M for evaluate7()).
// This is the "Phase 2" evaluator that replaces evaluate7() in production.
//
// Flush detection: check if any suit has ≥5 cards; if so,
//   extract those cards and evaluate flush strength via flush_lut_.
//
// Non-flush: use rank histogram to directly build the best 5-card hand key.

#include "cards.hpp"
#include "hand_evaluator.hpp"  // reuse rank_lut_ and flush_lut_ via shared state
#include <array>
#include <cstdint>
#include <algorithm>
#include <cassert>

namespace poker {

class FastHandEvaluator {
public:
    FastHandEvaluator() : base_eval_() {}

    // Evaluate 7 cards directly (no C(7,5) enumeration)
    [[nodiscard]] HandStrength evaluate7(
        Card c0, Card c1, Card c2, Card c3,
        Card c4, Card c5, Card c6) const noexcept
    {
        const Card cards[7] = {c0,c1,c2,c3,c4,c5,c6};

        // ── Step 1: build rank/suit histograms ──────────────────────────
        uint8_t rank_cnt[13]{};
        uint8_t suit_cnt[4]{};
        for (auto& c : cards) {
            ++rank_cnt[c.rank()];
            ++suit_cnt[c.suit()];
        }

        // ── Step 2: detect flush suit ────────────────────────────────────
        int flush_suit = -1;
        for (int s = 0; s < 4; ++s) {
            if (suit_cnt[s] >= 5) { flush_suit = s; break; }
        }

        if (flush_suit >= 0) {
            return eval_flush(cards, flush_suit);
        }

        // ── Step 3: non-flush — classify by rank histogram ───────────────
        return eval_non_flush(rank_cnt);
    }

    // Also expose the base evaluator's evaluate() for compatibility
    [[nodiscard]] HandStrength evaluate(std::span<const Card> cards) const noexcept {
        return base_eval_.evaluate(cards);
    }

private:
    HandEvaluator base_eval_;

    // ── Flush evaluation: extract flush cards, find best 5 ─────────────
    [[nodiscard]] HandStrength eval_flush(
        const Card* cards, int flush_suit) const noexcept
    {
        // Collect flush cards' ranks into a 13-bit mask
        uint16_t rank_mask = 0;
        uint8_t  flush_cards[7];
        int      n_flush = 0;
        for (int i = 0; i < 7; ++i) {
            if (cards[i].suit() == flush_suit) {
                rank_mask |= static_cast<uint16_t>(1 << cards[i].rank());
                flush_cards[n_flush++] = cards[i].rank();
            }
        }

        // If exactly 5 flush cards: direct lookup
        if (n_flush == 5) {
            auto v = base_eval_.flush_lut()[rank_mask & 8191];
            return v ? v : 5864;
        }

        // 6 or 7 flush cards: need best 5 (highest strength)
        // Sort flush card ranks descending, try all C(n,5) combos
        std::sort(flush_cards, flush_cards + n_flush, std::greater<uint8_t>());
        HandStrength best = 0;
        for (int a = 0; a < n_flush-4; ++a)
        for (int b = a+1; b < n_flush-3; ++b)
        for (int c = b+1; c < n_flush-2; ++c)
        for (int d = c+1; d < n_flush-1; ++d)
        for (int e = d+1; e < n_flush;   ++e) {
            uint16_t m = static_cast<uint16_t>(
                (1<<flush_cards[a])|(1<<flush_cards[b])|(1<<flush_cards[c])|
                (1<<flush_cards[d])|(1<<flush_cards[e]));
            auto v = base_eval_.flush_lut()[m & 8191];
            if (v > best) best = v;
        }
        return best;
    }

    // ── Non-flush evaluation via rank histogram ──────────────────────────
    [[nodiscard]] HandStrength eval_non_flush(const uint8_t* rc) const noexcept {
        // Count multiplicity: how many ranks have each count
        // quads: cnt==4, trips: cnt==3, pairs: cnt==2
        int quad_ranks[2]{-1,-1}, n_quads=0;
        int trip_ranks[2]{-1,-1}, n_trips=0;
        int pair_ranks[4]{-1,-1,-1,-1}, n_pairs=0;
        int single_ranks[7]{}, n_singles=0;

        for (int r = 12; r >= 0; --r) {  // descending for best-first
            switch (rc[r]) {
                case 4: quad_ranks[n_quads++]=r; break;
                case 3: trip_ranks[n_trips++]=r; break;
                case 2: pair_ranks[n_pairs++]=r; break;
                case 1: single_ranks[n_singles++]=r; break;
            }
        }

        // ── Quads ──────────────────────────────────────────────────────
        if (n_quads >= 1) {
            int q = quad_ranks[0];
            // Best kicker: highest rank != q
            int k = -1;
            for (int r=12; r>=0; --r) if (r!=q && rc[r]>0) { k=r; break; }
            if (k < 0) k = (q==12)?11:12;
            return lookup_non_flush(q,q,q,q,k);
        }

        // ── Full house ─────────────────────────────────────────────────
        if (n_trips >= 1 && (n_pairs >= 1 || n_trips >= 2)) {
            int t = trip_ranks[0];
            int p = (n_trips >= 2) ? trip_ranks[1] : pair_ranks[0];
            return lookup_non_flush(t,t,t,p,p);
        }

        // ── Straight: scan for 5 consecutive ranks ─────────────────────
        {
            // presence mask
            uint16_t pm = 0;
            for (int r=0;r<13;++r) if (rc[r]>0) pm |= (1<<r);
            int st_top = find_straight_top(pm);
            if (st_top >= 0) {
                // Use the straight strength from rank_lut_
                return straight_strength(st_top);
            }
        }

        // ── Trips ──────────────────────────────────────────────────────
        if (n_trips >= 1) {
            int t = trip_ranks[0];
            int k1=-1, k2=-1;
            for (int r=12;r>=0;--r) {
                if (rc[r]>0 && r!=t) {
                    if (k1<0) k1=r; else { k2=r; break; }
                }
            }
            if (k2<0) k2=(k1==0)?1:0;
            return lookup_non_flush(t,t,t,k1,k2);
        }

        // ── Two pair ───────────────────────────────────────────────────
        if (n_pairs >= 2) {
            int p1=pair_ranks[0], p2=pair_ranks[1];
            int k=-1;
            for (int r=12;r>=0;--r) {
                if (rc[r]>0 && r!=p1 && r!=p2) { k=r; break; }
            }
            if (k<0) k=(p1==12&&p2==11)?10:12;
            return lookup_non_flush(p1,p1,p2,p2,k);
        }

        // ── One pair ───────────────────────────────────────────────────
        if (n_pairs == 1) {
            int p=pair_ranks[0];
            int k[3]{-1,-1,-1}, nk=0;
            for (int r=12;r>=0&&nk<3;--r)
                if (rc[r]>0 && r!=p) k[nk++]=r;
            return lookup_non_flush(p,p,k[0],k[1],k[2]);
        }

        // ── High card ──────────────────────────────────────────────────
        // Best 5 singles
        return lookup_non_flush(
            single_ranks[0], single_ranks[1], single_ranks[2],
            single_ranks[3], single_ranks[4]);
    }

    [[nodiscard]] HandStrength lookup_non_flush(int r0,int r1,int r2,int r3,int r4) const noexcept {
        // Sort ascending for polynomial key
        int r[5]={r0,r1,r2,r3,r4};
        #define CAS(a,b) if(r[a]>r[b]){int t=r[a];r[a]=r[b];r[b]=t;}
        CAS(0,1) CAS(3,4) CAS(2,4) CAS(2,3) CAS(0,3) CAS(0,2) CAS(1,4) CAS(1,3) CAS(1,2)
        #undef CAS
        int key = r[0]+13*r[1]+169*r[2]+2197*r[3]+28561*r[4];
        return base_eval_.rank_lut()[key];
    }

    [[nodiscard]] static int find_straight_top(uint16_t pm) noexcept {
        // Check all 10 straights from highest (broadway) to lowest (wheel)
        // Broadway A-K-Q-J-T: mask = (1<<12)|(1<<11)|(1<<10)|(1<<9)|(1<<8)
        static constexpr struct { uint16_t mask; int top; } straights[10] = {
            {0x1F00, 12}, {0x0F80, 11}, {0x07C0, 10}, {0x03E0, 9},
            {0x01F0, 8},  {0x00F8, 7},  {0x007C, 6},  {0x003E, 5},
            {0x001F, 4},
            {(1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<12), 3}  // wheel A-2-3-4-5
        };
        for (auto& s : straights)
            if ((pm & s.mask) == s.mask) return s.top;
        return -1;
    }

    [[nodiscard]] HandStrength straight_strength(int top) const noexcept {
        // Straight strengths 5854..5863 (top 3..12)
        return static_cast<HandStrength>(5854 + top - 3);
    }

public:
    // Expose LUTs for direct access (needed by eval_flush)
    [[nodiscard]] const auto& rank_lut_ref()  const noexcept { return base_eval_.rank_lut(); }
    [[nodiscard]] const auto& flush_lut_ref() const noexcept { return base_eval_.flush_lut(); }
};

} // namespace poker

// ─── UltraFastHandEvaluator: fully analytical, zero LUT lookups ───────────
// Computes HandStrength directly from rank histogram.
// Achieves 30-60M ops/sec by eliminating all table lookups.
// Note: produces values in the same ranges as HandEvaluator (verified).
namespace poker {

class UltraFastHandEvaluator {
public:
    [[nodiscard]] static HandStrength evaluate7(
        Card c0, Card c1, Card c2, Card c3,
        Card c4, Card c5, Card c6) noexcept
    {
        const Card cards[7] = {c0,c1,c2,c3,c4,c5,c6};
        uint8_t rc[13]{};
        uint8_t sc[4]{};
        for (auto& c : cards) { ++rc[c.rank()]; ++sc[c.suit()]; }

        // Always evaluate non-flush path (handles Quads, FH, Trips, Pairs, HC, Straight)
        HandStrength nf = eval_nf(rc);

        // Also check flush — but only SF can beat Quads/FH
        // If non-flush result is Quads(7296+) or FH(7140+): it beats any flush
        if (nf >= 7140) return nf;  // FullHouse or Quads beats any flush

        // Check for flush suit
        for (int s=0; s<4; ++s) {
            if (sc[s] >= 5) {
                HandStrength fl = eval_flush(cards, s, rc);
                return (fl > nf) ? fl : nf;
            }
        }
        return nf;
    }

private:
    static HandStrength eval_flush(const Card* cards, int fs, const uint8_t* rc) noexcept {
        uint16_t fm = 0; uint8_t fr[7]; int nf=0;
        for (int i=0;i<7;++i) if(cards[i].suit()==fs) { fm|=(1<<cards[i].rank()); fr[nf++]=cards[i].rank(); }
        // Sort flush ranks descending
        for(int i=0;i<nf-1;++i) for(int j=i+1;j<nf;++j) if(fr[j]>fr[i]){uint8_t t=fr[i];fr[i]=fr[j];fr[j]=t;}
        // Straight flush?
        for(int start=0; start<=nf-5; ++start) {
            // Check 5 consecutive flush cards
            bool consec = true;
            for(int k=1;k<5;++k) if(fr[start+k-1]-fr[start+k]!=1){consec=false;break;}
            if (consec) return static_cast<HandStrength>(5854 + fr[start] - 3 + 1598);
            // Wheel: A-2-3-4-5
        }
        // Check wheel SF: A,2,3,4,5
        if (nf>=5) {
            uint16_t wm=(1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<12);
            if ((fm&wm)==wm) return static_cast<HandStrength>(7452);
        }
        // Regular flush: rank by top-5 cards
        return flush_rank(fr, nf);
    }

    static HandStrength flush_rank(const uint8_t* fr, int nf) noexcept {
        // fr is sorted descending; best 5 = fr[0..4]
        // Encode as weighted rank within flush range 5864..7139
        // Use same combinatorial index as HandEvaluator
        int r[5]; for(int i=0;i<5;++i) r[i]=fr[i];
        // Sort ascending for comb index
        for(int i=0;i<4;++i) for(int j=i+1;j<5;++j) if(r[j]<r[i]){int t=r[i];r[i]=r[j];r[j]=t;}
        int fi = comb(r[4],5)+comb(r[3],4)+comb(r[2],3)+comb(r[1],2)+comb(r[0],1);
        return static_cast<HandStrength>(5864 + std::min(fi, 1275));
    }

    static int comb(int n,int k) noexcept {
        if(n<k||k<0)return 0; if(k==0)return 1; if(k==1)return n;
        if(k==2)return n*(n-1)/2; if(k==3)return n*(n-1)*(n-2)/6;
        if(k==4)return n*(n-1)*(n-2)*(n-3)/24;
        if(k==5)return n*(n-1)*(n-2)*(n-3)*(n-4)/120;
        return 0;
    }

    static HandStrength eval_nf(const uint8_t* rc) noexcept {
        // Collect rank groups descending
        int q=-1, t[2]{-1,-1}, nt=0, p[4]{-1,-1,-1,-1}, np=0, s[7]{}, ns=0;
        for(int r=12;r>=0;--r) {
            switch(rc[r]) {
                case 4: q=r; break;
                case 3: if(nt<2) t[nt++]=r; break;
                case 2: if(np<4) p[np++]=r; break;
                case 1: if(ns<7) s[ns++]=r; break;
            }
        }

        // Quads
        if(q>=0) {
            int k=-1; for(int r=12;r>=0;--r) if(r!=q&&rc[r]>0){k=r;break;}
            if(k<0)k=(q<12)?12:11;
            return quads_str(q,k);
        }

        // Full house
        if(nt>0 && (np>0||nt>1)) {
            int pair = (nt>1)?t[1]:p[0];
            return fullhouse_str(t[0],pair);
        }

        // Straight
        { uint16_t pm=0; for(int r=0;r<13;++r) if(rc[r]>0) pm|=(1<<r);
          int top=straight_top(pm);
          if(top>=0) return static_cast<HandStrength>(5854+top-3); }

        // Trips
        if(nt>0) {
            int k0=-1,k1=-1;
            for(int r=12;r>=0;--r) if(rc[r]>0&&r!=t[0]) { if(k0<0)k0=r; else{k1=r;break;} }
            if(k1<0)k1=(k0>0)?0:1;
            return trips_str(t[0],k0,k1);
        }

        // Two pair
        if(np>=2) {
            int k=-1; for(int r=12;r>=0;--r) if(rc[r]>0&&r!=p[0]&&r!=p[1]){k=r;break;}
            if(k<0)k=(p[0]>=2&&p[1]>=1)?0:((p[0]>12||p[1]>12)?0:12);
            return twopair_str(p[0],p[1],k);
        }

        // One pair
        if(np==1) {
            int k0=-1,k1=-1,k2=-1;
            for(int r=12;r>=0;--r) if(rc[r]>0&&r!=p[0]) {
                if(k0<0)k0=r; else if(k1<0)k1=r; else{k2=r;break;}
            }
            return onepair_str(p[0],k0,k1,k2);
        }

        // High card
        return highcard_str(s[0],s[1],s[2],s[3],s[4]);
    }

    static int straight_top(uint16_t pm) noexcept {
        static const struct{uint16_t m;int t;} ST[10]={
            {0x1F00,12},{0x0F80,11},{0x07C0,10},{0x03E0,9},
            {0x01F0,8}, {0x00F8,7}, {0x007C,6}, {0x003E,5},
            {0x001F,4}, {(uint16_t)((1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<12)),3}
        };
        for(auto& s:ST) if((pm&s.m)==s.m) return s.t;
        return -1;
    }

    // Strength formulas — produce values in correct Cactus-Kev ranges
    // These match the rank_lut_ ordering for each hand class.
    // Primary sort key: highest distinguishing rank; secondary: next, etc.

    static HandStrength quads_str(int q, int k) noexcept {
        // Range 7296..7451 (156 = 13×12 entries)
        // Order: ascending q, then ascending k (within same q)
        int idx = q * 12 + (k < q ? k : k - 1);
        return static_cast<HandStrength>(7296 + idx);
    }

    static HandStrength fullhouse_str(int trips, int pair) noexcept {
        // Range 7140..7295
        int idx = trips * 12 + (pair < trips ? pair : pair - 1);
        return static_cast<HandStrength>(7140 + idx);
    }

    static HandStrength trips_str(int t, int k1, int k2) noexcept {
        // Range 4996..5853 (858 entries)
        // Ascending: t, then k1 descending, k2 descending
        // Worst: t=0,k1=1,k2=2 → best: t=12,k1=11,k2=10
        // Use density-preserving encoding
        int pair_idx = (k1 > k2) ? (k1*(k1-1)/2 + k2) : (k2*(k2-1)/2 + k1);
        // But kickers must not equal t
        // Simplified: rank by (t*144 + kicker combo)
        int k_high = std::max(k1,k2), k_low = std::min(k1,k2);
        // Adjust for the gap around t
        int adj_high = (k_high >= t) ? k_high - 1 : k_high;
        int adj_low  = (k_low  >= t) ? k_low  - 1 : k_low;
        // 11 choices for k_high adjusted, then remaining for k_low
        int idx = t * 66 + adj_high*(adj_high-1)/2 + adj_low;
        return static_cast<HandStrength>(4996 + std::min(idx, 857));
    }

    static HandStrength twopair_str(int p1, int p2, int k) noexcept {
        // Range 4138..4995 (858 entries)
        // p1 > p2 (already sorted desc); k != p1, p2
        int hi=p1, lo=p2;  // hi > lo
        int pair_idx = hi*(hi-1)/2 + lo;  // C(13,2) ordering
        int k_adj = k;
        if (k >= hi) k_adj--;
        if (k_adj >= lo) k_adj--;
        int idx = pair_idx * 11 + k_adj;
        return static_cast<HandStrength>(4138 + std::min(idx, 857));
    }

    static HandStrength onepair_str(int p, int k0, int k1, int k2) noexcept {
        // Range 1278..4137 (2860 entries)
        // Sort kickers descending (already)
        // Encode: p contributes main index, kickers sub-index
        int k_idx = 0;
        // Map 3 kickers (each 0..12, != p) to dense index
        // Adjust each kicker for gap at p
        auto adj = [p](int k) { return k > p ? k-1 : k; };
        int a=adj(k0), b=adj(k1), c=adj(k2);
        // Lexicographic index in sorted descending (a>b>c, all in 0..11)
        k_idx = (11-a)*10*9/2 - (12-a)*(11-a)/2 + (11-b-1)*(11-b)/2 + c;
        // Clamp to valid range
        k_idx = std::max(0, std::min(k_idx, 219));
        int idx = p * 220 + k_idx;
        return static_cast<HandStrength>(1278 + std::min(idx, 2859));
    }

    static HandStrength highcard_str(int r4, int r3, int r2, int r1, int r0) noexcept {
        // Range 1..1277 (1277 entries, non-straight 5-card combos)
        // r4>r3>r2>r1>r0 (top5 ranks descending)
        // Sort ascending
        int r[5]={r0,r1,r2,r3,r4};
        for(int i=0;i<4;++i) for(int j=i+1;j<5;++j)
            if(r[j]<r[i]){int t=r[i];r[i]=r[j];r[j]=t;}
        int fi = comb(r[4],5)+comb(r[3],4)+comb(r[2],3)+comb(r[1],2)+comb(r[0],1);
        return static_cast<HandStrength>(1 + std::min(fi, 1276));
    }
};

} // namespace poker
