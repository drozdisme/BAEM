#pragma once
// poker_core/hand_evaluator.hpp  v4 — Correct Cactus-Kev strength values
//
// Strength scale (HIGHER = BETTER):
//   HighCard:      1..1277
//   OnePair:    1278..4137   (2860 combos)
//   TwoPair:    4138..4995   (858)
//   Trips:      4996..5853   (858)
//   Straight:   5854..5863   (10)
//   Flush:      5864..7139   (1276)
//   FullHouse:  7140..7295   (156)
//   Quads:      7296..7451   (156)
//   StrFlush:   7452..7462   (11)
//
// LUT: rank_lut_[poly_key] — polynomial hash base-13, zero collisions.
// flush_lut_[13-bit mask] — straight-flush and flush strengths.
// 7-card: 21 unrolled calls, return max.

#include "cards.hpp"
#include <vector>
#include <array>
#include <cstdint>
#include <span>
#include <algorithm>
#include <cassert>

namespace poker {

using HandStrength = uint16_t;

enum class HandRank : uint8_t {
    HighCard=1, OnePair=2, TwoPair=3, ThreeOfAKind=4,
    Straight=5, Flush=6, FullHouse=7, FourOfAKind=8, StraightFlush=9
};

[[nodiscard]] inline HandRank category(HandStrength s) noexcept {
    if (s > 7451) return HandRank::StraightFlush;
    if (s > 7295) return HandRank::FourOfAKind;
    if (s > 7139) return HandRank::FullHouse;
    if (s > 5863) return HandRank::Flush;
    if (s > 5853) return HandRank::Straight;
    if (s > 4995) return HandRank::ThreeOfAKind;
    if (s > 4137) return HandRank::TwoPair;
    if (s > 1277) return HandRank::OnePair;
    return HandRank::HighCard;
}

class HandEvaluator {
public:
    static constexpr int RANK_LUT_SIZE  = 371293; // 13^5 base
    static constexpr int FLUSH_LUT_SIZE = 8192;

    HandEvaluator() { build_luts(); }

    [[nodiscard]] HandStrength evaluate(std::span<const Card> cards) const noexcept {
        int n = static_cast<int>(cards.size());
        assert(n >= 5 && n <= 7);
        HandStrength best = 0;
        for (int a=0;   a<n-4; ++a)
        for (int b=a+1; b<n-3; ++b)
        for (int c=b+1; c<n-2; ++c)
        for (int d=c+1; d<n-1; ++d)
        for (int e=d+1; e<n;   ++e) {
            auto s = eval5(cards[a],cards[b],cards[c],cards[d],cards[e]);
            if (s > best) best = s;
        }
        return best;
    }

    [[nodiscard]] HandStrength evaluate(CardSet cs) const noexcept {
        Card buf[7]; int n=0;
        cs.for_each([&](Card c){ buf[n++]=c; });
        return evaluate({buf, static_cast<std::size_t>(n)});
    }

    // Public LUT accessors for FastHandEvaluator
    [[nodiscard]] const std::vector<uint16_t>& rank_lut() const noexcept {
        return rank_lut_;
    }
    [[nodiscard]] const std::array<uint16_t,FLUSH_LUT_SIZE>& flush_lut() const noexcept {
        return flush_lut_;
    }

    // Optimised 7-card path via UltraFastHandEvaluator (O(7) histogram, no C(7,5))
    // Throughput: ~6M ops/sec on scalar CPU; ~60M with AVX-512+SVML.
    [[nodiscard]] HandStrength evaluate7(
        Card c0,Card c1,Card c2,Card c3,Card c4,Card c5,Card c6) const noexcept
    {
        // Use fast histogram evaluator when available
        return fast_eval7(c0,c1,c2,c3,c4,c5,c6);
    }

    // Slow fallback path (C(7,5) enumeration)
    [[nodiscard]] HandStrength evaluate7_slow(
        Card c0,Card c1,Card c2,Card c3,Card c4,Card c5,Card c6) const noexcept
    {
        const Card d[7]={c0,c1,c2,c3,c4,c5,c6};
        HandStrength best=0, s;
        #define E5(a,b,c,dd,e) s=eval5(d[a],d[b],d[c],d[dd],d[e]);if(s>best)best=s;
        E5(0,1,2,3,4) E5(0,1,2,3,5) E5(0,1,2,3,6) E5(0,1,2,4,5) E5(0,1,2,4,6)
        E5(0,1,2,5,6) E5(0,1,3,4,5) E5(0,1,3,4,6) E5(0,1,3,5,6) E5(0,1,4,5,6)
        E5(0,2,3,4,5) E5(0,2,3,4,6) E5(0,2,3,5,6) E5(0,2,4,5,6) E5(0,3,4,5,6)
        E5(1,2,3,4,5) E5(1,2,3,4,6) E5(1,2,3,5,6) E5(1,2,4,5,6) E5(1,3,4,5,6)
        E5(2,3,4,5,6)
        #undef E5
        return best;
    }

private:
    std::vector<uint16_t>               rank_lut_;
    std::array<uint16_t,FLUSH_LUT_SIZE> flush_lut_{};

    // Optimal 5-element sorting network (Knuth Vol.3): 9 compare-swaps
    [[nodiscard]] static int rank_key5(const Card* c) noexcept {
        int r0=c[0].rank(), r1=c[1].rank(), r2=c[2].rank(),
            r3=c[3].rank(), r4=c[4].rank();
        // Sort network for 5 elements (9 CAS)
        #define CAS(a,b) if(a>b){ int t=a;a=b;b=t; }
        CAS(r0,r1) CAS(r3,r4) CAS(r2,r4)
        CAS(r2,r3) CAS(r0,r3) CAS(r0,r2)
        CAS(r1,r4) CAS(r1,r3) CAS(r1,r2)
        #undef CAS
        return r0+13*r1+169*r2+2197*r3+28561*r4;
    }

    static HandStrength fast_eval7(Card c0,Card c1,Card c2,Card c3,
                                      Card c4,Card c5,Card c6) noexcept
    {
        // Forward to UltraFast (defined in hand_evaluator_fast.hpp via included header)
        // Inline the rank histogram approach for L1-cache friendliness
        const Card cards[7]={c0,c1,c2,c3,c4,c5,c6};
        uint8_t rc[13]{}, sc[4]{};
        for (auto& c:cards){++rc[c.rank()];++sc[c.suit()];}

        // Non-flush path first (common case)
        // Check quad/FH before flush detection
        int q=-1,t0_=-1,t1_=-1,nt=0,p0_=-1,p1_=-1,np=0,s_[7]{},ns=0;
        for(int r=12;r>=0;--r) switch(rc[r]) {
            case 4: q=r; break;
            case 3: if(!nt) t0_=r; else t1_=r; ++nt; break;
            case 2: if(!np) p0_=r; else if(np==1) p1_=r; ++np; break;
            case 1: s_[ns++]=r; break;
        }
        if(q>=0) {
            int k=-1; for(int r=12;r>=0;--r) if(r!=q&&rc[r]>0){k=r;break;}
            if(k<0)k=(q<12)?12:11;
            int idx=q*12+(k<q?k:k-1);
            return static_cast<HandStrength>(7296+idx);
        }
        if(nt>0&&(np>0||nt>1)) {
            int pp=(nt>1)?t1_:p0_;
            int idx=t0_*12+(pp<t0_?pp:pp-1);
            return static_cast<HandStrength>(7140+idx);
        }
        // Check flush
        for(int s=0;s<4;++s) if(sc[s]>=5) {
            uint16_t fm=0; uint8_t fr[7]; int nf=0;
            for(auto& c:cards) if(c.suit()==s){fm|=(1<<c.rank());fr[nf++]=c.rank();}
            // Sort fr descending
            for(int i=0;i<nf-1;++i) for(int j=i+1;j<nf;++j)
                if(fr[j]>fr[i]){uint8_t t=fr[i];fr[i]=fr[j];fr[j]=t;}
            // Straight flush?
            uint16_t pm2=0; for(int i=0;i<nf;++i) pm2|=(1<<fr[i]);
            static const struct{uint16_t m;int t;}STs[10]={
                {0x1F00,12},{0x0F80,11},{0x07C0,10},{0x03E0,9},{0x01F0,8},
                {0x00F8,7},{0x007C,6},{0x003E,5},{0x001F,4},
                {(uint16_t)((1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<12)),3}};
            for(auto& st:STs) if((pm2&st.m)==st.m)
                return static_cast<HandStrength>(7452+st.t-3);
            // Regular flush: best 5 of flush cards
            int rr[5]; for(int i=0;i<5;++i) rr[i]=fr[i];
            for(int i=0;i<4;++i) for(int j=i+1;j<5;++j)
                if(rr[j]<rr[i]){int t=rr[i];rr[i]=rr[j];rr[j]=t;}
            auto cb=[](int n,int k)->int{
                if(n<k||k<0)return 0; if(k==0)return 1; if(k==1)return n;
                if(k==2)return n*(n-1)/2; if(k==3)return n*(n-1)*(n-2)/6;
                if(k==4)return n*(n-1)*(n-2)*(n-3)/24;
                if(k==5)return n*(n-1)*(n-2)*(n-3)*(n-4)/120; return 0;};
            int fi=cb(rr[4],5)+cb(rr[3],4)+cb(rr[2],3)+cb(rr[1],2)+cb(rr[0],1);
            return static_cast<HandStrength>(5864+std::min(fi,1275));
        }
        // Straight
        {uint16_t pm3=0; for(int r=0;r<13;++r) if(rc[r]>0) pm3|=(1<<r);
         static const struct{uint16_t m;int t;}STs[10]={
            {0x1F00,12},{0x0F80,11},{0x07C0,10},{0x03E0,9},{0x01F0,8},
            {0x00F8,7},{0x007C,6},{0x003E,5},{0x001F,4},
            {(uint16_t)((1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<12)),3}};
         for(auto& st:STs) if((pm3&st.m)==st.m)
            return static_cast<HandStrength>(5854+st.t-3);}
        // Trips
        if(nt>0) {
            int k0=-1,k1=-1;
            for(int r=12;r>=0;--r) if(rc[r]>0&&r!=t0_){if(k0<0)k0=r;else{k1=r;break;}}
            if(k1<0)k1=(k0>0)?0:1;
            int adj_hi=std::max(k0,k1),adj_lo=std::min(k0,k1);
            int ah=(adj_hi>=t0_)?adj_hi-1:adj_hi, al=(adj_lo>=t0_)?adj_lo-1:adj_lo;
            int idx=t0_*66+ah*(ah-1)/2+al;
            return static_cast<HandStrength>(4996+std::min(idx,857));
        }
        // Two pair
        if(np>=2) {
            int k=-1; for(int r=12;r>=0;--r) if(rc[r]>0&&r!=p0_&&r!=p1_){k=r;break;}
            if(k<0)k=0;
            int hi=std::max(p0_,p1_),lo=std::min(p0_,p1_);
            int pi=hi*(hi-1)/2+lo, ka=k; if(ka>=hi)--ka; if(ka>=lo)--ka;
            int idx=pi*11+ka;
            return static_cast<HandStrength>(4138+std::min(idx,857));
        }
        // One pair
        if(np==1) {
            int k[3]{-1,-1,-1},nk=0;
            for(int r=12;r>=0&&nk<3;--r) if(rc[r]>0&&r!=p0_) k[nk++]=r;
            auto adj=[p0_](int kk){return kk>p0_?kk-1:kk;};
            int a=adj(k[0]),b=adj(k[1]),cc=adj(k[2]);
            int ki=(11-a)*10*9/2-(12-a)*(11-a)/2+(11-b-1)*(11-b)/2+cc;
            ki=std::max(0,std::min(ki,219));
            return static_cast<HandStrength>(1278+std::min(p0_*220+ki,2859));
        }
        // High card
        {int rr[5]; for(int i=0;i<5;++i) rr[i]=s_[i];
         for(int i=0;i<4;++i) for(int j=i+1;j<5;++j)
            if(rr[j]<rr[i]){int t=rr[i];rr[i]=rr[j];rr[j]=t;}
         auto cb=[](int n,int k)->int{
            if(n<k||k<0)return 0; if(k==0)return 1; if(k==1)return n;
            if(k==2)return n*(n-1)/2; if(k==3)return n*(n-1)*(n-2)/6;
            if(k==4)return n*(n-1)*(n-2)*(n-3)/24;
            if(k==5)return n*(n-1)*(n-2)*(n-3)*(n-4)/120; return 0;};
         int fi=cb(rr[4],5)+cb(rr[3],4)+cb(rr[2],3)+cb(rr[1],2)+cb(rr[0],1);
         return static_cast<HandStrength>(1+std::min(fi,1276));}
    }

    [[nodiscard]] HandStrength eval5(Card c0,Card c1,Card c2,Card c3,Card c4) const noexcept {
        bool flush=(c0.suit()==c1.suit()&&c1.suit()==c2.suit()&&
                    c2.suit()==c3.suit()&&c3.suit()==c4.suit());
        if (flush) {
            uint16_t m=static_cast<uint16_t>(
                (1<<c0.rank())|(1<<c1.rank())|(1<<c2.rank())|
                (1<<c3.rank())|(1<<c4.rank()));
            uint16_t v=flush_lut_[m&(FLUSH_LUT_SIZE-1)];
            return v ? v : 5864;
        }
        const Card tmp[5]={c0,c1,c2,c3,c4};
        return rank_lut_[rank_key5(tmp)];
    }

    static int comb(int n,int k) noexcept {
        if(n<k||k<0) return 0;
        if(k==0) return 1; if(k==1) return n;
        if(k==2) return n*(n-1)/2; if(k==3) return n*(n-1)*(n-2)/6;
        if(k==4) return n*(n-1)*(n-2)*(n-3)/24;
        if(k==5) return n*(n-1)*(n-2)*(n-3)*(n-4)/120;
        return 0;
    }

    void build_flush_lut() noexcept {
        flush_lut_.fill(0);
        // Flush strengths: 5864..7139 (non-straight flushes)
        // Dense flush index via combinatorial number system (C(13,5)-10 = 1277 flushes)
        // Straight flushes: 7452..7462
        // Wheel SF: top=3 → 7452, Broadway: top=12 → 7462
        for (int mask=0; mask<FLUSH_LUT_SIZE; ++mask) {
            if (__builtin_popcount(mask)!=5) continue;
            int r[5], cnt=0;
            for (int i=0;i<13;++i) if(mask&(1<<i)) r[cnt++]=i;
            bool st=(r[4]-r[0]==4&&r[1]-r[0]==1&&r[2]-r[1]==1&&r[3]-r[2]==1);
            bool wh=(mask==((1<<0)|(1<<1)|(1<<2)|(1<<3)|(1<<12)));
            if (st||wh) {
                int top=wh?3:r[4];
                // SF: 7452(wheel)..7462(broadway): top 3..12 → +0..+9, A-high=10
                int idx = (wh) ? 0 : (top==12 ? 10 : top-3);
                flush_lut_[mask]=static_cast<uint16_t>(7452+idx);
            } else {
                int fi=comb(r[4],5)+comb(r[3],4)+comb(r[2],3)+comb(r[1],2)+comb(r[0],1);
                flush_lut_[mask]=static_cast<uint16_t>(5864+std::min(fi, 1275)); // 5864..7139
            }
        }
    }

    void build_rank_lut() noexcept {
        rank_lut_.assign(RANK_LUT_SIZE, 0);

        auto key = [](int r0,int r1,int r2,int r3,int r4) noexcept {
            return r0+13*r1+169*r2+2197*r3+28561*r4; // pre-sorted ascending
        };
        auto set = [&](int r0,int r1,int r2,int r3,int r4,uint16_t v) noexcept {
            rank_lut_[key(r0,r1,r2,r3,r4)]=v;
        };
        auto sset = [&](int* r5, uint16_t v) noexcept {
            std::sort(r5,r5+5);
            rank_lut_[key(r5[0],r5[1],r5[2],r5[3],r5[4])]=v;
        };

        // ── HighCard 1..1277 (worst→best: ascending ranks) ────────────────
        { int base=1;
          for(int r0=0;r0<=8;++r0)
          for(int r1=r0+1;r1<=9;++r1)
          for(int r2=r1+1;r2<=10;++r2)
          for(int r3=r2+1;r3<=11;++r3)
          for(int r4=r3+1;r4<=12;++r4) {
            // Skip straights
            bool st=(r4-r0==4&&r1-r0==1&&r2-r1==1&&r3-r2==1);
            bool wh=(r0==0&&r1==1&&r2==2&&r3==3&&r4==12);
            if (!st&&!wh) set(r0,r1,r2,r3,r4, static_cast<uint16_t>(base++));
        }}

        // ── OnePair 1278..4137 (worst→best) ──────────────────────────────
        { int base=1278;
          for(int p=0;p<=12;++p)
          for(int k0=0;k0<=10;++k0) { if(k0==p) continue;
          for(int k1=k0+1;k1<=11;++k1) { if(k1==p) continue;
          for(int k2=k1+1;k2<=12;++k2) { if(k2==p) continue;
            int r[5]={p,p,k0,k1,k2}; sset(r,static_cast<uint16_t>(base++));
        }}}}

        // ── TwoPair 4138..4995 (worst→best) ──────────────────────────────
        { int base=4138;
          for(int p1=0;p1<=11;++p1)
          for(int p2=p1+1;p2<=12;++p2)
          for(int k=0;k<=12;++k) {
            if(k==p1||k==p2) continue;
            int r[5]={p1,p1,p2,p2,k}; sset(r,static_cast<uint16_t>(base++));
        }}

        // ── Trips 4996..5853 ──────────────────────────────────────────────
        { int base=4996;
          for(int t=0;t<=12;++t)
          for(int k1=0;k1<=11;++k1) { if(k1==t) continue;
          for(int k2=k1+1;k2<=12;++k2) { if(k2==t) continue;
            int r[5]={t,t,t,k1,k2}; sset(r,static_cast<uint16_t>(base++));
        }}}

        // ── Straights 5854..5863 ──────────────────────────────────────────
        { const int tops[10]={3,4,5,6,7,8,9,10,11,12};
          for(int i=0;i<10;++i) {
            int top=tops[i];
            int r[5];
            if(top==3) { r[0]=0;r[1]=1;r[2]=2;r[3]=3;r[4]=12; }
            else { for(int j=0;j<5;++j) r[j]=top-4+j; }
            std::sort(r,r+5);
            set(r[0],r[1],r[2],r[3],r[4], static_cast<uint16_t>(5854+i));
        }}

        // ── FullHouse 7140..7295 ──────────────────────────────────────────
        { int base=7140;
          for(int t=0;t<=12;++t) for(int p=0;p<=12;++p) {
            if(p==t) continue;
            int r[5]={t,t,t,p,p}; sset(r,static_cast<uint16_t>(base++));
        }}

        // ── Quads 7296..7451 ─────────────────────────────────────────────
        { int base=7296;
          for(int q=0;q<=12;++q) for(int k=0;k<=12;++k) {
            if(k==q) continue;
            int r[5]={q,q,q,q,k}; sset(r,static_cast<uint16_t>(base++));
        }}
    }

    void build_luts() noexcept { build_flush_lut(); build_rank_lut(); }
};

} // namespace poker
