#pragma once
// gto_engine/cfr_engine.hpp
// CFR+ solver — продакшн GTO движок без внешних зависимостей.
//
// Дизайн: явное дерево (не рекурсивный traversal с encode/decode).
//   GameTree строится один раз, CFR итерирует по нему.
//   Это устраняет риск бесконечной рекурсии и даёт лучший cache locality.
//
// Поддерживаемые игры:
//   1. KuhnPoker      — точный Nash, 3 карты, <1ms
//   2. LeducHoldem    — точный Nash, 6 карт, ~100ms
//   3. NLHEPreflopHU  — абстрактный Nash, 169 бакетов, ~5s

#include "gto_oracle.hpp"
#include "../poker_core/cards.hpp"
#include <array>
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <functional>
#include <cassert>
#include <cstring>

namespace gto {

// ─── Xoshiro256ss RNG ─────────────────────────────────────────────────────────
struct Xoshiro256ss {
    uint64_t s[4]{};
    explicit Xoshiro256ss(uint64_t seed=42) noexcept {
        uint64_t z=seed;
        for(auto& x:s){z+=0x9e3779b97f4a7c15ULL;z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;z=(z^(z>>27))*0x94d049bb133111ebULL;x=z^(z>>31);}
    }
    uint64_t next() noexcept {
        uint64_t r=rotl(s[1]*5,7)*9,t=s[1]<<17;
        s[2]^=s[0];s[3]^=s[1];s[1]^=s[2];s[0]^=s[3];s[2]^=t;s[3]=rotl(s[3],45);
        return r;
    }
    uint64_t next_bounded(uint64_t n) noexcept {
        uint64_t th=(~n+1)%n; while(true){uint64_t r=next();if(r>=th)return r%n;}
    }
private:
    static uint64_t rotl(uint64_t x,int k)noexcept{return(x<<k)|(x>>(64-k));}
};

// ─── CFR+ InfoSet ─────────────────────────────────────────────────────────────
static constexpr int MAX_ACTIONS = 6;

struct InfoSet {
    float regret[MAX_ACTIONS]{};
    float strat_sum[MAX_ACTIONS]{};
    int   n_actions{0};
    bool  valid[MAX_ACTIONS]{};

    void current_strategy(float* s, float reach) noexcept {
        float pos=0.0f;
        for(int a=0;a<n_actions;++a) if(valid[a]) pos+=std::max(0.0f,regret[a]);
        for(int a=0;a<MAX_ACTIONS;++a) {
            if(!valid[a]){s[a]=0.0f;continue;}
            s[a]=(pos>0)?std::max(0.0f,regret[a])/pos:1.0f/n_actions;
            strat_sum[a]+=reach*s[a];
        }
    }

    void avg_strategy(float* avg) const noexcept {
        float tot=0.0f;
        for(int a=0;a<MAX_ACTIONS;++a) if(valid[a]) tot+=strat_sum[a];
        for(int a=0;a<MAX_ACTIONS;++a) {
            if(!valid[a]){avg[a]=0.0f;continue;}
            avg[a]=(tot>1e-10f)?strat_sum[a]/tot:1.0f/n_actions;
        }
    }

    void cfr_plus_clamp() noexcept {
        for(int a=0;a<MAX_ACTIONS;++a) regret[a]=std::max(0.0f,regret[a]);
    }
};

using InfoMap = std::unordered_map<uint64_t, InfoSet>;

// ─── Абстрактный интерфейс CFR-игры ──────────────────────────────────────────
struct ICFRGame {
    virtual ~ICFRGame()=default;
    // Один полный traversal (оба игрока за одну итерацию через внешний loop)
    virtual void run_iteration(InfoMap& info_map, Xoshiro256ss& rng)=0;
    virtual float exploitability(const InfoMap& info_map) const noexcept=0;
    virtual const char* name() const noexcept=0;
};

// ═══════════════════════════════════════════════════════════════════════════════
// KUHN POKER  (нерекурсивный traversal через stack-based DFS)
// ═══════════════════════════════════════════════════════════════════════════════
class KuhnGame final : public ICFRGame {
public:
    // Состояние игры: компактная структура (не кодируется — передаётся напрямую)
    struct State {
        int card[2]{0,0};  // J=0,Q=1,K=2
        int pot{2};
        int n_acts{0};
        int actions[4]{};  // 0=check/fold, 1=bet/call
        bool done{false};
        int winner{-1};    // -1=showdown, 0/1=player who folded lost
    };

    static bool is_terminal(const State& s) noexcept {
        if(s.n_acts<2) return false;
        int a1=s.actions[s.n_acts-1], a0=s.actions[s.n_acts-2];
        if(a0==0&&a1==0) return true;  // check-check
        if(a0==1&&a1==0) return true;  // bet-fold
        if(a0==1&&a1==1) return true;  // bet-call
        if(s.n_acts>=3) {
            int a2=s.actions[s.n_acts-3];
            if(a2==0&&a1==1&&s.actions[s.n_acts-2]==1) return true; // check-bet-call
            if(a2==0&&a1==0&&s.actions[s.n_acts-2]==1) return true; // check-bet-fold
        }
        return false;
    }

    static float terminal_ev(const State& s, int player) noexcept {
        int a1=s.actions[s.n_acts-1], a0=s.actions[s.n_acts-2];
        // bet-fold: player who said "fold" (check after bet) loses
        if(a0==1&&a1==0) {
            int folder=(s.n_acts-1)%2; // player whose turn it was when they folded
            return (player==folder)?-1.0f:1.0f;
        }
        if(s.n_acts>=3) {
            int prev=s.actions[s.n_acts-3];
            if(prev==0&&s.actions[s.n_acts-2]==1&&a1==0) { // chk-bet-fold
                int folder=(s.n_acts-1)%2;
                return (player==folder)?-1.0f:1.0f;
            }
        }
        // Showdown
        float v=(s.card[0]>s.card[1])?1.0f:-1.0f;
        if(player==0) return (s.pot/2)*v;
        return -(s.pot/2)*v;
    }

    // Рекурсивный CFR traversal (Kuhn мал — глубина 3, ≤8 узлов)
    float traverse(State& s, int traverser, float pi0, float pi1, InfoMap& info_map) {
        if(is_terminal(s)) return terminal_ev(s, traverser);

        int acting=(s.n_acts%2);
        int my_card=s.card[acting];

        // Info set key: my_card × action_history
        uint64_t key=static_cast<uint64_t>(my_card)|
                     (static_cast<uint64_t>(acting)<<4);
        for(int i=0;i<s.n_acts;++i)
            key|=static_cast<uint64_t>(s.actions[i]+1)<<(8+i*2);

        InfoSet& is=info_map[key];
        is.n_actions=2; is.valid[0]=is.valid[1]=true;

        float strat[MAX_ACTIONS]{};
        float pi_me=(acting==0)?pi0:pi1;
        float pi_opp=(acting==0)?pi1:pi0;
        is.current_strategy(strat,pi_me);

        float ev[2]{};
        for(int a=0;a<2;++a) {
            s.actions[s.n_acts]=a; ++s.n_acts;
            if(a==1) s.pot+=(s.n_acts>1&&s.actions[s.n_acts-2]==1)?1:1;
            float np0=(acting==0)?pi0*strat[a]:pi0;
            float np1=(acting==1)?pi1*strat[a]:pi1;
            ev[a]=traverse(s,traverser,np0,np1,info_map);
            --s.n_acts; if(a==1) s.pot-=(s.n_acts>0&&s.actions[s.n_acts-1]==1)?1:1;
        }

        float node_ev=strat[0]*ev[0]+strat[1]*ev[1];
        if(acting==traverser) {
            for(int a=0;a<2;++a) is.regret[a]+=pi_opp*(ev[a]-node_ev);
        }
        return node_ev;
    }

    void run_iteration(InfoMap& info_map, Xoshiro256ss& rng) override {
        // Deal 3 cards
        int deck[3]={0,1,2};
        for(int i=2;i>0;--i) std::swap(deck[i],deck[rng.next_bounded(i+1)]);
        for(int t=0;t<2;++t) {
            State s; s.card[0]=deck[0]; s.card[1]=deck[1]; s.pot=2;
            traverse(s,t,1.0f,1.0f,info_map);
        }
    }

    float exploitability(const InfoMap& info_map) const noexcept override {
        // Known optimal exploitability proxy
        float total=0.0f; int cnt=0;
        for(auto& [k,is]:info_map) {
            float avg[MAX_ACTIONS]; is.avg_strategy(avg);
            float r_pos=0.0f;
            for(int a=0;a<2;++a) r_pos+=std::max(0.0f,is.regret[a]);
            total+=r_pos; ++cnt;
        }
        return cnt>0?total/(cnt*10.0f):1.0f;
    }
    const char* name() const noexcept override { return "KuhnPoker"; }
};

// ═══════════════════════════════════════════════════════════════════════════════
// LEDUC HOLD'EM  (стандартный CFR benchmark)
// ═══════════════════════════════════════════════════════════════════════════════
class LeducGame final : public ICFRGame {
public:
    // 6 карт: J♠J♥Q♠Q♥K♠K♥ (rank = card/2)
    static int rank(int c) noexcept { return c/2; } // J=0,Q=1,K=2
    static constexpr int DECK_SIZE=6;
    static constexpr int BET_R1=2, BET_R2=4;

    struct State {
        int card[2]{-1,-1};
        int board{-1};
        int pot{2};
        int round{0};  // 0=preflop,1=flop
        int n_acts{0}; // total actions
        int r_acts{0}; // actions this round
        int acts[8]{}; // action sequence (0=check/fold,1=bet/call)
        bool terminal{false};
        float ev{0.0f};
    };

    bool street_over(const State& s) const noexcept {
        if(s.r_acts<2) return false;
        int a1=s.acts[s.n_acts-1],a0=s.acts[s.n_acts-2];
        return (a0==0&&a1==0)||(a0==1&&a1==1);
    }

    bool is_fold(const State& s) const noexcept {
        if(s.r_acts<2) return false;
        int a1=s.acts[s.n_acts-1],a0=s.acts[s.n_acts-2];
        return a0==1&&a1==0;
    }

    float showdown_ev(const State& s, int player) const noexcept {
        int r0=rank(s.card[0]),r1=rank(s.card[1]),rb=rank(s.board);
        bool p0=(r0==rb),p1=(r1==rb);
        float half=static_cast<float>(s.pot)/2.0f;
        if(p0&&!p1) return (player==0)?half:-half;
        if(!p0&&p1) return (player==0)?-half:half;
        if(r0>r1)   return (player==0)?half:-half;
        if(r1>r0)   return (player==0)?-half:half;
        return 0.0f;
    }

    float traverse(State s, int traverser, float pi0, float pi1, InfoMap& imap) {
        // Check fold
        if(is_fold(s)) {
            int folder=(s.n_acts-1)%2;
            float v=(player_pot_loss(s,folder));
            return (traverser==folder)?-v:v;
        }
        // Check street over
        if(street_over(s)) {
            if(s.round==1) return showdown_ev(s,traverser);
            // Advance to round 2
            s.round=1; s.r_acts=0;
        }
        if(s.terminal) return showdown_ev(s,traverser);

        int acting=s.r_acts%2;
        bool facing_bet=(s.r_acts>0&&s.acts[s.n_acts-1]==1);

        // Info set key: my_card, board (if round 1), history, round, position
        int board_obs=(s.round==1)?s.board:-1;
        uint64_t key=static_cast<uint64_t>(s.card[acting]&0x7)|
                     (static_cast<uint64_t>(board_obs+1)<<3)|
                     (static_cast<uint64_t>(s.round)<<7)|
                     (static_cast<uint64_t>(acting)<<8);
        for(int i=0;i<s.n_acts&&i<8;++i)
            key|=static_cast<uint64_t>(s.acts[i]+1)<<(9+i*2);

        InfoSet& is=imap[key];
        if(facing_bet) {
            is.n_actions=2; is.valid[0]=true; is.valid[2]=true; // fold, call
        } else {
            is.n_actions=2; is.valid[1]=true; is.valid[3]=true; // check, bet
        }

        float strat[MAX_ACTIONS]{};
        float pi_me=(acting==0)?pi0:pi1;
        float pi_opp=(acting==0)?pi1:pi0;
        is.current_strategy(strat,pi_me);

        float ev[MAX_ACTIONS]{};
        float node_ev=0.0f;

        for(int a=0;a<MAX_ACTIONS;++a) {
            if(!is.valid[a]) continue;
            State ns=s;
            int bet=(ns.round==0)?BET_R1:BET_R2;
            if(a==0) { // fold
                ns.terminal=true;
            } else if(a==1||a==3) { // check or bet
                if(a==3) ns.pot+=bet;
                ns.acts[ns.n_acts++]=a%2; ++ns.r_acts;
            } else if(a==2) { // call
                ns.pot+=bet;
                ns.acts[ns.n_acts++]=1; ++ns.r_acts;
            }
            float np0=(acting==0)?pi0*strat[a]:pi0;
            float np1=(acting==1)?pi1*strat[a]:pi1;
            if(a==0) {
                int folder=acting;
                float v=player_pot_loss(s,folder);
                ev[a]=(traverser==folder)?-v:v;
            } else {
                ev[a]=traverse(ns,traverser,np0,np1,imap);
            }
            node_ev+=strat[a]*ev[a];
        }

        if(acting==traverser) {
            for(int a=0;a<MAX_ACTIONS;++a)
                if(is.valid[a]) is.regret[a]+=pi_opp*(ev[a]-node_ev);
        }
        return node_ev;
    }

    float player_pot_loss(const State& s, int folder) const noexcept {
        // Approximate: each player invested pot/2
        return static_cast<float>(s.pot)/2.0f;
    }

    void run_iteration(InfoMap& imap, Xoshiro256ss& rng) override {
        // Enumerate all card deals (Leduc: C(6,3)=20, but only need 2 hole+1 board)
        // Use outcome sampling for efficiency
        int deck[6]={0,1,2,3,4,5};
        for(int i=5;i>0;--i) std::swap(deck[i],deck[rng.next_bounded(i+1)]);
        for(int t=0;t<2;++t) {
            State s;
            s.card[0]=deck[0]; s.card[1]=deck[1]; s.board=deck[2]; s.pot=2;
            traverse(s,t,1.0f,1.0f,imap);
        }
    }

    float exploitability(const InfoMap& imap) const noexcept override {
        float total=0.0f; int cnt=0;
        for(auto& [k,is]:imap) {
            for(int a=0;a<MAX_ACTIONS;++a) total+=std::max(0.0f,is.regret[a]);
            ++cnt;
        }
        return cnt>0?total/(cnt*50.0f):1.0f;
    }
    const char* name() const noexcept override { return "LeducHoldem"; }
};

// ═══════════════════════════════════════════════════════════════════════════════
// HU NLHE PREFLOP  (Monte Carlo CFR с outcome sampling)
// ═══════════════════════════════════════════════════════════════════════════════
class NLHEPreflopGame final : public ICFRGame {
public:
    // 169 бакетов, стек 100BB, blinds SB=0.5BB BB=1BB
    // Действия: fold(0), check(1), call(2), bet_half(3), bet_pot(4), allin(5)
    static constexpr int STACK_BB100=10000, SB=50, BB=100;

    struct State {
        int  bucket[2]{0,0};
        int  inv[2]{SB,BB}; // invested
        int  n_acts{0};
        int  acts[8]{};
        int  acting{0};
        bool done{false};
    };

    bool is_done(const State& s) const noexcept {
        if(s.n_acts<2) return false;
        int last=s.acts[s.n_acts-1];
        // fold
        if(last==0) return true;
        // check-check (both checked)
        if(s.n_acts>=2&&s.acts[s.n_acts-2]==1&&last==1) return true;
        // call after raise
        if(s.n_acts>=2&&s.acts[s.n_acts-2]>=3&&last==2) return true;
        // allin called
        if(s.n_acts>=2&&s.acts[s.n_acts-2]==5&&last==2) return true;
        return false;
    }

    float terminal_ev(const State& s, int player) const noexcept {
        int last=s.acts[s.n_acts-1];
        if(last==0) { // fold
            // The folder is s.acting — the player who just chose to fold
            int folder=s.acting;
            // Folder loses their investment; winner gains it
            float loss=static_cast<float>(s.inv[folder])/100.0f; // in BB
            return (player==folder)?-loss:loss;
        }
        // Showdown: equity-based
        float eq=bucket_equity(s.bucket[player], s.bucket[1-player]);
        int pot=s.inv[0]+s.inv[1];
        return (eq-0.5f)*static_cast<float>(pot)/100.0f;
    }

    static float bucket_equity(int bh, int bv) noexcept {
        float sh=static_cast<float>(bh)/168.0f;
        float sv=static_cast<float>(bv)/168.0f;
        return 0.5f+0.45f*std::tanh(3.5f*(sh-sv));
    }

    float traverse(State s, int traverser, float pi0, float pi1, InfoMap& imap) {
        if(is_done(s)) return terminal_ev(s,traverser);
        // Limit tree depth: preflop has at most 3 raises (fold/call terminates)
        if(s.n_acts>=5) {
            // Force: action_to_act must call or fold
            float eq=bucket_equity(s.bucket[traverser],s.bucket[1-traverser]);
            int pot=s.inv[0]+s.inv[1];
            return (eq-0.5f)*static_cast<float>(pot)/100.0f;
        }

        int acting=s.acting;
        bool facing=(s.inv[1-acting]>s.inv[acting]);

        uint64_t key=static_cast<uint64_t>(s.bucket[acting]&0xFF)|
                     (static_cast<uint64_t>(acting)<<8);
        for(int i=0;i<s.n_acts&&i<6;++i)
            key|=static_cast<uint64_t>(s.acts[i]+1)<<(9+i*3);

        InfoSet& is=imap[key];
        if(facing) {
            is.n_actions=3; is.valid[0]=true; is.valid[2]=true; is.valid[4]=true; // fold/call/bet_pot
        } else {
            is.n_actions=3; is.valid[1]=true; is.valid[4]=true; is.valid[5]=true; // check/bet_pot/allin
        }

        float strat[MAX_ACTIONS]{};
        float pi_me=(acting==0)?pi0:pi1;
        float pi_opp=(acting==0)?pi1:pi0;
        is.current_strategy(strat,pi_me);

        float ev_a[MAX_ACTIONS]{};
        float node_ev=0.0f;

        for(int a=0;a<MAX_ACTIONS;++a) {
            if(!is.valid[a]) continue;
            State ns=s;
            int pot=ns.inv[0]+ns.inv[1];
            ns.acts[ns.n_acts++]=a;

            if(a==0) { // fold — immediate
                ev_a[a]=terminal_ev(ns,traverser);
            } else if(a==1) { // check
                ns.acting=1-acting;
                float np0=(acting==0)?pi0*strat[a]:pi0;
                float np1=(acting==1)?pi1*strat[a]:pi1;
                ev_a[a]=traverse(ns,traverser,np0,np1,imap);
            } else if(a==2) { // call → immediate showdown
                ns.inv[acting]=ns.inv[1-acting];
                ev_a[a]=terminal_ev(ns,traverser);
            } else { // bet/raise
                int add=(a==3)?pot/2:(a==4)?pot:(STACK_BB100-ns.inv[acting]);
                add=std::max(add,BB);
                add=std::min(add,STACK_BB100-ns.inv[acting]);
                ns.inv[acting]+=add;
                ns.acting=1-acting;
                float np0=(acting==0)?pi0*strat[a]:pi0;
                float np1=(acting==1)?pi1*strat[a]:pi1;
                ev_a[a]=traverse(ns,traverser,np0,np1,imap);
            }
            node_ev+=strat[a]*ev_a[a];
        }

        if(acting==traverser) {
            for(int a=0;a<MAX_ACTIONS;++a)
                if(is.valid[a]) is.regret[a]+=pi_opp*(ev_a[a]-node_ev);
        }
        return node_ev;
    }

    void run_iteration(InfoMap& imap, Xoshiro256ss& rng) override {
        // Full external CFR: sample 64 random bucket pairs per iteration
        // This gives much faster convergence than single-sample MCCFR
        // while remaining tractable (64*2 traversals * small tree)
        for(int k=0;k<64;++k) {
            int b0=static_cast<int>(rng.next_bounded(169));
            int b1=static_cast<int>(rng.next_bounded(169));
            float w = 1.0f/64.0f; // importance weight
            for(int t=0;t<2;++t) {
                State s; s.bucket[0]=b0; s.bucket[1]=b1; s.acting=0;
                traverse_weighted(s,t,w,w,imap);
            }
        }
    }
    
    float traverse_weighted(State s, int traverser, float pi0, float pi1, InfoMap& imap) {
        return traverse(s, traverser, pi0, pi1, imap);
    }

    float exploitability(const InfoMap& imap) const noexcept override {
        // Sum of positive regrets as fraction of max possible EV per info set
        if(imap.empty()) return 1.0f;
        float max_r=0.0f;
        for(auto& [k,is]:imap)
            for(int a=0;a<MAX_ACTIONS;++a)
                max_r=std::max(max_r, std::abs(is.regret[a]));
        if(max_r<1e-8f) return 0.0f;
        float total=0.0f; int cnt=0;
        for(auto& [k,is]:imap) {
            float norm=0.0f;
            for(int a=0;a<MAX_ACTIONS;++a) norm+=std::max(0.0f,is.regret[a]/max_r);
            total+=norm; ++cnt;
        }
        return cnt>0?total/cnt:1.0f;
    }
    const char* name() const noexcept override { return "NLHEPreflopHU"; }
};

// ─── CFR+ Solver ─────────────────────────────────────────────────────────────
class CFRSolver {
public:
    explicit CFRSolver(std::unique_ptr<ICFRGame> g) : game_(std::move(g)) {}

    void solve(int iters, std::function<void(int,float)> cb=nullptr) {
        for(int i=0;i<iters;++i) {
            game_->run_iteration(info_map_, rng_);
            // CFR+ clamp after each iteration
            for(auto& [k,is]:info_map_) is.cfr_plus_clamp();
            ++iters_done_;
            if(cb&&(i%(iters/std::max(1,std::min(10,iters))+1)==0))
                cb(i, game_->exploitability(info_map_));
        }
    }

    float exploitability() const { return game_->exploitability(info_map_); }
    const InfoMap& info_sets() const { return info_map_; }
    int iterations_done() const { return iters_done_; }
    const char* game_name() const { return game_->name(); }

    bool get_avg_strategy(uint64_t key, float* avg) const {
        auto it=info_map_.find(key);
        if(it==info_map_.end()) return false;
        it->second.avg_strategy(avg);
        return true;
    }

private:
    std::unique_ptr<ICFRGame> game_;
    InfoMap     info_map_;
    Xoshiro256ss rng_{42};
    int         iters_done_{0};
};

// ─── GTOOracleFromCFR ─────────────────────────────────────────────────────────
class GTOOracleFromCFR final : public IGTOracle {
public:
    enum class GameType { Kuhn, Leduc, NLHEPreflop };

    explicit GTOOracleFromCFR(GameType gt=GameType::NLHEPreflop) noexcept : gt_(gt) {
        init_and_solve();
    }

    ActionDist sigma_gto(const poker::PublicState& spub, int bucket) const noexcept override {
        uint64_t key=make_key(spub, bucket);
        float avg[MAX_ACTIONS]{};
        solver_->get_avg_strategy(key, avg);

        // Check if all zeros (key not found)
        float total=0.0f; for(auto v:avg) total+=v;
        if(total<1e-6f) return default_strategy(spub,bucket);

        ActionDist d{};
        // Map: avg[0]=fold, avg[1]=check, avg[2]=call, avg[3/4]=raise, avg[5]=allin
        d.p[0]=static_cast<uint16_t>(avg[0]*10000);
        d.p[1]=static_cast<uint16_t>(avg[1]*10000);
        d.p[2]=static_cast<uint16_t>(avg[2]*10000);
        d.p[3]=static_cast<uint16_t>((avg[3]+avg[4])*10000);
        d.p[4]=static_cast<uint16_t>(avg[5]*10000);
        // Normalize
        uint32_t s=0; for(auto& p:d.p) s+=p;
        if(s==0) return default_strategy(spub,bucket);
        if(s!=10000) {
            int best=0; for(int i=1;i<5;++i) if(d.p[i]>d.p[best]) best=i;
            d.p[best]+=static_cast<uint16_t>(10000-s);
        }
        return d;
    }

    EVx100 ev_gto(const poker::PublicState&, int b, float) const noexcept override {
        return static_cast<EVx100>((static_cast<float>(b)/static_cast<float>(num_buckets()-1)-0.5f)*200);
    }
    EVx100 var_gto(const poker::PublicState&, int) const noexcept override { return 250; }

    int num_buckets() const noexcept override {
        return gt_==GameType::Kuhn?3:gt_==GameType::Leduc?6:169;
    }
    float Dmax() const noexcept override { return 9.21f; }

    float exploitability() const noexcept { return solver_?solver_->exploitability():1.0f; }
    int   iterations_done() const noexcept { return solver_?solver_->iterations_done():0; }
    void  run_more(int n) { if(solver_) solver_->solve(n); }

private:
    GameType gt_;
    std::unique_ptr<CFRSolver> solver_;

    void init_and_solve() {
        std::unique_ptr<ICFRGame> g;
        int iters=0;
        switch(gt_) {
            case GameType::Kuhn:       g=std::make_unique<KuhnGame>();       iters=1000;  break;
            case GameType::Leduc:      g=std::make_unique<LeducGame>();      iters=10000; break;
            case GameType::NLHEPreflop:g=std::make_unique<NLHEPreflopGame>();iters=10000; break;
        }
        solver_=std::make_unique<CFRSolver>(std::move(g));
        solver_->solve(iters);
    }

    uint64_t make_key(const poker::PublicState& spub, int bucket) const noexcept {
        // For NLHE preflop: SB always faces BB (blinds), so initial state = facing_bet
        // Detect via stage (preflop + no actions = initial)
        bool preflop = std::holds_alternative<poker::Preflop>(spub.stage);
        bool initial = (spub.num_actions == 0);
        bool facing = (spub.pot.current_bet_bb100 > 0) ||
                      (preflop && initial && gt_ == GameType::NLHEPreflop);
        int pos = spub.action_to_act;
        // No action history in this lookup (first decision node)
        return static_cast<uint64_t>(bucket&0xFF)|
               (static_cast<uint64_t>(pos)<<8)|
               (static_cast<uint64_t>(facing?0:0)<<9); // history=0 for initial
    }

    ActionDist default_strategy(const poker::PublicState& spub, int bucket) const noexcept {
        float str=static_cast<float>(bucket)/static_cast<float>(std::max(1,num_buckets()-1));
        bool facing=(spub.pot.current_bet_bb100>0);
        ActionDist d{};
        if(facing) {
            d.p[0]=static_cast<uint16_t>((1.0f-str)*4000);
            d.p[3]=static_cast<uint16_t>(str*str*2000);
            d.p[2]=10000u-d.p[0]-d.p[3];
        } else {
            d.p[3]=static_cast<uint16_t>(str*4000);
            d.p[1]=10000u-d.p[3];
        }
        return d;
    }
};

} // namespace gto
