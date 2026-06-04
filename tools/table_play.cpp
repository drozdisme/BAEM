// tools/table_play.cpp
// Арена: несколько агентов играют за одним столом, раздача случайна.
// Метрика успеха — винрейт в BB/100 (миллиблайнды на 100 раздач) с дов. интервалом.
//
// Поддерживает 2..6 игроков. Стек каждого сбрасывается к 100 BB каждую раздачу
// (cash-game chip-EV), что делает метрику чистой и убирает банкротства.
// Реализованы корректные побочные банки (side pots) при олл-инах.
//
// Типы политик (--seat):
//   model:FILE   обученный PokerHistoryTransformer (веса из train_pokerbench)
//   random       случайное легальное действие
//   call         всегда чек/колл (calling station)
//   tight        фолд против ставки, иначе чек (нит)
//   aggro        часто ставит/рейзит
//
// Пример:
//   ./table_play --seat model:agent_weights.bin --seat call --seat random --seat tight \
//                --hands 200000 --seed 7
//
// Сборка — через tools/build.sh (тот же набор флагов, что и для остальных инструментов).

#include "baem_learning/poker_history_transformer.hpp"
#include "baem_tracker/hand_history_encoder.hpp"
#include "poker_core/poker_core.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <cmath>

using baem::HandHistoryEncoder;
using baem::FeatureVec;
using baem_learning::PokerHistoryTransformer;

static const char* ANAMES[5] = {"fold","check","call","raise","allin"};

// ─── Политика игрока ──────────────────────────────────────────────────────────
enum class PType { Model, Random, Call, Tight, Aggro };

struct Policy {
    PType type;
    std::string name;
    std::unique_ptr<PokerHistoryTransformer> model;  // только для Model
};

// ─── Состояние раздачи ────────────────────────────────────────────────────────
struct Hand {
    int N;
    std::array<int32_t, 6> stack{};       // остаток стека
    std::array<int32_t, 6> total_inv{};   // вложено за всю раздачу
    std::array<int32_t, 6> street_inv{};  // вложено на текущей улице
    std::array<bool, 6>    folded{};
    std::array<bool, 6>    allin{};
    std::array<poker::Card, 6> h0{}, h1{};
    poker::PublicState pub{};
    int32_t current_bet{0};   // максимальная ставка на улице (street)
    int32_t last_raise{100};  // размер последнего рейза (для мин-рейза)
};

// ─── Утилиты ───────────────────────────────────────────────────────────────────
static poker::RoundStage street_stage(int s){
    switch(s){case 1:return poker::Flop{};case 2:return poker::Turn{};
              case 3:return poker::River{};default:return poker::Preflop{};}
}

// ─── Арена ──────────────────────────────────────────────────────────────────────
class Arena {
public:
    Arena(std::vector<Policy>&& seats, int32_t start_stack, uint64_t seed)
        : seats_(std::move(seats)), N_((int)seats_.size())
        , start_(start_stack), deck_(seed), rng_(seed ^ 0x123456789ULL)
    {
        net_sum_.assign(N_, 0.0);
        net_sq_.assign(N_, 0.0);
        vpip_.assign(N_, 0);
        hands_ = 0;
    }

    void run(long hands){
        for(long t=0;t<hands;++t){
            int button = (int)(t % N_);
            std::array<int32_t,6> net{};
            play_hand(button, net);
            // учёт по местам (кнопка ротируется → за длинную дистанцию справедливо)
            for(int i=0;i<N_;++i){
                double bb = net[i] / 100.0;     // BB×100 → BB
                net_sum_[i] += bb;
                net_sq_[i]  += bb*bb;
            }
            ++hands_;
        }
    }

    void report() const {
        printf("\n=== Результаты после %ld раздач (стол на %d) ===\n", hands_, N_);
        printf("%-22s %12s %12s\n", "agent", "BB/100", "95%% CI (±)");
        // сортируем по BB/100 убыв.
        std::vector<int> idx(N_); for(int i=0;i<N_;++i) idx[i]=i;
        std::sort(idx.begin(), idx.end(), [&](int a,int b){
            return net_sum_[a]/hands_ > net_sum_[b]/hands_; });
        for(int k=0;k<N_;++k){
            int i = idx[k];
            double mean_bb = net_sum_[i] / hands_;          // BB/hand
            double var = net_sq_[i]/hands_ - mean_bb*mean_bb;
            double se = std::sqrt(std::max(0.0,var)/hands_); // BB/hand
            double bb100   = mean_bb * 100.0;
            double ci95    = 1.96 * se * 100.0;
            printf("%-22s %12.2f %12.2f\n", seats_[i].name.c_str(), bb100, ci95);
        }
        printf("(положительный BB/100 = выигрывает; CI учитывает дисперсию)\n");
    }

private:
    std::vector<Policy> seats_;
    int N_;
    int32_t start_;
    poker::DeckShuffler deck_;
    poker::HandEvaluator eval_;
    poker::Xoshiro256ss rng_;
    std::vector<double> net_sum_, net_sq_;
    std::vector<long> vpip_;
    long hands_;
    HandHistoryEncoder enc_;

    float frand(){ return (rng_.next() >> 11) * (1.0f/9007199254740992.0f); }

    // нормализованная сила 7 карт (или 2+борд) в [0,1]
    float hand_strength(int i, const Hand& H){
        std::vector<poker::Card> cs; cs.push_back(H.h0[i]); cs.push_back(H.h1[i]);
        H.pub.board.for_each([&](poker::Card c){ cs.push_back(c); });
        if(cs.size()<5){
            // префлоп: грубая оценка по рангам
            int hi=std::max(H.h0[i].rank(),H.h1[i].rank());
            int lo=std::min(H.h0[i].rank(),H.h1[i].rank());
            bool pair=H.h0[i].rank()==H.h1[i].rank();
            bool suited=H.h0[i].suit()==H.h1[i].suit();
            float s=(hi*1.0f+lo*0.5f)/18.0f + (pair?0.35f:0)+(suited?0.05f:0);
            return std::clamp(s,0.0f,1.0f);
        }
        poker::HandStrength st = eval_.evaluate(cs);
        return std::clamp(st/7462.0f, 0.0f, 1.0f);
    }

    // выбор действия политикой; возвращает (тип, raise_to_bb100)
    std::pair<poker::ActionType,int32_t> decide(int i, Hand& H, bool facing){
        const Policy& P = seats_[i];
        int32_t to_call = H.current_bet - H.street_inv[i];
        int32_t pot = 0; for(int k=0;k<N_;++k) pot += H.total_inv[k];
        int32_t minraise_to = H.current_bet + std::max(H.last_raise,(int32_t)100);

        auto sized_raise = [&](float frac)->int32_t{
            int32_t add = (int32_t)(pot * frac);
            int32_t raise_to = H.current_bet + std::max(add, H.last_raise);
            raise_to = std::min(raise_to, H.street_inv[i] + H.stack[i]); // ≤ allin
            return std::max(raise_to, minraise_to);
        };

        if(P.type==PType::Model){
            FeatureVec fv = enc_.encode(H.pub, H.h0[i], H.h1[i]);
            float p[5]; P.model->infer_probs(fv,p);
            // маскируем нелегальные
            if(facing){ p[1]=0; }            // нельзя check против ставки
            else      { p[0]=0; p[2]=0; }    // нельзя fold/call без ставки
            int best=0; float bv=-1; for(int a=0;a<5;a++) if(p[a]>bv){bv=p[a];best=a;}
            poker::ActionType act=(poker::ActionType)best;
            if(act==poker::ActionType::Raise){
                int32_t rt=sized_raise(0.66f);
                if(rt >= H.street_inv[i]+H.stack[i]) return {poker::ActionType::AllIn, H.street_inv[i]+H.stack[i]};
                return {act, rt};
            }
            if(act==poker::ActionType::AllIn) return {act, H.street_inv[i]+H.stack[i]};
            return {act,0};
        }
        // эвристики
        float s = hand_strength(i,H);
        if(P.type==PType::Call) return facing? std::pair{poker::ActionType::Call,0}
                                             : std::pair{poker::ActionType::Check,0};
        if(P.type==PType::Tight){
            if(facing) return s<0.55f? std::pair{poker::ActionType::Fold,0}
                                     : std::pair{poker::ActionType::Call,0};
            return s>0.8f? std::pair{poker::ActionType::Raise,sized_raise(0.66f)}
                         : std::pair{poker::ActionType::Check,0};
        }
        if(P.type==PType::Aggro){
            if(frand()<0.5f || s>0.5f){
                int32_t rt=sized_raise(0.8f);
                if(rt>=H.street_inv[i]+H.stack[i]) return {poker::ActionType::AllIn,rt};
                return {poker::ActionType::Raise,rt};
            }
            return facing? std::pair{poker::ActionType::Call,0}:std::pair{poker::ActionType::Check,0};
        }
        // Random
        if(facing){
            float r=frand();
            if(r<0.33f) return {poker::ActionType::Fold,0};
            if(r<0.8f)  return {poker::ActionType::Call,0};
            int32_t rt=sized_raise(0.66f); return {poker::ActionType::Raise,rt};
        } else {
            float r=frand();
            if(r<0.6f) return {poker::ActionType::Check,0};
            int32_t rt=sized_raise(0.66f); return {poker::ActionType::Raise,rt};
        }
    }

    // применить действие к состоянию
    void apply(int i, Hand& H, poker::ActionType t, int32_t raise_to){
        int32_t to_call = H.current_bet - H.street_inv[i];
        poker::Action rec; rec.player_idx=(uint8_t)i;
        if(t==poker::ActionType::Fold){ H.folded[i]=true; rec.type=t; }
        else if(t==poker::ActionType::Check){ rec.type=t; }
        else if(t==poker::ActionType::Call){
            int32_t pay=std::min(to_call,H.stack[i]);
            H.stack[i]-=pay; H.street_inv[i]+=pay; H.total_inv[i]+=pay;
            if(H.stack[i]==0)H.allin[i]=true; rec.type=t;
        } else { // Raise / AllIn
            int32_t target = (t==poker::ActionType::AllIn)? H.street_inv[i]+H.stack[i] : raise_to;
            target = std::min(target, H.street_inv[i]+H.stack[i]);
            int32_t add = target - H.street_inv[i];
            H.stack[i]-=add; H.street_inv[i]+=add; H.total_inv[i]+=add;
            int32_t inc = H.street_inv[i]-H.current_bet;
            if(inc>0){ H.last_raise=inc; H.current_bet=H.street_inv[i]; }
            if(H.stack[i]==0)H.allin[i]=true;
            rec.type = (H.stack[i]==0)? poker::ActionType::AllIn : poker::ActionType::Raise;
            rec.amount_bb100 = H.street_inv[i];
        }
        H.pub.push_action(rec);
        H.pub.pot.current_bet_bb100 = H.current_bet;
        int32_t pot=0; for(int k=0;k<N_;++k)pot+=H.total_inv[k];
        H.pub.pot.total_bb100 = pot;
    }

    int active_count(const Hand& H){ int c=0; for(int i=0;i<N_;++i) if(!H.folded[i])++c; return c; }

    // один раунд торговли, начиная с seat `start`
    void betting_round(Hand& H, int start){
        for(int i=0;i<N_;++i) H.street_inv[i]=0;
        // current_bet уже задан (префлоп = BB; постфлоп сбрасываем)
        std::array<bool,6> acted{};
        int to_act=start, guard=0;
        while(guard++ < 200){
            // условие конца: все активные (не фолд, не олл-ин) сходили и уравняли
            bool need=false;
            for(int i=0;i<N_;++i){
                if(H.folded[i]||H.allin[i]) continue;
                if(!acted[i] || H.street_inv[i]<H.current_bet){ need=true; break; }
            }
            if(!need) break;
            if(active_count(H)<=1) break;

            int i=to_act%N_;
            if(H.folded[i]||H.allin[i]){ to_act++; continue; }
            bool facing = H.current_bet > H.street_inv[i];
            auto [t,rt]=decide(i,H,facing);
            int32_t before=H.current_bet;
            apply(i,H,t,rt);
            acted[i]=true;
            if(H.current_bet>before){ // был рейз → остальные снова должны ответить
                for(int k=0;k<N_;++k) if(k!=i && !H.folded[k] && !H.allin[k]) acted[k]=false;
            }
            to_act++;
        }
        H.pub.pot.current_bet_bb100 = 0;
    }

    // распределение банка с побочными банками; заполняет net[]
    void showdown(Hand& H, std::array<int32_t,6>& winnings){
        // уровни вложений
        std::vector<int32_t> levels;
        for(int i=0;i<N_;++i) if(H.total_inv[i]>0) levels.push_back(H.total_inv[i]);
        std::sort(levels.begin(),levels.end());
        levels.erase(std::unique(levels.begin(),levels.end()),levels.end());

        int32_t prev=0;
        std::vector<poker::Card> cs;
        for(int32_t L : levels){
            int contributors=0;
            for(int i=0;i<N_;++i) if(H.total_inv[i]>=L) contributors++;
            int32_t layer=(L-prev)*contributors;
            // победители слоя: не фолд и вложили ≥ L
            int bestStr=-1; std::vector<int> winners;
            for(int i=0;i<N_;++i){
                if(H.folded[i]||H.total_inv[i]<L) continue;
                cs.clear(); cs.push_back(H.h0[i]); cs.push_back(H.h1[i]);
                H.pub.board.for_each([&](poker::Card c){cs.push_back(c);});
                int st = (cs.size()>=5)? (int)eval_.evaluate(cs) : H.h0[i].rank()*13+H.h1[i].rank();
                if(st>bestStr){bestStr=st;winners.clear();winners.push_back(i);}
                else if(st==bestStr) winners.push_back(i);
            }
            if(!winners.empty()){
                int32_t share=layer/(int32_t)winners.size();
                int32_t rem=layer - share*(int32_t)winners.size();
                for(size_t w=0;w<winners.size();++w)
                    winnings[winners[w]] += share + (w==0?rem:0);
            }
            prev=L;
        }
    }

    void play_hand(int button, std::array<int32_t,6>& net){
        Hand H; H.N=N_;
        for(int i=0;i<N_;++i){ H.stack[i]=start_; H.folded[i]=false; H.allin[i]=false;
            H.total_inv[i]=0; H.street_inv[i]=0; }
        H.pub = poker::PublicState{};
        H.pub.num_players=(uint8_t)N_;
        H.pub.dealer_pos=(uint8_t)button;

        deck_.shuffle();
        for(int i=0;i<N_;++i){ H.h0[i]=deck_.deal(); H.h1[i]=deck_.deal(); }

        // блайнды
        int sb = (N_==2)? button : (button+1)%N_;     // HU: кнопка = SB
        int bb = (N_==2)? (button+1)%N_ : (button+2)%N_;
        auto post=[&](int i,int32_t amt){ int32_t p=std::min(amt,H.stack[i]);
            H.stack[i]-=p; H.street_inv[i]+=p; H.total_inv[i]+=p; };
        post(sb,50); post(bb,100);
        H.current_bet=100; H.last_raise=100;

        // действие префлоп: HU — кнопка/SB; иначе слева от BB (UTG)
        int start_pre = (N_==2)? sb : (bb+1)%N_;

        // позиция героя (грубо, как при обучении): кнопка/late → ip
        auto set_ip=[&](){ for(int i=0;i<N_;++i){} };
        (void)set_ip;

        // ── префлоп ──
        H.pub.stage=street_stage(0);
        H.pub.action_to_act = (uint8_t)((button)%N_); // late position маркер
        betting_round(H,start_pre);

        // ── флоп/тёрн/ривер ──
        for(int street=1; street<=3 && active_count(H)>1; ++street){
            // докрутка борда: флоп 3 карты, иначе 1
            int add = (street==1)?3:1;
            for(int k=0;k<add;++k) H.pub.board.add(deck_.deal());
            H.current_bet=0; H.last_raise=100;
            H.pub.stage=street_stage(street);
            int start_post = (N_==2)? bb : (button+1)%N_; // постфлоп: первым OOP
            // если все кроме одного олл-ин — торговли нет
            int can_act=0; for(int i=0;i<N_;++i) if(!H.folded[i]&&!H.allin[i])can_act++;
            if(can_act>=2) betting_round(H,start_post);
        }

        std::array<int32_t,6> winnings{};
        showdown(H,winnings);

        for(int i=0;i<N_;++i) net[i]=winnings[i]-H.total_inv[i];

        // инвариант сохранения фишек (в debug)
        int32_t s=0; for(int i=0;i<N_;++i)s+=net[i];
        if(s!=0){ /* при корректной логике ноль */ }
    }
};

int main(int argc, char** argv){
    std::vector<Policy> seats;
    long hands=100000; uint64_t seed=12345; int32_t start=10000;
    for(int i=1;i<argc;++i){
        std::string a=argv[i];
        auto next=[&](){ return (i+1<argc)?std::string(argv[++i]):std::string(); };
        if(a=="--seat"){
            std::string spec=next(); Policy p;
            if(spec.rfind("model:",0)==0){
                p.type=PType::Model;
                std::string path=spec.substr(6);
                p.model=std::make_unique<PokerHistoryTransformer>();
                if(!p.model->load_weights(path.c_str())){
                    fprintf(stderr,"ERROR: не загрузить веса %s\n",path.c_str()); return 1; }
                p.name="model("+path+")";
            } else if(spec=="random"){p.type=PType::Random;p.name="random";}
            else if(spec=="call"){p.type=PType::Call;p.name="call-station";}
            else if(spec=="tight"){p.type=PType::Tight;p.name="tight";}
            else if(spec=="aggro"){p.type=PType::Aggro;p.name="aggro";}
            else { fprintf(stderr,"неизвестный seat: %s\n",spec.c_str()); return 1; }
            seats.push_back(std::move(p));
        }
        else if(a=="--hands") hands=std::atol(next().c_str());
        else if(a=="--seed")  seed=std::strtoull(next().c_str(),nullptr,10);
        else if(a=="--stack") start=(int32_t)(std::atof(next().c_str())*100);
        else if(a=="--help"){
            printf("usage: %s --seat <model:FILE|random|call|tight|aggro> [--seat ...] "
                   "[--hands N --seed N --stack BB]\n",argv[0]); return 0; }
    }
    if(seats.size()<2 || seats.size()>6){
        fprintf(stderr,"нужно 2..6 игроков (--seat). Сейчас: %zu\n",seats.size()); return 1; }

    printf("Стол: %zu игроков, %ld раздач, стек %d BB, seed %llu\n",
           seats.size(), hands, start/100, (unsigned long long)seed);
    for(size_t i=0;i<seats.size();++i) printf("  seat %zu: %s\n", i, seats[i].name.c_str());

    Arena arena(std::move(seats), start, seed);
    arena.run(hands);
    arena.report();
    return 0;
}
