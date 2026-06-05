// tools/arena_v2.cpp
// Стол с сайзингом и согласованным кодированием (EncoderV2 — тот же, что в обучении).
// Места: model2:FILE | call | random | tight | aggro | teacher
// Режим генерации данных: --gen FILE  (логирует решения teacher как обучающие примеры)
//
//   ./arena_v2 --seat model2:policy_v2.bin --seat call --seat random --seat tight --hands 100000
//   ./arena_v2 --gen teacher.txt --hands 200000           # сгенерировать датасет учителя

#include "poker_v2.hpp"
#include "pushfold_gto.hpp"
#include "../poker_core/poker_core.hpp"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <memory>
#include <algorithm>
#include <cmath>
#include <fstream>

using namespace pv2;

enum class PT { Model, ModelX, Call, Random, Tight, Aggro, Teacher };
struct Seat { PT type; std::string name; std::unique_ptr<PolicyNet> net; };

// ─── Онлайн-модель соперника (σ̂_opp на публичной инфо, §XIV статьи) ──────────
// Накапливается ЧЕРЕЗ раздачи → агент учится отклонениям пула и эксплуатирует.
struct OppStats {
    double face_n=0, fold_k=0;     // fold-to-bet
    double aggr=0, passive=0;       // агрессия = (bet+raise)/(bet+raise+call+check)
    void obs(int klass, bool facing){
        if(facing){ face_n++; if(klass==FOLD) fold_k++; }
        if(klass>=R33 && klass<=ALLIN) aggr++;
        else if(klass==CALL||klass==CHECK) passive++;
    }
    double fold_to_bet() const { return face_n>0 ? fold_k/face_n : 0.45; }
    double aggression()  const { double t=aggr+passive; return t>0?aggr/t:0.35; }
    double n() const { return aggr+passive; }
};

struct H {
    int N;
    std::array<int,6> stack{},tinv{},sinv{}; std::array<bool,6> fold{},allin{};
    std::array<poker::Card,6> a{},b{};
    poker::CardSet board{};
    int cur=0,lastr=100,street=0,button=0;
};

struct Arena {
    std::vector<Seat> seats; int N; int start; 
    poker::DeckShuffler deck; poker::Xoshiro256ss rng; poker::HandEvaluator ev; EncoderV2 enc;
    std::vector<double> sum,sq; long hands=0;
    std::ofstream* gen=nullptr;
    std::array<OppStats,6> opp_{};   // онлайн-модель соперников (персистентна через раздачи)
    double amax=0.95;                // верхняя граница α*
    pfgto::PushFold PF;              // живой GTO push/fold модуль (шаг 4b)
    int shortstack_bb=16;            // порог короткого стека для push/fold-GTO

    Arena(std::vector<Seat>&& s,int st,uint64_t seed)
        : seats(std::move(s)),N((int)seats.size()),start(st),deck(seed),rng(seed^0x99),enc(&ev){
        sum.assign(N,0);sq.assign(N,0);}

    float fr(){ return (rng.next()>>11)*(1.0f/9007199254740992.0f); }
    int pot(const H&h){int p=0;for(int i=0;i<N;i++)p+=h.tinv[i];return p;}
    int active(const H&h){int c=0;for(int i=0;i<N;i++)if(!h.fold[i])c++;return c;}
    int hero_ip(const H&h,int i){ return (i==h.button)?1:0; }

    // ── MC equity: hero против nopp случайных рук (hero должен побить ВСЕХ) ──
    float equity(poker::Card c0,poker::Card c1,const poker::CardSet&board,int samples,int nopp=1){
        if(nopp<1)nopp=1;
        poker::CardSet used; used.add(c0);used.add(c1); board.for_each([&](poker::Card c){used.add(c);});
        int nb=board.size(); int win=0; double acc=0; int tot=0;
        std::vector<poker::Card> deckv; for(int i=0;i<52;i++){poker::Card c{(uint8_t)i}; if(!used.contains(c))deckv.push_back(c);}
        int need=5-nb;
        for(int s=0;s<samples;s++){
            for(int k=(int)deckv.size()-1;k>0;k--){int j=(int)(rng.next_bounded(k+1));std::swap(deckv[k],deckv[j]);}
            int idx=0;
            std::vector<poker::Card> common; board.for_each([&](poker::Card c){common.push_back(c);});
            for(int k=0;k<need;k++) common.push_back(deckv[idx++]);
            std::vector<poker::Card> hero=common; hero.push_back(c0); hero.push_back(c1);
            int hs=ev.evaluate(hero);
            bool beat_all=true; bool tie=false;
            for(int o=0;o<nopp;o++){
                std::vector<poker::Card> opp=common; opp.push_back(deckv[idx++]); opp.push_back(deckv[idx++]);
                int os=ev.evaluate(opp);
                if(os>hs){beat_all=false;break;} if(os==hs)tie=true;
            }
            if(beat_all){ acc += tie?0.5:1.0; } tot++;
        }
        return tot? (float)(acc/tot) : 0.5f;
    }

    // ── teacher: equity + pot-odds + сайзинг ──
    int teacher_class(const H&h,int i,bool facing){
        int p=pot(h), tocall=h.cur-h.sinv[i];
        int samp = (h.board.size()>=4)?160:120;
        float eq=equity(h.a[i],h.b[i],h.board,samp);
        if(facing){
            float odds = (p+tocall>0)? (float)tocall/(p+tocall):0;
            if(eq < odds-0.02f) return FOLD;
            if(eq > 0.80f) return R100;          // сильная — рейз в банк
            if(eq > 0.66f) return (fr()<0.6f?R66:CALL);
            return CALL;                          // достаточно эквити для колла
        } else {
            if(eq>0.82f) return R100;             // вэлью
            if(eq>0.66f) return R66;
            if(eq>0.55f) return (fr()<0.5f?R33:CHECK);
            if(fr()<0.12f) return R66;            // редкий блеф
            return CHECK;
        }
    }

    int random_class(bool facing){
        if(facing){ float r=fr(); if(r<0.30f)return FOLD; if(r<0.78f)return CALL;
            return R33+(int)(fr()*3.99f); }
        float r=fr(); if(r<0.60f)return CHECK; return R33+(int)(fr()*3.99f);
    }
    int heuristic_class(const H&h,int i,bool facing,PT t){
        // грубая сила (без MC) для дешёвых ботов
        std::vector<poker::Card> cs={h.a[i],h.b[i]}; h.board.for_each([&](poker::Card c){cs.push_back(c);});
        float s; if(cs.size()>=5)s=ev.evaluate(cs)/7462.0f;
        else { int hi=std::max(h.a[i].rank(),h.b[i].rank()),lo=std::min(h.a[i].rank(),h.b[i].rank());
               s=std::min(1.0f,(hi+ (h.a[i].rank()==h.b[i].rank()?6.0f:0))/18.0f); }
        if(t==PT::Call) return facing?CALL:CHECK;
        if(t==PT::Tight){ if(facing)return s<0.55f?FOLD:CALL; return s>0.8f?R66:CHECK; }
        // Aggro
        if(fr()<0.5f||s>0.5f) return facing?R66:R66; return facing?CALL:CHECK;
    }

    int decide_class(const H&h,int i,bool facing){
        const Seat&S=seats[i];
        if(S.type==PT::ModelX) return decide_modelx(h,i,facing);
        if(S.type==PT::Model){
            int tocall=h.cur-h.sinv[i];
            auto fv=enc.encode(h.board,h.a[i],h.b[i],h.street,pot(h),tocall,hero_ip(h,i));
            float pr[NUM_ACT]; S.net->infer(fv,pr);
            if(facing)pr[CHECK]=0; else {pr[FOLD]=0;pr[CALL]=0;}
            int best=0;float bv=-1;for(int k=0;k<NUM_ACT;k++)if(pr[k]>bv){bv=pr[k];best=k;}
            return best;
        }
        if(S.type==PT::Teacher) return teacher_class(h,i,facing);
        if(S.type==PT::Random)  return random_class(facing);
        return heuristic_class(h,i,facing,S.type);
    }

    // ── Полноценный BAEM-агент: прайор Pθ + онлайн-эксплойт (Алгоритм 1) ──────
    int decide_modelx(const H&h,int i,bool facing){
        const Seat&S=seats[i];
        // ── ШАГ 4b: на коротком HU-префлопе играем ТОЧНУЮ GTO push/fold ──
        if(N==2 && h.street==0){
            int eff_bb = std::min(h.stack[0]+h.tinv[0], h.stack[1]+h.tinv[1])/100;
            if(eff_bb>=2 && eff_bb<=shortstack_bb){
                int tc=h.cur-h.sinv[i];
                if(tc>0) return PF.bb_call(h.a[i],h.b[i],eff_bb)? CALL : FOLD;   // BB vs шов
                else     return PF.sb_push(h.a[i],h.b[i],eff_bb)? ALLIN : FOLD;  // SB шов/фолд
            }
        }
        int P=pot(h), tocall=h.cur-h.sinv[i], stack=h.stack[i], sinv=h.sinv[i];

        // (1) прайор π_GTO от обученной сети Pθ (ур. 11)
        auto fv=enc.encode(h.board,h.a[i],h.b[i],h.street,P,tocall,hero_ip(h,i));
        float prior[NUM_ACT]; S.net->infer(fv,prior);

        // (2) σ̂_opp: агрегируем наблюдения только по АКТИВНЫМ в раздаче соперникам
        //     (в HU-банке — по единственному реальному оппоненту; не «фантомный пул»)
        OppStats o; for(int s=0;s<N;s++) if(s!=i && !h.fold[s]){ o.face_n+=opp_[s].face_n;o.fold_k+=opp_[s].fold_k;
            o.aggr+=opp_[s].aggr;o.passive+=opp_[s].passive; }
        double f=o.fold_to_bet(), ag=o.aggression(), nobs=o.n();

        // (3) λ̂ через отклонение от GTO (ур. 9/21). Базовые GTO-частоты ~ {f≈0.45, aggr≈0.35}
        //     3-классовое σ̂_opp = {fold-to-bet, агрессия, пассив}; D_KL к GTO, норм. на Dmax.
        double opp_pass = std::max(0.0, 1.0-f-ag);
        const double e=1e-4; auto sm=[&](double x){return (1-e)*x+e/3.0;};
        double qg[3]={0.45,0.35,0.20};                 // базовые GTO-частоты {fold,aggr,pass}
        double Dmax=0; for(int k=0;k<3;k++) Dmax=std::max(Dmax,-std::log(sm(qg[k])));  // sup KL = -ln(min q)
        auto kl3=[&](double a0,double a1,double a2,double b0,double b1,double b2){
            double p[3]={sm(a0),sm(a1),sm(a2)}, q[3]={sm(b0),sm(b1),sm(b2)};
            double s=0; for(int k=0;k<3;k++) s+=p[k]*std::log(p[k]/q[k]); return std::max(0.0,s); };
        double dkl = kl3(f,ag,opp_pass, qg[0],qg[1],qg[2]);
        double lam_hat = std::clamp(1.0 - dkl/Dmax, 0.0, 1.0);     // λ̂ ∈ [0,1]
        // эксплуатируем только ЯВНОЕ отклонение (мёртвая зона 0.30): слабые отклонения → чистый GTO
        double raw_expl = 1.0 - lam_hat;
        double exploitability = std::clamp((raw_expl - 0.30)/0.70, 0.0, 1.0);

        // (4) уверенность h(ε), ε=O(1/√t) (ур. 8/13)
        double h_eps = std::clamp(1.0 - 3.0/std::sqrt(nobs+1.0), 0.0, 1.0);
        // в мультивее чистая эксплуатация невозможна (соперники разнородны) → демпфируем
        int nactive=0; for(int s=0;s<N;s++) if(s!=i && !h.fold[s]) nactive++;
        double mw = 1.0/std::sqrt((double)std::max(1,nactive));
        double alpha = std::clamp(exploitability,0.0,1.0) * h_eps * amax * mw;   // α*(t)

        // (5) EV(a) против модели соперника → softmax(EV/τ) = π_exploit (строка 15)
        double EV[NUM_ACT]; bool legal[NUM_ACT];
        for(int k=0;k<NUM_ACT;k++){legal[k]=false;EV[k]=-1e9;}
        int nopp = std::max(1, nactive);
        double eqv = equity(h.a[i],h.b[i],h.board, h.board.size()>=4?160:130, nopp);
        auto raise_add=[&](int k){ float fr=RAISE_FRAC[k]; int add=(int)((P+tocall)*fr);
            int rt=h.cur+std::max(add,100); rt=std::min(rt,sinv+stack); return rt-sinv; };
        // P(все активные спасуют) растёт с размером, но степень nopp (каждый должен сбросить)
        auto pfold=[&](int dadd){ double sizefrac=(P>0)?(double)dadd/P:1.0;
            double pf1=f*std::clamp(0.6+0.5*sizefrac,0.4,1.6); pf1=std::clamp(pf1,0.0,0.97);
            return std::pow(pf1, (double)nopp); };
        if(facing){
            legal[FOLD]=true; EV[FOLD]=0.0;
            legal[CALL]=true; EV[CALL]=eqv*(P+tocall)-tocall;
            for(int k=R33;k<=ALLIN;k++){ int d=(k==ALLIN)?(sinv+stack-sinv):raise_add(k);
                if(d<=tocall) continue;
                if(k==ALLIN && eqv<0.80) continue;      // дисперсия: олл-ин только с сильной рукой
                if(k==R150  && eqv<0.68) continue;
                legal[k]=true; double pf=pfold(d);
                EV[k]=pf*P + (1-pf)*(eqv*(P+2*d)-d); }
        } else {
            legal[CHECK]=true; EV[CHECK]=eqv*P;
            for(int k=R33;k<=ALLIN;k++){ int d=(k==ALLIN)?stack:raise_add(k);
                if(d<=0) continue;
                if(k==ALLIN && eqv<0.80) continue;       // ур. (6): σ²/E[X]² — не джемим тонко
                if(k==R150  && eqv<0.68) continue;
                legal[k]=true; double pf=pfold(d);
                EV[k]=pf*P + (1-pf)*(eqv*(P+2*d)-d); }
        }
        // τ(t): убывает с накоплением данных (ур. 16, упрощённо) → резче к argmax EV
        double tau = (50.0 + 250.0*std::exp(-nobs/400.0));   // в единицах BB×100
        double ex[NUM_ACT]={0}; double mx=-1e18;
        for(int k=0;k<NUM_ACT;k++) if(legal[k]) mx=std::max(mx,EV[k]/tau);
        double Z=0; for(int k=0;k<NUM_ACT;k++) if(legal[k]){ ex[k]=std::exp(EV[k]/tau-mx); Z+=ex[k]; }
        for(int k=0;k<NUM_ACT;k++) ex[k]= (legal[k]&&Z>0)? ex[k]/Z : 0.0;

        // (6) маскируем прайор по легальности и нормируем
        if(facing) prior[CHECK]=0; else {prior[FOLD]=0;prior[CALL]=0;}
        double ps=0; for(int k=0;k<NUM_ACT;k++){ if(!legal[k])prior[k]=0; ps+=prior[k]; }
        if(ps>0) for(int k=0;k<NUM_ACT;k++) prior[k]/=ps;

        // (7) π = (1−α*)·π_GTO + α*·π_exploit  → argmax
        int best=0; double bv=-1;
        for(int k=0;k<NUM_ACT;k++){ if(!legal[k])continue;
            double pmix=(1-alpha)*prior[k] + alpha*ex[k];
            if(pmix>bv){bv=pmix;best=k;} }
        return best;
    }

    void apply(H&h,int i,int klass){
        int tocall=h.cur-h.sinv[i];
        auto [t,rt]=act_from_class(klass,pot(h),tocall,h.sinv[i],h.stack[i],h.cur);
        if(t==poker::ActionType::Fold){h.fold[i]=true;return;}
        if(t==poker::ActionType::Check)return;
        if(t==poker::ActionType::Call){int pay=std::min(tocall,h.stack[i]);h.stack[i]-=pay;h.sinv[i]+=pay;h.tinv[i]+=pay;if(h.stack[i]==0)h.allin[i]=true;return;}
        int target=std::min(rt,h.sinv[i]+h.stack[i]); int add=target-h.sinv[i];
        h.stack[i]-=add;h.sinv[i]+=add;h.tinv[i]+=add;
        int inc=h.sinv[i]-h.cur; if(inc>0){h.lastr=inc;h.cur=h.sinv[i];}
        if(h.stack[i]==0)h.allin[i]=true;
    }

    // лог v2-примера решения teacher для места i (текущее состояние)
    void log_example(const H&h,int i){
        if(!gen)return;
        int tocall=h.cur-h.sinv[i];
        int klass=teacher_class(h,i,tocall>0);
        std::vector<int> bs; h.board.for_each([&](poker::Card c){bs.push_back(c.idx);});
        (*gen)<<klass<<" "<<h.street<<" "<<pot(h)<<" "<<tocall<<" "<<hero_ip(h,i)
              <<" "<<(int)h.a[i].idx<<" "<<(int)h.b[i].idx<<" "<<bs.size();
        for(int c:bs)(*gen)<<" "<<c;
        (*gen)<<" 0\n";  // историю опускаем (agg=0) — для простоты и стабильности
    }

    void betting(H&h,int start){
        for(int i=0;i<N;i++)h.sinv[i]=0;
        std::array<bool,6> acted{}; int to=start,guard=0;
        while(guard++<300){
            bool need=false; for(int i=0;i<N;i++){if(h.fold[i]||h.allin[i])continue; if(!acted[i]||h.sinv[i]<h.cur){need=true;break;}}
            if(!need||active(h)<=1)break;
            int i=to%N; if(h.fold[i]||h.allin[i]){to++;continue;}
            bool facing=h.cur>h.sinv[i];
            if(gen)log_example(h,i);              // лог ДО применения
            int kl=decide_class(h,i,facing);
            opp_[i].obs(kl,facing);               // онлайн-обновление модели соперника
            int before=h.cur; apply(h,i,kl); acted[i]=true;
            if(h.cur>before)for(int k=0;k<N;k++)if(k!=i&&!h.fold[k]&&!h.allin[k])acted[k]=false;
            to++;
        }
    }

    void showdown(H&h,std::array<int,6>&win){
        std::vector<int> lv; for(int i=0;i<N;i++)if(h.tinv[i]>0)lv.push_back(h.tinv[i]);
        std::sort(lv.begin(),lv.end()); lv.erase(std::unique(lv.begin(),lv.end()),lv.end());
        int prev=0; std::vector<poker::Card> cs;
        for(int L:lv){ int contrib=0; for(int i=0;i<N;i++)if(h.tinv[i]>=L)contrib++;
            int layer=(L-prev)*contrib; int bs=-1; std::vector<int> w;
            for(int i=0;i<N;i++){ if(h.fold[i]||h.tinv[i]<L)continue;
                cs.clear();cs.push_back(h.a[i]);cs.push_back(h.b[i]);h.board.for_each([&](poker::Card c){cs.push_back(c);});
                int st=(cs.size()>=5)?(int)ev.evaluate(cs):h.a[i].rank()*13+h.b[i].rank();
                if(st>bs){bs=st;w.clear();w.push_back(i);}else if(st==bs)w.push_back(i);}
            if(!w.empty()){int sh=layer/(int)w.size(),rem=layer-sh*(int)w.size();
                for(size_t k=0;k<w.size();k++)win[w[k]]+=sh+(k==0?rem:0);}
            prev=L; }
    }

    void play(int button,std::array<int,6>&net){
        H h; h.N=N; h.button=button;
        for(int i=0;i<N;i++){h.stack[i]=start;h.fold[i]=h.allin[i]=false;h.tinv[i]=h.sinv[i]=0;}
        deck.shuffle(); for(int i=0;i<N;i++){h.a[i]=deck.deal();h.b[i]=deck.deal();}
        int sb=(N==2)?button:(button+1)%N, bb=(N==2)?(button+1)%N:(button+2)%N;
        auto post=[&](int i,int amt){int p=std::min(amt,h.stack[i]);h.stack[i]-=p;h.sinv[i]+=p;h.tinv[i]+=p;};
        post(sb,50);post(bb,100); h.cur=100;h.lastr=100;
        int start_pre=(N==2)?sb:(bb+1)%N;
        h.street=0; betting(h,start_pre);
        for(int st=1;st<=3&&active(h)>1;st++){ int add=(st==1)?3:1;
            for(int k=0;k<add;k++)h.board.add(deck.deal());
            h.cur=0;h.lastr=100;h.street=st;
            int ca=0;for(int i=0;i<N;i++)if(!h.fold[i]&&!h.allin[i])ca++;
            int sp=(N==2)?bb:(button+1)%N; if(ca>=2)betting(h,sp); }
        std::array<int,6> win{}; showdown(h,win);
        for(int i=0;i<N;i++)net[i]=win[i]-h.tinv[i];
    }

    void run(long n){ for(long t=0;t<n;t++){ std::array<int,6> net{}; play((int)(t%N),net);
        for(int i=0;i<N;i++){double bb=net[i]/100.0;sum[i]+=bb;sq[i]+=bb*bb;} hands++; } }

    void report(){
        printf("\n=== %ld раздач, стол на %d ===\n%-26s %10s %10s\n",hands,N,"agent","BB/100","95%CI±");
        std::vector<int> id(N);for(int i=0;i<N;i++)id[i]=i;
        std::sort(id.begin(),id.end(),[&](int a,int b){return sum[a]/hands>sum[b]/hands;});
        for(int k=0;k<N;k++){int i=id[k];double m=sum[i]/hands,var=sq[i]/hands-m*m,se=std::sqrt(std::max(0.0,var)/hands);
            printf("%-26s %10.2f %10.2f\n",seats[i].name.c_str(),m*100,1.96*se*100);}
    }
};

int main(int argc,char**argv){
    std::vector<Seat> seats; long hands=100000; uint64_t seed=7; int start=10000;
    std::string genfile;
    for(int i=1;i<argc;i++){std::string a=argv[i];auto nx=[&]{return (i+1<argc)?std::string(argv[++i]):std::string();};
        if(a=="--seat"){std::string sp=nx();Seat s;
            if(sp.rfind("model2x:",0)==0){s.type=PT::ModelX;s.net=std::make_unique<PolicyNet>();
                if(!s.net->load(sp.substr(8).c_str())){fprintf(stderr,"no weights %s\n",sp.substr(8).c_str());return 1;}
                s.name="BAEM("+sp.substr(8)+")";}
            else if(sp.rfind("model2:",0)==0){s.type=PT::Model;s.net=std::make_unique<PolicyNet>();
                if(!s.net->load(sp.substr(7).c_str())){fprintf(stderr,"no weights %s\n",sp.substr(7).c_str());return 1;}
                s.name="model2("+sp.substr(7)+")";}
            else if(sp=="call"){s.type=PT::Call;s.name="call-station";}
            else if(sp=="random"){s.type=PT::Random;s.name="random";}
            else if(sp=="tight"){s.type=PT::Tight;s.name="tight";}
            else if(sp=="aggro"){s.type=PT::Aggro;s.name="aggro";}
            else if(sp=="teacher"){s.type=PT::Teacher;s.name="teacher";}
            else{fprintf(stderr,"bad seat %s\n",sp.c_str());return 1;}
            seats.push_back(std::move(s));}
        else if(a=="--hands")hands=atol(nx().c_str());
        else if(a=="--seed")seed=strtoull(nx().c_str(),0,10);
        else if(a=="--gen")genfile=nx();
        else if(a=="--stackbb")start=atoi(nx().c_str())*100;
    }

    if(!genfile.empty()){
        // генерация: смешанный стол из «шумных» ботов; логируем решения teacher
        if(seats.empty()){ for(int i=0;i<4;i++){Seat s; s.type=(i%2)?PT::Aggro:PT::Random; s.name="gen";seats.push_back(std::move(s));} }
        Arena ar(std::move(seats),start,seed);
        std::ofstream out(genfile); ar.gen=&out;
        printf("[gen] %ld раздач → %s\n",hands,genfile.c_str());
        ar.run(hands);
        printf("[gen] done\n"); return 0;
    }

    if(seats.size()<2||seats.size()>6){fprintf(stderr,"нужно 2..6 мест\n");return 1;}
    printf("Стол: %zu, %ld раздач, стек %d BB, seed %llu\n",seats.size(),hands,start/100,(unsigned long long)seed);
    for(size_t i=0;i<seats.size();i++)printf("  seat %zu: %s\n",i,seats[i].name.c_str());
    Arena ar(std::move(seats),start,seed); ar.run(hands); ar.report();
    return 0;
}
