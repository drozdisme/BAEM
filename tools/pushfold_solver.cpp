// tools/pushfold_solver.cpp  — ШАГ 1 плана «сильнейший».
// Реальный GTO-солвер NLHE для HU push/fold (короткий стек): 169 классов рук,
// настоящие эквити через HandEvaluator. CFR сходится к равновесию; exploitability→0.
// Результат — используемые в покере push/fold диапазоны по глубине стека.
//
//   g++ -std=c++20 -O3 -march=native -I.. pushfold_solver.cpp -o pushfold_solver
//   ./pushfold_solver --stack 10 --iters 4000
//
// Матрица эквити кэшируется в data/eq169.bin (считается один раз ~минуту).

#include "../poker_core/cards.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include "../poker_core/deck_rng.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <array>
#include <string>
#include <cmath>
#include <algorithm>

using poker::Card; using poker::HandEvaluator;

static const char* RC="23456789TJQKA";
struct Cls{ int hi,lo; bool suited,pair; int combos; std::string name; };

// построить 169 классов
static std::vector<Cls> build_classes(){
    std::vector<Cls> v;
    for(int hi=12;hi>=0;hi--){
        // пары
        Cls p; p.hi=hi;p.lo=hi;p.pair=true;p.suited=false;p.combos=6;
        p.name=std::string(1,RC[hi])+RC[hi]; v.push_back(p);
        // suited + offsuit (hi>lo)
        for(int lo=hi-1;lo>=0;lo--){
            Cls s; s.hi=hi;s.lo=lo;s.pair=false;s.suited=true;s.combos=4;
            s.name=std::string(1,RC[hi])+RC[lo]+"s"; v.push_back(s);
        }
        for(int lo=hi-1;lo>=0;lo--){
            Cls o; o.hi=hi;o.lo=lo;o.pair=false;o.suited=false;o.combos=12;
            o.name=std::string(1,RC[hi])+RC[lo]+"o"; v.push_back(o);
        }
    }
    return v; // 13 + 78 + 78 = 169
}
// конкретные две карты для класса (с выбором мастей)
static void cards_for(const Cls& c, int variant, Card& a, Card& b){
    // variant подбирает разные масти, чтобы избегать коллизий
    int s1=variant%4, s2=(variant/4)%4;
    if(c.pair){ if(s2==s1)s2=(s1+1)%4; a=Card{(uint8_t)(s1*13+c.hi)}; b=Card{(uint8_t)(s2*13+c.lo)}; }
    else if(c.suited){ a=Card{(uint8_t)(s1*13+c.hi)}; b=Card{(uint8_t)(s1*13+c.lo)}; }
    else { if(s2==s1)s2=(s1+1)%4; a=Card{(uint8_t)(s1*13+c.hi)}; b=Card{(uint8_t)(s2*13+c.lo)}; }
}

static HandEvaluator EV;
static poker::Xoshiro256ss RNG(12345);

// эквити класса i против класса j (MC по бордам и мастям)
static float equity_ij(const Cls& ci,const Cls& cj,int samples){
    int win=0; double acc=0; int tot=0;
    for(int s=0;s<samples;s++){
        Card a,b,c,d; cards_for(ci,s,a,b); cards_for(cj,s/4+1,c,d);
        // отбросить коллизии карт
        std::array<int,4> ids={a.idx,b.idx,c.idx,d.idx};
        bool clash=false; for(int x=0;x<4&&!clash;x++)for(int y=x+1;y<4;y++)if(ids[x]==ids[y])clash=true;
        if(clash) continue;
        // случайный борд из оставшихся
        bool used[52]={false}; for(int id:ids)used[id]=true;
        int board[5]; int got=0;
        while(got<5){ int r=(int)RNG.next_bounded(52); if(!used[r]){used[r]=true;board[got++]=r;} }
        std::vector<Card> h1={a,b},h2={c,d};
        for(int k=0;k<5;k++){h1.push_back(Card{(uint8_t)board[k]});h2.push_back(Card{(uint8_t)board[k]});}
        int v1=EV.evaluate(h1),v2=EV.evaluate(h2);
        acc += (v1>v2)?1.0:(v1==v2?0.5:0.0); tot++;
    }
    return tot? (float)(acc/tot):0.5f;
}

int main(int argc,char**argv){
    double S=10.0; long iters=4000; int eqsamples=400; std::string cache="data/eq169.bin";
    for(int i=1;i<argc;i++){ std::string a=argv[i];
        if(a=="--stack"&&i+1<argc)S=atof(argv[++i]);
        else if(a=="--iters"&&i+1<argc)iters=atol(argv[++i]);
        else if(a=="--eqsamples"&&i+1<argc)eqsamples=atoi(argv[++i]);
        else if(a=="--cache"&&i+1<argc)cache=argv[++i]; }

    auto C=build_classes(); int n=(int)C.size();
    printf("Классов: %d. Эффективный стек S=%.1f BB.\n",n,S);

    // ── матрица эквити (кэш) ──
    std::vector<float> e((size_t)n*n,0.5f);
    FILE* cf=std::fopen(cache.c_str(),"rb");
    if(cf){ size_t rd=std::fread(e.data(),sizeof(float),(size_t)n*n,cf); std::fclose(cf);
        if(rd==(size_t)n*n) printf("[eq] загружена из %s\n",cache.c_str());
        else { cf=nullptr; } }
    if(!cf){
        printf("[eq] считаю матрицу %dx%d (один раз)...\n",n,n);
        for(int i=0;i<n;i++){
            for(int j=i;j<n;j++){
                float eij=equity_ij(C[i],C[j],eqsamples);
                e[(size_t)i*n+j]=eij; e[(size_t)j*n+i]=1.0f-eij;
            }
            if(i%30==0)printf("  строка %d/%d\n",i,n);
        }
        std::system("mkdir -p data");
        FILE* wf=std::fopen(cache.c_str(),"wb");
        if(wf){ std::fwrite(e.data(),sizeof(float),(size_t)n*n,wf); std::fclose(wf); printf("[eq] сохранена в %s\n",cache.c_str()); }
    }

    // веса классов (combo counts), нормированные
    std::vector<double> w(n); double wsum=0; for(int i=0;i<n;i++){w[i]=C[i].combos; wsum+=w[i];}
    for(int i=0;i<n;i++) w[i]/=wsum;

    // ── CFR: SB push/fold, BB call/fold ──
    // регреты: SB по классам (push vs fold), BB по классам (call vs fold)
    std::vector<std::array<double,2>> rSB(n,{0,0}), sSB(n,{0,0}), rBB(n,{0,0}), sBB(n,{0,0});
    auto strat=[&](std::array<double,2>& r){ double a=std::max(0.0,r[0]),b=std::max(0.0,r[1]),s=a+b;
        return std::array<double,2>{ s>0?a/s:0.5, s>0?b/s:0.5 }; };

    for(long t=1;t<=iters;t++){
        // текущие стратегии
        std::vector<std::array<double,2>> stSB(n),stBB(n);
        double pushfreq=0; // P(SB пушит) общий, для BB
        std::vector<double> bbCall(n);
        for(int i=0;i<n;i++){ stSB[i]=strat(rSB[i]); stBB[i]=strat(rBB[i]); }
        // BB EV: для класса j, EV(call) vs EV(fold=-1) при условии что столкнулись с пушем.
        // P(SB пушит с i) = stSB[i][0]. Веса соперника ~ w[i].
        for(int j=0;j<n;j++){
            double num=0,den=0;
            for(int i=0;i<n;i++){ double pp=w[i]*stSB[i][0]; if(pp<=0)continue;
                double eq=e[(size_t)j*n+i]; // эквити BB(j) против SB(i)
                num += pp*((2*eq-1)*S); den += pp; }
            double evCall = den>0? num/den : 0.0;       // EV колла (на 1 пуш)
            double evFold = -1.0;                         // BB теряет свой блайнд 1.0
            // regret matching update для BB (взвешено вероятностью что столкнётся с пушем = den)
            double cur0=stBB[j][0],cur1=stBB[j][1]; double node=cur0*evCall+cur1*evFold;
            rBB[j][0]+= den*(evCall-node); rBB[j][1]+= den*(evFold-node);
            sBB[j][0]+=cur0; sBB[j][1]+=cur1;
        }
        // SB EV: для класса i, EV(push) vs EV(fold=-0.5).
        for(int i=0;i<n;i++){
            double evPush=0;
            for(int j=0;j<n;j++){ double pcall=stBB[j][0];
                double eq=e[(size_t)i*n+j];
                double showdown=(2*eq-1)*S;
                evPush += w[j]*( pcall*showdown + (1-pcall)*(1.0) ); // BB фолд → SB +1.0
            }
            double evFold=-0.5;
            double cur0=stSB[i][0],cur1=stSB[i][1]; double node=cur0*evPush+cur1*evFold;
            rSB[i][0]+= (evPush-node); rSB[i][1]+= (evFold-node);
            sSB[i][0]+=cur0; sSB[i][1]+=cur1;
        }
    }

    // средние стратегии
    auto avg=[&](std::array<double,2>& s){ double t=s[0]+s[1]; return t>0? s[0]/t : 0.0; };
    printf("\n=== GTO push/fold, стек %.1f BB ===\n",S);
    // SB push range
    double sbPushCombos=0,totCombos=0;
    printf("SB пушит (частота по классам, показаны >0.5):\n  ");
    int shown=0;
    for(int i=0;i<n;i++){ double p=avg(sSB[i]); totCombos+=C[i].combos; if(p>0.5){sbPushCombos+=C[i].combos*p;
        if(shown++<40)printf("%s ",C[i].name.c_str()); } else sbPushCombos+=C[i].combos*p; }
    printf("\n  доля диапазона SB пуш = %.1f%%\n", 100.0*sbPushCombos/totCombos);
    double bbCallCombos=0;
    printf("BB коллит пуш (показаны >0.5):\n  "); shown=0;
    for(int j=0;j<n;j++){ double p=avg(sBB[j]); bbCallCombos+=C[j].combos*p; if(p>0.5&&shown++<40)printf("%s ",C[j].name.c_str()); }
    printf("\n  доля диапазона BB колл = %.1f%%\n", 100.0*bbCallCombos/totCombos);

    // ── exploitability: best-response обеих сторон против средних стратегий ──
    // BB BR: для каждого j выбрать max(EV_call, EV_fold) против средней SB.
    std::vector<std::array<double,2>> avgSB(n),avgBB(n);
    for(int i=0;i<n;i++){ double a=avg(sSB[i]); avgSB[i]={a,1-a}; double b=avg(sBB[i]); avgBB[i]={b,1-b}; }
    // value SB under avg profile
    auto sbEV=[&](int i,bool push)->double{ if(!push)return -0.5; double ev=0;
        for(int j=0;j<n;j++){double pc=avgBB[j][0]; double eq=e[(size_t)i*n+j]; ev+=w[j]*(pc*((2*eq-1)*S)+(1-pc)*1.0);} return ev; };
    auto bbEV=[&](int j,bool call)->double{ double num=0,den=0;
        for(int i=0;i<n;i++){double pp=w[i]*avgSB[i][0]; if(pp<=0)continue; double eq=e[(size_t)j*n+i]; num+=pp*(call?((2*eq-1)*S):(-1.0)); den+=pp;} return den>0?num/den:(call?0.0:0.0); };
    double exploit=0;
    // SB BR gain
    double valSB=0,brSB=0;
    for(int i=0;i<n;i++){ double cur=avgSB[i][0]*sbEV(i,true)+avgSB[i][1]*sbEV(i,false);
        double best=std::max(sbEV(i,true),sbEV(i,false)); valSB+=w[i]*cur; brSB+=w[i]*best; }
    // BB BR gain (на условие пуша)
    double valBB=0,brBB=0,pushmass=0;
    for(int j=0;j<n;j++){ double den=0; for(int i=0;i<n;i++)den+=w[i]*avgSB[i][0];
        double cur=avgBB[j][0]*bbEV(j,true)+avgBB[j][1]*bbEV(j,false);
        double best=std::max(bbEV(j,true),bbEV(j,false)); valBB+=w[j]*cur; brBB+=w[j]*best; }
    exploit=(brSB-valSB)+(brBB-valBB);
    printf("\nexploitability (NashConv, BB/раздача) = %.5f  (→0 = точный GTO push/fold)\n",exploit);
    printf("эталон: на 10bb SB пушит широко (~55-70%%), BB коллит уже (~35-45%%).\n");
    return 0;
}
