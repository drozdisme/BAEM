// tools/probe_v2.cpp — диагностика «понимает ли агент покер и деньги».
// Печатает политику обученного прайора в канонических спотах (где верный ход очевиден),
// тест дисциплины по пот-оддсам и корреляцию агрессии с реальным эквити.
//
//   ./probe_v2 --weights policy_v2.bin
//
// Это проверка ПРАЙОРА (выученной сети) — именно он показывает, что модель усвоила.

#include "poker_v2.hpp"
#include "../poker_core/poker_core.hpp"
#include <cstdio>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>

using namespace pv2;
static poker::HandEvaluator EV;
static poker::Xoshiro256ss RNG(20260605);

static poker::Card C(int rank,int suit){ return poker::Card{(uint8_t)(suit*13+rank)}; } // rank0=2..12=A; suit0=c1=d2=h3=s
// MC-эквити hero против 1 случайной руки на докрученном борде
static float equity(poker::Card a,poker::Card b,const poker::CardSet&board,int N=4000){
    poker::CardSet used; used.add(a);used.add(b); board.for_each([&](poker::Card c){used.add(c);});
    std::vector<poker::Card> d; for(int i=0;i<52;i++){poker::Card c{(uint8_t)i}; if(!used.contains(c))d.push_back(c);}
    int nb=board.size(), need=5-nb; double acc=0;
    std::vector<poker::Card> common,hero,opp;
    for(int s=0;s<N;s++){
        for(int k=(int)d.size()-1;k>0;k--){int j=(int)RNG.next_bounded(k+1);std::swap(d[k],d[j]);}
        common.clear(); board.for_each([&](poker::Card c){common.push_back(c);});
        for(int k=0;k<need;k++)common.push_back(d[k]);
        hero=common;hero.push_back(a);hero.push_back(b);
        opp=common;opp.push_back(d[need]);opp.push_back(d[need+1]);
        int hs=EV.evaluate(hero),os=EV.evaluate(opp);
        acc += (hs>os)?1.0:(hs==os?0.5:0.0);
    }
    return (float)(acc/N);
}

static EncoderV2 ENC(&EV);
static PolicyNet NET;

static void show(const char* label, const poker::CardSet&board, poker::Card a,poker::Card b,
                 int street,int pot,int tocall,int ip, const char* expect){
    auto fv=ENC.encode(board,a,b,street,pot,tocall,ip);
    float p[NUM_ACT]; NET.infer(fv,p);
    if(tocall>0)p[CHECK]=0; else {p[FOLD]=0;p[CALL]=0;}
    float s=0; for(int k=0;k<NUM_ACT;k++)s+=p[k]; if(s>0)for(int k=0;k<NUM_ACT;k++)p[k]/=s;
    int best=0;float bv=-1;for(int k=0;k<NUM_ACT;k++)if(p[k]>bv){bv=p[k];best=k;}
    float eq=equity(a,b,board);
    printf("%-34s eq=%.0f%% | top=%-5s | ", label, eq*100, ANAME[best]);
    for(int k=0;k<NUM_ACT;k++) if(p[k]>0.04f) printf("%s %.2f  ",ANAME[k],p[k]);
    printf("\n   ожидание: %s\n", expect);
}

int main(int argc,char**argv){
    std::string w="policy_v2.bin";
    for(int i=1;i<argc;i++){std::string a=argv[i]; if(a=="--weights"&&i+1<argc)w=argv[++i];}
    if(!NET.load(w.c_str())){fprintf(stderr,"не загрузить веса %s\n",w.c_str());return 1;}
    printf("=== ПОЛИТИКА В КАНОНИЧЕСКИХ СПОТАХ (прайор) ===\n");
    enum{c=0,d=1,h=2,s=3}; enum{_2=0,_3,_4,_5,_6,_7,_8,_9,_T,_J,_Q,_K,_A};
    poker::CardSet none{};

    // префлоп
    show("AA, префлоп, открытие", none, C(_A,s),C(_A,h), 0, 150, 0, 1,
         "сильный рейз");
    show("72o, префлоп vs рейз 3bb", none, C(_7,c),C(_2,d), 0, 450, 300, 0,
         "фолд");
    show("KK, префлоп vs рейз", none, C(_K,c),C(_K,d), 0, 450, 300, 1,
         "колл/рейз, не фолд");

    // постфлоп: натсы и мусор
    poker::CardSet b1{}; b1.add(C(_A,h));b1.add(C(_K,h));b1.add(C(_5,h)); // монотонный борд
    show("натс-флеш (Qh Jh) на флопе, vs ставка", b1, C(_Q,h),C(_J,h), 1, 600, 300, 1,
         "рейз/колл (никогда фолд)");
    show("воздух (7c 2d) на A-high флопе, vs ставка", b1, C(_7,c),C(_2,d), 1, 600, 400, 0,
         "фолд");

    poker::CardSet b2{}; b2.add(C(_A,s));b2.add(C(_7,d));b2.add(C(_2,c)); // сухой борд
    show("сет (77) на A72 сухом, без ставки", b2, C(_7,h),C(_7,s), 1, 200, 0, 1,
         "ставка (вэлью)");
    show("топ-пара (AK) на A72, без ставки", b2, C(_A,c),C(_K,d), 1, 200, 0, 1,
         "ставка (вэлью)");
    show("K-high (KQ) на A72 vs крупная ставка", b2, C(_K,c),C(_Q,d), 1, 300, 250, 0,
         "фолд (нет пары, дорого)");

    // ── Дисциплина по пот-оддсам ──
    printf("\n=== ДИСЦИПЛИНА ПО ПОТ-ОДДСАМ ===\n");
    printf("Рука с фиксированным эквити, растёт цена колла — P(fold) должна расти:\n");
    poker::CardSet bd{}; bd.add(C(_9,h));bd.add(C(_8,h));bd.add(C(_2,c)); // флоп под дро
    poker::Card dh1=C(_J,h), dh2=C(_T,h); // флеш+стрит дро
    float eqd=equity(dh1,dh2,bd);
    printf("дро JhTh на 9h8h2c, эквити≈%.0f%%:\n", eqd*100);
    int pot=600;
    for(int tc : {100,300,600,1200,2400}){
        auto fv=ENC.encode(bd,dh1,dh2,1,pot,tc,0); float p[NUM_ACT]; NET.infer(fv,p);
        p[CHECK]=0; float ss=0;for(int k=0;k<NUM_ACT;k++)ss+=p[k]; if(ss>0)for(int k=0;k<NUM_ACT;k++)p[k]/=ss;
        float potodds=(float)tc/(pot+tc);
        printf("  колл %4d в банк %d (нужно %.0f%% эквити): P(fold)=%.2f P(call)=%.2f P(raise)=%.2f\n",
               tc,pot,potodds*100,p[FOLD],p[CALL],p[R33]+p[R66]+p[R100]+p[R150]+p[ALLIN]);
    }

    // ── Корреляция агрессии с эквити (вэлью-понимание) ──
    printf("\n=== АГРЕССИЯ vs ЭКВИТИ (понимание вэлью) ===\n");
    printf("1000 случайных рук на случайном флопе, без ставки. Бакеты по эквити:\n");
    int nb_buckets=5; double aggr_sum[5]={0}; long aggr_n[5]={0};
    for(int t=0;t<1000;t++){
        // случайный флоп + рука
        int idx[52]; for(int i=0;i<52;i++)idx[i]=i;
        for(int k=51;k>0;k--){int j=(int)RNG.next_bounded(k+1);std::swap(idx[k],idx[j]);}
        poker::CardSet bd2{}; bd2.add(poker::Card{(uint8_t)idx[0]});bd2.add(poker::Card{(uint8_t)idx[1]});bd2.add(poker::Card{(uint8_t)idx[2]});
        poker::Card a=poker::Card{(uint8_t)idx[3]}, b=poker::Card{(uint8_t)idx[4]};
        float eq=equity(a,b,bd2,500);
        auto fv=ENC.encode(bd2,a,b,1,300,0,1); float p[NUM_ACT]; NET.infer(fv,p);
        float aggr=p[R33]+p[R66]+p[R100]+p[R150]+p[ALLIN];
        int bk=std::min(4,(int)(eq*5)); aggr_sum[bk]+=aggr; aggr_n[bk]++;
    }
    for(int k=0;k<5;k++) printf("  эквити %2d-%2d%%: P(ставка)=%.2f (n=%ld)\n",
        k*20,(k+1)*20, aggr_n[k]?aggr_sum[k]/aggr_n[k]:0.0, aggr_n[k]);
    printf("\n(хороший агент: P(fold) растёт с ценой; P(ставка) растёт с эквити)\n");
    return 0;
}