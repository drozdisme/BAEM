// tools/lbr_eval.cpp — ШАГ 2: эксплуатируемость агента через Local Best Response.
// LBR (в позиции) отслеживает диапазон агента как K частиц (сэмпл комбо), по его
// политике, и выбирает максимально-EV действие {чек,бет-пот,олл-ин} с роллаутом;
// агент отвечает по политике; исход — по реальным картам. Выигрыш LBR (bb/100) =
// НИЖНЯЯ ГРАНИЦА эксплуатируемости (без блефов из руки агента и ре-рейзов).
//   ./lbr_eval --weights policy_v2.bin --hands 4000 [--particles 64]
#include "poker_v2.hpp"
#include "../poker_core/poker_core.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>
using namespace pv2;
static poker::HandEvaluator EV;
static poker::Xoshiro256ss RNG(99999);
static EncoderV2 ENC(&EV);
static PolicyNet NET;
struct P{ poker::Card a,b; double w; };
static double frand(){ return (RNG.next()>>11)*(1.0/9007199254740992.0); }
static void pol(const poker::CardSet& bd,poker::Card a,poker::Card b,int st,int pot,int tc,int ip,float* p){
    auto fv=ENC.encode(bd,a,b,st,pot,tc,ip); NET.infer(fv,p);
    if(tc>0)p[CHECK]=0; else {p[FOLD]=0;p[CALL]=0;}
    float s=0; for(int k=0;k<NUM_ACT;k++)s+=p[k]; if(s>0)for(int k=0;k<NUM_ACT;k++)p[k]/=s;
}
// эквити LBR против частиц (1 докрут борда на частицу)
static double eq_vs(poker::Card l1,poker::Card l2,const poker::CardSet& bd,std::vector<P>& R){
    int nb=bd.size(); poker::CardSet base; base.add(l1);base.add(l2); bd.for_each([&](poker::Card c){base.add(c);});
    double acc=0,ws=0;
    for(auto&pr:R){ if(pr.w<=0)continue; if(base.contains(pr.a)||base.contains(pr.b))continue;
        poker::CardSet u=base; u.add(pr.a);u.add(pr.b); int need=5-nb,bb[5],g=0;
        while(g<need){int x=(int)RNG.next_bounded(52);poker::Card cx{(uint8_t)x};if(!u.contains(cx)){u.add(cx);bb[g++]=x;}}
        std::vector<poker::Card> hl={l1,l2},ho={pr.a,pr.b}; bd.for_each([&](poker::Card c){hl.push_back(c);ho.push_back(c);});
        for(int k=0;k<need;k++){hl.push_back(poker::Card{(uint8_t)bb[k]});ho.push_back(poker::Card{(uint8_t)bb[k]});}
        int v1=EV.evaluate(hl),v2=EV.evaluate(ho); acc+=pr.w*((v1>v2)?1.0:(v1==v2?0.5:0.0)); ws+=pr.w; }
    return ws>0?acc/ws:0.5;
}
// для бета b: P(fold) и эквити LBR против КОЛЛ-диапазона (а не всего)
static void eval_bet(poker::Card l1,poker::Card l2,const poker::CardSet& bd,std::vector<P>& R,
                     int st,int pot,int b,double& pf,double& eq_call){
    int nb=bd.size(); poker::CardSet base; base.add(l1);base.add(l2); bd.for_each([&](poker::Card c){base.add(c);});
    double fmass=0,ws=0,acc=0,cw=0;
    for(auto&pr:R){ if(pr.w<=0)continue; if(bd.contains(pr.a)||bd.contains(pr.b))continue;
        if(base.contains(pr.a)||base.contains(pr.b)){ ws+=pr.w; continue; }
        float p[NUM_ACT]; pol(bd,pr.a,pr.b,st,pot+b,b,0,p);
        fmass+=pr.w*p[FOLD]; ws+=pr.w;
        double callw=pr.w*(1.0-p[FOLD]); if(callw<=0)continue;
        poker::CardSet u=base; u.add(pr.a);u.add(pr.b); int need=5-nb,bb[5],g=0;
        while(g<need){int x=(int)RNG.next_bounded(52);poker::Card cx{(uint8_t)x};if(!u.contains(cx)){u.add(cx);bb[g++]=x;}}
        std::vector<poker::Card> hl={l1,l2},ho={pr.a,pr.b}; bd.for_each([&](poker::Card c){hl.push_back(c);ho.push_back(c);});
        for(int k=0;k<need;k++){hl.push_back(poker::Card{(uint8_t)bb[k]});ho.push_back(poker::Card{(uint8_t)bb[k]});}
        int v1=EV.evaluate(hl),v2=EV.evaluate(ho); acc+=callw*((v1>v2)?1.0:(v1==v2?0.5:0.0)); cw+=callw;
    }
    pf=ws>0?fmass/ws:0; eq_call=cw>0?acc/cw:0.0;
}
int main(int argc,char**argv){
    std::string w="policy_v2.bin"; long hands=4000; int K=64;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a=="--weights"&&i+1<argc)w=argv[++i]; else if(a=="--hands"&&i+1<argc)hands=atol(argv[++i]);
        else if(a=="--particles"&&i+1<argc)K=atoi(argv[++i]); }
    if(!NET.load(w.c_str())){fprintf(stderr,"нет весов %s\n",w.c_str());return 1;}
    printf("LBR vs агент %s, %ld раздач, %d частиц (постфлоп, LBR IP, стек 100bb).\n",w.c_str(),hands,K);
    double sum=0,sq=0; long n=0;
    for(long H=0;H<hands;H++){
        int idx[52]; for(int i=0;i<52;i++)idx[i]=i;
        for(int k=51;k>0;k--){int j=(int)RNG.next_bounded(k+1);std::swap(idx[k],idx[j]);}
        poker::Card ag1{(uint8_t)idx[0]},ag2{(uint8_t)idx[1]},lb1{(uint8_t)idx[2]},lb2{(uint8_t)idx[3]};
        int bc[5]={idx[4],idx[5],idx[6],idx[7],idx[8]};
        int lbrInv=100,agInv=100,stack=9900;
        // частицы диапазона агента: K случайных комбо без карт LBR
        std::vector<P> R; poker::CardSet dead; dead.add(lb1);dead.add(lb2);
        while((int)R.size()<K){ int x=(int)RNG.next_bounded(52),y=(int)RNG.next_bounded(52); if(x==y)continue;
            poker::Card cx{(uint8_t)x},cy{(uint8_t)y}; if(dead.contains(cx)||dead.contains(cy))continue; R.push_back({cx,cy,1.0}); }
        bool done=false; int net=0;
        for(int street=1; street<=3 && !done; street++){
            poker::CardSet bd{}; int nb=(street==1?3:street==2?4:5); for(int k=0;k<nb;k++)bd.add(poker::Card{(uint8_t)bc[k]});
            for(auto&pr:R) if(bd.contains(pr.a)||bd.contains(pr.b))pr.w=0;
            int pot=lbrInv+agInv; double eq=eq_vs(lb1,lb2,bd,R);
            int betsz=std::min(pot,stack); double pf,eqc; eval_bet(lb1,lb2,bd,R,street,pot,betsz,pf,eqc);
            double evCheck=eq*pot-lbrInv;
            double evBet=pf*agInv+(1-pf)*(eqc*(pot+2*betsz)-(lbrInv+betsz));
            int allin=stack; double pfA,eqcA; eval_bet(lb1,lb2,bd,R,street,pot,allin,pfA,eqcA);
            double evAll=pfA*agInv+(1-pfA)*(eqcA*(pot+2*allin)-(lbrInv+allin));
            double best=std::max(std::max(evCheck,evBet),evAll);
            int action=(best==evAll)?2:(best==evBet)?1:0;
            if(action==0)continue;
            int b=(action==2)?allin:betsz;
            float p[NUM_ACT]; pol(bd,ag1,ag2,street,pot+b,b,0,p);
            double r=frand(),acc=0; int aresp=CALL; for(int kk=0;kk<NUM_ACT;kk++){acc+=p[kk];if(r<=acc){aresp=kk;break;}}
            for(auto&pr:R){ if(pr.w<=0)continue; float pp[NUM_ACT]; pol(bd,pr.a,pr.b,street,pot+b,b,0,pp); pr.w*=(1.0-pp[FOLD]); }
            if(aresp==FOLD){ net=agInv; done=true; break; }
            lbrInv+=b; agInv+=b; stack-=b;
            if(stack<=0){ poker::CardSet f{}; for(int k=0;k<5;k++)f.add(poker::Card{(uint8_t)bc[k]});
                std::vector<poker::Card> hl={lb1,lb2},ho={ag1,ag2}; f.for_each([&](poker::Card c){hl.push_back(c);ho.push_back(c);});
                int v1=EV.evaluate(hl),v2=EV.evaluate(ho); net=(v1>v2)?agInv:(v1==v2?0:-lbrInv); done=true; break; }
        }
        if(!done){ poker::CardSet f{}; for(int k=0;k<5;k++)f.add(poker::Card{(uint8_t)bc[k]});
            std::vector<poker::Card> hl={lb1,lb2},ho={ag1,ag2}; f.for_each([&](poker::Card c){hl.push_back(c);ho.push_back(c);});
            int v1=EV.evaluate(hl),v2=EV.evaluate(ho); net=(v1>v2)?agInv:(v1==v2?0:-lbrInv); }
        double bb=net/100.0; sum+=bb; sq+=bb*bb; n++;
    }
    double m=sum/n,var=sq/n-m*m,se=std::sqrt(std::max(0.0,var)/n);
    printf("\nLBR winrate = %.1f bb/100  (95%% CI +/- %.1f)\n",m*100,1.96*se*100);
    printf("Это НИЖНЯЯ ГРАНИЦА эксплуатируемости агента (GTO дал бы ~0).\n");
    return 0;
}
