// tools/blueprint.cpp — ШАГ 4d (ядро): абстрактный ПРЕФЛОП-blueprint через CFR.
// Дерево: SB {fold,raise->R, allin}; BB {fold,call,allin}; постфлоп = реализация
// эквити (упрощение — полный deep-stack preflop+flop blueprint компьютер-bound,
// масштаб солвера). Переиспользует матрицу эквити 169x169 (pushfold_gto).
//   ./blueprint --stack 50 --raise 3 --iters 5000
#include "pushfold_gto.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
using namespace pfgto;
static PushFold PF;
static double S=50, R=3; // стек и размер опен-рейза (bb)
static int N;
// EV для SB при реализации эквити (банк=2*invest, вложено invest каждым)
static double realize(double eq,double invest){ return (2*eq-1.0)*invest; }
struct Node{ std::array<double,3> reg{{0,0,0}},str{{0,0,0}}; int nact; };
static std::vector<Node> sbRoot, sbVsAllin, bbVsRaise, bbVsAllin; // по классам
static std::vector<float>* EQ;
static void rmStrat(Node& nd,std::array<double,3>& s){ double sum=0; for(int k=0;k<nd.nact;k++){s[k]=std::max(0.0,nd.reg[k]);sum+=s[k];} for(int k=0;k<nd.nact;k++)s[k]=(sum>0)?s[k]/sum:1.0/nd.nact; }
int main(int argc,char**argv){
    long iters=5000;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a=="--stack"&&i+1<argc)S=atof(argv[++i]); else if(a=="--raise"&&i+1<argc)R=atof(argv[++i]);
        else if(a=="--iters"&&i+1<argc)iters=atol(argv[++i]);}
    PF.init("data/eq169.bin",300); N=PF.N; EQ=&PF.eq;
    auto C=PF.classes(); std::vector<double> w(N); double wsum=0; for(int i=0;i<N;i++){w[i]=C[i].combos;wsum+=w[i];} for(int i=0;i<N;i++)w[i]/=wsum;
    sbRoot.assign(N,{}); sbVsAllin.assign(N,{}); bbVsRaise.assign(N,{}); bbVsAllin.assign(N,{});
    for(int i=0;i<N;i++){sbRoot[i].nact=3;sbVsAllin[i].nact=2;bbVsRaise[i].nact=3;bbVsAllin[i].nact=2;}
    printf("Префлоп-blueprint: стек %.0fbb, опен-рейз %.1fbb, итер %ld\n",S,R,iters);

    for(long t=0;t<iters;t++){
        // средние текущие стратегии не нужны — обновляем регреты напрямую (vanilla, enum пар)
        // Для скорости: для каждого SB-класса i усредняем по BB-классам j (вес w[j]).
        // BB-узлы обновляем симметрично.
        // 1) предвычислить текущие стратегии
        std::vector<std::array<double,3>> sR(N),sA(N),bR(N),bA(N);
        for(int i=0;i<N;i++){rmStrat(sbRoot[i],sR[i]);rmStrat(sbVsAllin[i],sA[i]);rmStrat(bbVsRaise[i],bR[i]);rmStrat(bbVsAllin[i],bA[i]);}
        // 2) BB vs allin (узел "a"): EV колла vs фолда, по SB-олл-ин диапазону
        for(int j=0;j<N;j++){ double num=0,den=0;
            for(int i=0;i<N;i++){ double pa=w[i]*sR[i][2]; if(pa<=0)continue; double e=(*EQ)[(size_t)j*N+i]; num+=pa*realize(e,S); den+=pa; }
            double evC=den>0?num/den:0, evF=-1.0; double n0=bA[j][0]*evC+bA[j][1]*evF;
            bbVsAllin[j].reg[0]+=den*(evC-n0); bbVsAllin[j].reg[1]+=den*(evF-n0);
            bbVsAllin[j].str[0]+=bA[j][0];bbVsAllin[j].str[1]+=bA[j][1]; }
        // 3) BB vs raise (узел "r"): {fold,call(->flop),allin}
        for(int j=0;j<N;j++){ double evF=-1.0; // BB сбрасывает блайнд (1bb) -> -1
            double numC=0,denC=0, numA=0,denA=0;
            for(int i=0;i<N;i++){ double pr=w[i]*sR[i][1]; if(pr<=0)continue; double e=(*EQ)[(size_t)j*N+i];
                numC+=pr*realize(e,R); denC+=pr; // call -> реализация эквити при вложении R
                // BB allin -> SB решает (sA[i]): fold(-R для SB => +R для BB? нет, для BB EV) ...
                double sbFold=sA[i][0], sbCall=sA[i][1];
                double bbEVallin = sbFold*( (double)R ) + sbCall*realize(1-e,S); // BB выигрывает R если SB фолд; иначе шоудаун (эквити BB=1-e)
                numA+=pr*bbEVallin; denA+=pr; }
            double evC=denC>0?numC/denC:0, evA=denA>0?numA/denA:0;
            double n0=bR[j][0]*evF+bR[j][1]*evC+bR[j][2]*evA;
            bbVsRaise[j].reg[0]+=(evF-n0);bbVsRaise[j].reg[1]+=(evC-n0);bbVsRaise[j].reg[2]+=(evA-n0);
            for(int k=0;k<3;k++)bbVsRaise[j].str[k]+=bR[j][k]; }
        // 4) SB vs allin (узел "ra"): после SB raise, BB allin -> SB {fold,call}
        for(int i=0;i<N;i++){ double num=0,den=0; // против BB-олл-ин-рейз-диапазона
            for(int j=0;j<N;j++){ double pa=w[j]*bR[j][2]; if(pa<=0)continue; double e=(*EQ)[(size_t)i*N+j]; num+=pa*realize(e,S); den+=pa; }
            double evCall=den>0?num/den:0, evFold=-R; double n0=sA[i][0]*evFold+sA[i][1]*evCall;
            sbVsAllin[i].reg[0]+=den*(evFold-n0);sbVsAllin[i].reg[1]+=den*(evCall-n0);
            sbVsAllin[i].str[0]+=sA[i][0];sbVsAllin[i].str[1]+=sA[i][1]; }
        // 5) SB root: {fold,raise,allin}
        for(int i=0;i<N;i++){
            double evFold=-0.5;
            // raise: BB отвечает bR[j]
            double evR=0;
            for(int j=0;j<N;j++){ double e=(*EQ)[(size_t)i*N+j];
                double bbFold=bR[j][0],bbCall=bR[j][1],bbAllin=bR[j][2];
                double vCall=realize(e,R);
                double vAllin = sA[i][0]*(-R) + sA[i][1]*realize(e,S); // BB ре-шов: SB fold(-R) или call(шоудаун)
                evR += w[j]*( bbFold*1.0 + bbCall*vCall + bbAllin*vAllin );
            }
            // allin: BB отвечает bA[j]
            double evA=0; for(int j=0;j<N;j++){ double e=(*EQ)[(size_t)i*N+j]; evA+=w[j]*( bA[j][0]*1.0 + bA[j][1]*realize(e,S) ); }
            double n0=sR[i][0]*evFold+sR[i][1]*evR+sR[i][2]*evA;
            sbRoot[i].reg[0]+=(evFold-n0);sbRoot[i].reg[1]+=(evR-n0);sbRoot[i].reg[2]+=(evA-n0);
            for(int k=0;k<3;k++)sbRoot[i].str[k]+=sR[i][k]; }
    }
    // вывод blueprint: SB-частоты
    auto avg=[&](Node& nd,int k){double s=0;for(int z=0;z<nd.nact;z++)s+=nd.str[z];return s>0?nd.str[k]/s:0;};
    auto C2=PF.classes();
    double tot=0,rai=0,alli=0,fld=0;
    for(int i=0;i<N;i++){tot+=C2[i].combos; rai+=C2[i].combos*avg(sbRoot[i],1); alli+=C2[i].combos*avg(sbRoot[i],2); fld+=C2[i].combos*avg(sbRoot[i],0);}
    printf("\nSB-blueprint (%.0fbb): open-raise %.1f%%, allin %.1f%%, fold %.1f%%\n",S,100*rai/tot,100*alli/tot,100*fld/tot);
    auto sh=[&](const char* nm,poker::Card a,poker::Card b){int idx=PushFold::classidx(a,b);
        printf("  %s: raise=%.2f allin=%.2f fold=%.2f\n",nm,avg(sbRoot[idx],1),avg(sbRoot[idx],2),avg(sbRoot[idx],0));};
    auto Cd=[&](int r,int s){return poker::Card{(uint8_t)(s*13+r)};};
    sh("AA",Cd(12,3),Cd(12,2)); sh("KK",Cd(11,3),Cd(11,2)); sh("AKs",Cd(12,0),Cd(11,0));
    sh("76s",Cd(5,0),Cd(4,0)); sh("T2o",Cd(8,0),Cd(0,1)); sh("72o",Cd(5,0),Cd(0,1));
    double bbc=0;for(int j=0;j<N;j++)bbc+=C2[j].combos*(avg(bbVsRaise[j],1)+avg(bbVsRaise[j],2));
    printf("BB продолжает против опен-рейза (call+3bet/allin): %.1f%%\n",100*bbc/tot);
    printf("\n(blueprint-ядро: префлоп-дерево + реализация эквити постфлопа; полный deep-stack\n preflop+flop blueprint требует абстракции борда и компьютинга масштаба солвера.)\n");
    return 0;
}
