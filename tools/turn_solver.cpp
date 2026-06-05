// tools/turn_solver.cpp — ШАГ 4a: turn-солвер (постфлоп-CFR + chance-слой ривера).
// Борд из 4 карт; ставки на тёрне; исход шоудауна = эквити, усреднённая по всем
// ривер-картам (предрасчёт). CFR сходится к GTO тёрна; exploitability->~0.
// (Упрощение: на ривере торговли нет — рука доезжает до шоудауна. Полная
//  многоуличность с ривер-беттингом — шаг 4c/blueprint.)
//   ./turn_solver --board AsKd7h2c --iters 1000 --range 60
#include "../poker_core/cards.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include "../poker_core/deck_rng.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <array>
#include <unordered_map>
#include <algorithm>
#include <cmath>
using poker::Card; using poker::HandEvaluator;
static HandEvaluator EVAL;
static int RK(char c){const char* r="23456789TJQKA";for(int i=0;i<13;i++)if(r[i]==c)return i;return -1;}
static int SK(char c){const char* s="cdhs";for(int i=0;i<4;i++)if(s[i]==c)return i;return -1;}
static bool terminal(const std::string& h){return h=="xx"||h=="xbf"||h=="xbc"||h=="bf"||h=="bc"||h=="brf"||h=="brc";}
static int actor(const std::string& h){if(h==""||h=="xb"||h=="br")return 0;if(h=="x"||h=="b")return 1;return -1;}
static const char* LEG(const std::string& h,int& n){
    if(h==""||h=="x"){n=2;return "xb";} if(h=="b"){n=3;return "fcr";}
    if(h=="xb"||h=="br"){n=2;return "fc";} n=0;return "";}
static double P0=100,BET=75,STK=250;
static void contribs(const std::string& h,double& co,double& ci){ co=ci=0;
    if(h=="b"||h=="bf"||h=="bc"||h=="br"||h=="brf"||h=="brc")co=BET;
    if(h=="bc")ci=BET; if(h=="br"||h=="brf"||h=="brc")ci=STK; if(h=="brc")co=STK;
    if(h=="xb"||h=="xbf"||h=="xbc"){ci=BET; if(h=="xbc")co=BET;} }
static double showdown_net(const std::string& h,double e){ double co,ci;contribs(h,co,ci);
    double pot=P0+co+ci; return pot*e-(P0/2+co); } // e = P(oop выигрывает)+0.5*tie
static double fold_net(const std::string& h){ double co,ci;contribs(h,co,ci); double pot=P0+co+ci;
    bool oopf=(h=="xbf"||h=="brf"); return (oopf?0:pot)-(P0/2+co); }
static bool is_sd(const std::string& h){return h=="xx"||h=="xbc"||h=="bc"||h=="brc";}
struct Node{ std::array<double,3> reg{{0,0,0}},str{{0,0,0}}; };
static std::unordered_map<std::string,Node> G;
static inline std::string key(int a,const std::string& h,int hand){return std::to_string(a)+h+"#"+std::to_string(hand);}
static std::vector<float>* EQ; static int NC;
static double cfr(const std::string& h,int i,int j,double po,double pi){
    if(terminal(h)){ if(is_sd(h))return showdown_net(h,(*EQ)[(size_t)i*NC+j]); return fold_net(h);}
    int act=actor(h);int n;const char* L=LEG(h,n);int hand=(act==0)?i:j;
    Node& nd=G[key(act,h,hand)];
    std::array<double,3> st{{0,0,0}};double sum=0;
    for(int k=0;k<n;k++){st[k]=std::max(0.0,nd.reg[k]);sum+=st[k];}
    for(int k=0;k<n;k++)st[k]=(sum>0)?st[k]/sum:1.0/n;
    double rm=(act==0)?po:pi; for(int k=0;k<n;k++)nd.str[k]+=rm*st[k];
    std::array<double,3> u{{0,0,0}};double nu=0;
    for(int k=0;k<n;k++){std::string h2=h+L[k];u[k]=(act==0)?cfr(h2,i,j,po*st[k],pi):cfr(h2,i,j,po,pi*st[k]);nu+=st[k]*u[k];}
    double cf=(act==0)?pi:po; for(int k=0;k<n;k++){double r=(act==0)?(u[k]-nu):(nu-u[k]);nd.reg[k]+=cf*r;}
    return nu;
}
static void As(int act,const std::string& h,int hand,int n,std::array<double,3>& out){
    auto it=G.find(key(act,h,hand));double s=0;
    if(it!=G.end()){for(int k=0;k<n;k++){out[k]=it->second.str[k];s+=out[k];}}
    if(s>0)for(int k=0;k<n;k++)out[k]/=s; else for(int k=0;k<n;k++)out[k]=1.0/n;
}
static double brrec(const std::string& h,int brp,int bh,const std::vector<double>& reach){
    if(terminal(h)){double v=0;bool sd=is_sd(h);
        for(int k=0;k<(int)reach.size();k++){if(reach[k]<=0)continue;
            int oop=(brp==0)?bh:k,ip=(brp==0)?k:bh; double net;
            if(sd){double e=(*EQ)[(size_t)oop*NC+ip];net=showdown_net(h,e);} else net=fold_net(h);
            v+=reach[k]*((brp==0)?net:-net);} return v;}
    int act=actor(h);int n;const char* L=LEG(h,n);
    if(act==brp){double best=-1e18;for(int k=0;k<n;k++)best=std::max(best,brrec(h+L[k],brp,bh,reach));return best;}
    double total=0;
    for(int a=0;a<n;a++){std::vector<double> r2(reach.size(),0.0);bool any=false;
        for(int k=0;k<(int)reach.size();k++){if(reach[k]<=0)continue;std::array<double,3> s;As(act,h,k,n,s);r2[k]=reach[k]*s[a];if(r2[k]>0)any=true;}
        if(any)total+=brrec(h+L[a],brp,bh,r2);}
    return total;
}
int main(int argc,char**argv){
    std::string board="AsKd7h2c"; long iters=1000; int rangeN=60;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a=="--board"&&i+1<argc)board=argv[++i];else if(a=="--iters"&&i+1<argc)iters=atol(argv[++i]);
        else if(a=="--range"&&i+1<argc)rangeN=atoi(argv[++i]);}
    std::vector<Card> bd;for(size_t i=0;i+1<board.size();i+=2){int r=RK(board[i]),s=SK(board[i+1]);if(r<0||s<0){fprintf(stderr,"bad board\n");return 1;}bd.push_back(Card{(uint8_t)(s*13+r)});}
    if(bd.size()!=4){fprintf(stderr,"нужно 4 карты борда (тёрн)\n");return 1;}
    poker::CardSet dead;for(auto c:bd)dead.add(c);
    std::vector<std::array<Card,2>> combos;
    for(int i=0;i<52;i++){Card ci{(uint8_t)i};if(dead.contains(ci))continue;for(int j=i+1;j<52;j++){Card cj{(uint8_t)j};if(dead.contains(cj))continue;combos.push_back({ci,cj});}}
    // топ-N по текущей силе на тёрне (4 карты + 2) для диапазона
    auto strnow=[&](std::array<Card,2>& c){std::vector<Card> h={c[0],c[1]};for(auto x:bd)h.push_back(x);
        // на 4 картах оценим по лучшей из доборов? используем 6-карт оценку как прокси силы
        return EVAL.evaluate(h);};
    if(rangeN>0&&rangeN<(int)combos.size()){std::sort(combos.begin(),combos.end(),[&](auto&A,auto&B){auto a=A,b=B;return strnow(a)>strnow(b);});combos.resize(rangeN);}
    int nc=combos.size(); NC=nc;
    printf("Тёрн %s. Комбо: %d. банк=%.0f бет=%.0f стек=%.0f. итер=%ld\n",board.c_str(),nc,P0,BET,STK,iters);
    // матрица эквити по ривер-ранаутам: eq[i][j]=P(i бьёт j) по всем ривер-картам
    std::vector<float> eq((size_t)nc*nc,0.5f); EQ=&eq;
    std::vector<int> riv; for(int r=0;r<52;r++){Card cr{(uint8_t)r};if(!dead.contains(cr))riv.push_back(r);}
    printf("[eq] матрица %dx%d по %d ривер-картам...\n",nc,nc,(int)riv.size());
    for(int i=0;i<nc;i++){
        for(int j=0;j<nc;j++){ if(j==i)continue;
            // конфликт карт
            if(combos[i][0].idx==combos[j][0].idx||combos[i][0].idx==combos[j][1].idx||combos[i][1].idx==combos[j][0].idx||combos[i][1].idx==combos[j][1].idx){eq[(size_t)i*nc+j]=0.5f;continue;}
            double acc=0;int tot=0;
            for(int rc:riv){ Card cr{(uint8_t)rc};
                if(cr.idx==combos[i][0].idx||cr.idx==combos[i][1].idx||cr.idx==combos[j][0].idx||cr.idx==combos[j][1].idx)continue;
                std::vector<Card> hi={combos[i][0],combos[i][1]},hj={combos[j][0],combos[j][1]};
                for(auto x:bd){hi.push_back(x);hj.push_back(x);} hi.push_back(cr);hj.push_back(cr);
                int vi=EVAL.evaluate(hi),vj=EVAL.evaluate(hj); acc+=(vi>vj)?1.0:(vi==vj?0.5:0.0); tot++; }
            eq[(size_t)i*nc+j]=tot?(float)(acc/tot):0.5f;
        }
        if(i%20==0)printf("  строка %d/%d\n",i,nc);
    }
    auto conflict=[&](int i,int j){return combos[i][0].idx==combos[j][0].idx||combos[i][0].idx==combos[j][1].idx||combos[i][1].idx==combos[j][0].idx||combos[i][1].idx==combos[j][1].idx;};
    auto exploit=[&](){double br0=0,br1=0;int cnt=0;
        for(int bh=0;bh<nc;bh++){std::vector<double> reach(nc,0.0);int v=0;
            for(int k=0;k<nc;k++)if(k!=bh&&!conflict(bh,k)){reach[k]=1.0;v++;} if(!v)continue; for(auto&x:reach)x/=v;
            br0+=brrec("",0,bh,reach);br1+=brrec("",1,bh,reach);cnt++;} return (br0+br1)/cnt;};
    for(long t=1;t<=iters;t++){
        for(int i=0;i<nc;i++)for(int j=0;j<nc;j++){if(j==i||conflict(i,j))continue;cfr("",i,j,1.0,1.0);}
        if(t==1||t%(iters/6)==0)printf("  итер %5ld: exploitability(NashConv) = %.4f\n",t,exploit());
    }
    printf("\nЧастота ставки OOP по силе руки (тёрн):\n");
    std::vector<int> ord(nc);for(int k=0;k<nc;k++)ord[k]=k;std::sort(ord.begin(),ord.end(),[&](int a,int b){auto x=combos[a],y=combos[b];return strnow(x)>strnow(y);});
    for(int band=0;band<5;band++){int lo=band*nc/5,hi=(band+1)*nc/5;double bet=0;int cnt=0;
        for(int z=lo;z<hi;z++){std::array<double,3> s;As(0,"",ord[z],2,s);bet+=s[1];cnt++;}
        printf("  сила %d/5: P(bet)=%.2f\n",5-band,cnt?bet/cnt:0);}
    printf("Готово. exploitability->~0 = GTO тёрна. Барбель = верная поляризация.\n");
    return 0;
}
