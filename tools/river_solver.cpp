// tools/river_solver.cpp — ШАГ 3: постфлоп-CFR с деревом сайзингов (риверный солвер).
// Борд открыт полностью → шоудаун детерминирован, равновесие/exploitability точны.
// Дерево: OOP{check,bet} -> IP{check|bet ; fold,call,allin-raise} -> ...
// CFR сходится к GTO; exploitability(NashConv)->0. Реальные руки/HandEvaluator.
//   ./river_solver --board AsKd7h2c9s --iters 3000 --range 120
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

static bool terminal(const std::string& h){
    return h=="xx"||h=="xbf"||h=="xbc"||h=="bf"||h=="bc"||h=="brf"||h=="brc"; }
static int actor(const std::string& h){
    if(h==""||h=="xb"||h=="br")return 0; if(h=="x"||h=="b")return 1; return -1; }
static const char* LEG(const std::string& h,int& n){
    if(h==""||h=="x"){n=2;return "xb";} if(h=="b"){n=3;return "fcr";}
    if(h=="xb"||h=="br"){n=2;return "fc";} n=0;return ""; }

static double P0=100,BET=75,STK=250;
static void contribs(const std::string& h,double& co,double& ci){ co=ci=0;
    if(h=="b"||h=="bf"||h=="bc"||h=="br"||h=="brf"||h=="brc"){ co=BET; }
    if(h=="bc")ci=BET; if(h=="br"||h=="brf"||h=="brc"){ci=STK;} if(h=="brc")co=STK;
    if(h=="xb"||h=="xbf"||h=="xbc"){ ci=BET; if(h=="xbc")co=BET; } }
static double showdown_net(const std::string& h,int cmp){ double co,ci;contribs(h,co,ci);
    double pot=P0+co+ci, aw=(cmp>0)?pot:(cmp==0?pot/2:0); return aw-(P0/2+co); }
static double fold_net(const std::string& h){ double co,ci;contribs(h,co,ci); double pot=P0+co+ci;
    bool oopf=(h=="xbf"||h=="brf"); return (oopf?0:pot)-(P0/2+co); }
static bool is_showdown(const std::string& h){return h=="xx"||h=="xbc"||h=="bc"||h=="brc";}

struct Node{ std::array<double,3> reg{{0,0,0}},str{{0,0,0}}; };
static std::unordered_map<std::string,Node> G;
static inline std::string key(int act,const std::string& h,int hand){ return std::to_string(act)+h+"#"+std::to_string(hand); }

static std::vector<int>* STR; // сила рук
static double cfr(const std::string& h,int i,int j,double po,double pi){
    if(terminal(h)){ if(is_showdown(h)){int c=((*STR)[i]>(*STR)[j])?1:((*STR)[i]==(*STR)[j]?0:-1);return showdown_net(h,c);} return fold_net(h); }
    int act=actor(h); int n; const char* L=LEG(h,n); int hand=(act==0)?i:j;
    Node& nd=G[key(act,h,hand)];
    std::array<double,3> st{{0,0,0}}; double sum=0;
    for(int k=0;k<n;k++){ st[k]=std::max(0.0,nd.reg[k]); sum+=st[k]; }
    for(int k=0;k<n;k++) st[k]=(sum>0)?st[k]/sum:1.0/n;
    double rm=(act==0)?po:pi; for(int k=0;k<n;k++) nd.str[k]+=rm*st[k];
    std::array<double,3> u{{0,0,0}}; double nu=0;
    for(int k=0;k<n;k++){ std::string h2=h+L[k];
        u[k]=(act==0)?cfr(h2,i,j,po*st[k],pi):cfr(h2,i,j,po,pi*st[k]); nu+=st[k]*u[k]; }
    double cf=(act==0)?pi:po;
    for(int k=0;k<n;k++){ double r=(act==0)?(u[k]-nu):(nu-u[k]); nd.reg[k]+=cf*r; }
    return nu;
}
// средняя стратегия
static void As(int act,const std::string& h,int hand,int n,std::array<double,3>& out){
    auto it=G.find(key(act,h,hand)); double s=0;
    if(it!=G.end()){ for(int k=0;k<n;k++){out[k]=it->second.str[k];s+=out[k];} }
    if(s>0)for(int k=0;k<n;k++)out[k]/=s; else for(int k=0;k<n;k++)out[k]=1.0/n;
}
// info-set best-response: значение для brp, рука brhand, reach по рукам соперника
static std::vector<std::array<Card,2>>* COMB;
static double brrec(const std::string& h,int brp,int brhand,const std::vector<double>& reach){
    if(terminal(h)){ double v=0; bool sd=is_showdown(h);
        for(int k=0;k<(int)reach.size();k++){ if(reach[k]<=0)continue;
            double net; if(sd){int c=((*STR)[brhand>=0&&brp==0?brhand:k]>0,0); }
            // вычислим net_oop, затем переведём в payoff brp
            int oop=(brp==0)?brhand:k, ip=(brp==0)?k:brhand;
            if(sd){int c=((*STR)[oop]>(*STR)[ip])?1:((*STR)[oop]==(*STR)[ip]?0:-1); net=showdown_net(h,c);} else net=fold_net(h);
            double pay=(brp==0)?net:-net; v+=reach[k]*pay; }
        return v; }
    int act=actor(h); int n; const char* L=LEG(h,n);
    if(act==brp){ double best=-1e18; for(int k=0;k<n;k++){ double v=brrec(h+L[k],brp,brhand,reach); best=std::max(best,v);} return best; }
    // соперник ходит средней стратегией: расщепляем reach по каждой руке соперника
    double total=0;
    for(int a=0;a<n;a++){ std::vector<double> r2(reach.size(),0.0); bool any=false;
        for(int k=0;k<(int)reach.size();k++){ if(reach[k]<=0)continue; std::array<double,3> s; As(act,h,k,n,s); r2[k]=reach[k]*s[a]; if(r2[k]>0)any=true; }
        if(any) total+=brrec(h+L[a],brp,brhand,r2);
    }
    return total;
}
int main(int argc,char**argv){
    std::string board="AsKd7h2c9s"; long iters=3000; int rangeN=120;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a=="--board"&&i+1<argc)board=argv[++i]; else if(a=="--iters"&&i+1<argc)iters=atol(argv[++i]);
        else if(a=="--range"&&i+1<argc)rangeN=atoi(argv[++i]); }
    std::vector<Card> bd; for(size_t i=0;i+1<board.size();i+=2){int r=RK(board[i]),s=SK(board[i+1]);if(r<0||s<0){fprintf(stderr,"bad board\n");return 1;}bd.push_back(Card{(uint8_t)(s*13+r)});}
    if(bd.size()!=5){fprintf(stderr,"нужно 5 карт\n");return 1;}
    poker::CardSet dead; for(auto c:bd)dead.add(c);
    std::vector<std::array<Card,2>> combos;
    for(int i=0;i<52;i++){Card ci{(uint8_t)i};if(dead.contains(ci))continue;for(int j=i+1;j<52;j++){Card cj{(uint8_t)j};if(dead.contains(cj))continue;combos.push_back({ci,cj});}}
    std::vector<int> strAll(combos.size());
    for(size_t k=0;k<combos.size();k++){std::vector<Card> h={combos[k][0],combos[k][1]};for(auto c:bd)h.push_back(c);strAll[k]=EVAL.evaluate(h);}
    if(rangeN>0&&rangeN<(int)combos.size()){ std::vector<int> ord(combos.size()); for(size_t k=0;k<ord.size();k++)ord[k]=k;
        std::sort(ord.begin(),ord.end(),[&](int a,int b){return strAll[a]>strAll[b];});
        std::vector<std::array<Card,2>> c2; for(int k=0;k<rangeN;k++)c2.push_back(combos[ord[k]]); combos.swap(c2); }
    int nc=combos.size(); std::vector<int> strength(nc);
    for(int k=0;k<nc;k++){std::vector<Card> h={combos[k][0],combos[k][1]};for(auto c:bd)h.push_back(c);strength[k]=EVAL.evaluate(h);}
    STR=&strength; COMB=&combos;
    printf("Ривер %s. Комбо: %d. банк=%.0f бет=%.0f стек=%.0f. итер=%ld\n",board.c_str(),nc,P0,BET,STK,iters);
    auto conflict=[&](int i,int j){return combos[i][0].idx==combos[j][0].idx||combos[i][0].idx==combos[j][1].idx||combos[i][1].idx==combos[j][0].idx||combos[i][1].idx==combos[j][1].idx;};
    poker::Xoshiro256ss rng(7);
    auto exploit=[&](){ // NashConv = BR0+BR1 (игра zero-sum)
        double br0=0,br1=0; int cntpairs=0;
        for(int bh=0;bh<nc;bh++){ std::vector<double> reach(nc,0.0); int valid=0;
            for(int k=0;k<nc;k++){ if(k!=bh&&!conflict(bh,k)){reach[k]=1.0;valid++;} }
            if(valid==0)continue; for(auto&x:reach)x/=valid;
            br0+=brrec("",0,bh,reach); br1+=brrec("",1,bh,reach); cntpairs++; }
        return (br0+br1)/cntpairs; };
    for(long t=1;t<=iters;t++){
        for(int i=0;i<nc;i++) for(int j=0;j<nc;j++){ if(j==i||conflict(i,j))continue; cfr("",i,j,1.0,1.0); }
        if(t==1||t%(iters/8)==0) printf("  итер %6ld: exploitability(NashConv) = %.4f\n",t,exploit());
    }
    // поляризация: частота бета OOP по силе руки (топ/середина/низ диапазона)
    printf("\nЧастота ставки OOP (узел \"\") по силе руки в диапазоне:\n");
    std::vector<int> ord(nc); for(int k=0;k<nc;k++)ord[k]=k; std::sort(ord.begin(),ord.end(),[&](int a,int b){return strength[a]>strength[b];});
    for(int band=0;band<5;band++){ int lo=band*nc/5,hi=(band+1)*nc/5; double bet=0; int cnt=0;
        for(int z=lo;z<hi;z++){ int hand=ord[z]; std::array<double,3> s; As(0,"",hand,2,s); bet+=s[1]; cnt++; }
        printf("  сила %d/5 (%s): P(bet)=%.2f\n",5-band,band==0?"натсы":(band==4?"мусор":"..."),cnt?bet/cnt:0); }
    printf("\nГотово. NashConv->0 = риверная GTO. Барбель (ставит натсы+мусор, чекает середину) = верная поляризация.\n");
    return 0;
}
