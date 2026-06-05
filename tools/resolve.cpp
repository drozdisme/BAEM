// tools/resolve.cpp — ШАГ 4c: in-play re-solving. Агент в реальном споте решает
// своё поддерево (river=5 карт детерминированно, turn=4 карты с chance-слоем ривера)
// CFR'ом на лету и выдаёт GTO-стратегию для СВОЕЙ руки. Это превращает офлайн-солверы
// в живой мозг: вызываешь на спобе -> получаешь GTO-действие.
//   ./resolve --board AsKd7h2c9s --hero QhJh --pos oop --iters 800 --range 60
//   ./resolve --board AsKd7h2c   --hero QhJh ...   (тёрн: ривер как chance)
#include "../poker_core/cards.hpp"
#include "../poker_core/hand_evaluator.hpp"
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
static void contribs(const std::string& h,double& co,double& ci){co=ci=0;
    if(h=="b"||h=="bf"||h=="bc"||h=="br"||h=="brf"||h=="brc")co=BET;
    if(h=="bc")ci=BET; if(h=="br"||h=="brf"||h=="brc")ci=STK; if(h=="brc")co=STK;
    if(h=="xb"||h=="xbf"||h=="xbc"){ci=BET;if(h=="xbc")co=BET;} }
static double sd_net(const std::string& h,double e){double co,ci;contribs(h,co,ci);double pot=P0+co+ci;return pot*e-(P0/2+co);}
static double fold_net(const std::string& h){double co,ci;contribs(h,co,ci);double pot=P0+co+ci;bool oopf=(h=="xbf"||h=="brf");return (oopf?0:pot)-(P0/2+co);}
static bool is_sd(const std::string& h){return h=="xx"||h=="xbc"||h=="bc"||h=="brc";}
struct Node{std::array<double,3> reg{{0,0,0}},str{{0,0,0}};};
static std::unordered_map<std::string,Node> G;
static inline std::string key(int a,const std::string& h,int hand){return std::to_string(a)+h+"#"+std::to_string(hand);}
static std::vector<float>* EQ; static int NC;
static double cfr(const std::string& h,int i,int j,double po,double pi){
    if(terminal(h)){if(is_sd(h))return sd_net(h,(*EQ)[(size_t)i*NC+j]);return fold_net(h);}
    int act=actor(h);int n;const char* L=LEG(h,n);int hand=(act==0)?i:j; Node& nd=G[key(act,h,hand)];
    std::array<double,3> st{{0,0,0}};double sum=0;for(int k=0;k<n;k++){st[k]=std::max(0.0,nd.reg[k]);sum+=st[k];}
    for(int k=0;k<n;k++)st[k]=(sum>0)?st[k]/sum:1.0/n; double rm=(act==0)?po:pi;for(int k=0;k<n;k++)nd.str[k]+=rm*st[k];
    std::array<double,3> u{{0,0,0}};double nu=0;
    for(int k=0;k<n;k++){std::string h2=h+L[k];u[k]=(act==0)?cfr(h2,i,j,po*st[k],pi):cfr(h2,i,j,po,pi*st[k]);nu+=st[k]*u[k];}
    double cf=(act==0)?pi:po;for(int k=0;k<n;k++){double r=(act==0)?(u[k]-nu):(nu-u[k]);nd.reg[k]+=cf*r;}
    return nu;
}
static void Avg(int act,const std::string& h,int hand,int n,std::array<double,3>& o){auto it=G.find(key(act,h,hand));double s=0;
    if(it!=G.end()){for(int k=0;k<n;k++){o[k]=it->second.str[k];s+=o[k];}} if(s>0)for(int k=0;k<n;k++)o[k]/=s;else for(int k=0;k<n;k++)o[k]=1.0/n;}
int main(int argc,char**argv){
    std::string board="AsKd7h2c9s",hero="QhJh",pos="oop"; long iters=800; int rangeN=60;
    for(int i=1;i<argc;i++){std::string a=argv[i];
        if(a=="--board"&&i+1<argc)board=argv[++i];else if(a=="--hero"&&i+1<argc)hero=argv[++i];
        else if(a=="--pos"&&i+1<argc)pos=argv[++i];else if(a=="--iters"&&i+1<argc)iters=atol(argv[++i]);
        else if(a=="--range"&&i+1<argc)rangeN=atoi(argv[++i]);}
    std::vector<Card> bd;for(size_t i=0;i+1<board.size();i+=2){int r=RK(board[i]),s=SK(board[i+1]);bd.push_back(Card{(uint8_t)(s*13+r)});}
    if(bd.size()!=4&&bd.size()!=5){fprintf(stderr,"борд 4 (тёрн) или 5 (ривер) карт\n");return 1;}
    int hr=RK(hero[0]),hs=SK(hero[1]),hr2=RK(hero[2]),hs2=SK(hero[3]);
    Card H1{(uint8_t)(hs*13+hr)},H2{(uint8_t)(hs2*13+hr2)};
    bool turn=(bd.size()==4);
    poker::CardSet dead;for(auto c:bd)dead.add(c);
    std::vector<std::array<Card,2>> combos;
    for(int i=0;i<52;i++){Card ci{(uint8_t)i};if(dead.contains(ci))continue;for(int j=i+1;j<52;j++){Card cj{(uint8_t)j};if(dead.contains(cj))continue;combos.push_back({ci,cj});}}
    auto strnow=[&](std::array<Card,2>& c){std::vector<Card> h={c[0],c[1]};for(auto x:bd)h.push_back(x);return EVAL.evaluate(h);};
    // гарантируем, что рука героя в диапазоне
    int heroIdx=-1;
    if(rangeN>0&&rangeN<(int)combos.size()){std::sort(combos.begin(),combos.end(),[&](auto&A,auto&B){auto a=A,b=B;return strnow(a)>strnow(b);});
        std::vector<std::array<Card,2>> rep; double stride=(double)combos.size()/rangeN; for(int k=0;k<rangeN;k++)rep.push_back(combos[(int)(k*stride)]); combos.swap(rep);}
    // добавить руку героя, если её нет
    for(int k=0;k<(int)combos.size();k++) if((combos[k][0].idx==H1.idx&&combos[k][1].idx==H2.idx)||(combos[k][0].idx==H2.idx&&combos[k][1].idx==H1.idx))heroIdx=k;
    if(heroIdx<0){combos.push_back({H1,H2});heroIdx=combos.size()-1;}
    int nc=combos.size();NC=nc;
    printf("Re-solve: %s %s, рука героя %s (%s). Комбо: %d, итер: %ld\n",turn?"ТЁРН":"РИВЕР",board.c_str(),hero.c_str(),pos.c_str(),nc,iters);
    // матрица эквити
    std::vector<float> eq((size_t)nc*nc,0.5f);EQ=&eq;
    std::vector<int> riv; if(turn){for(int r=0;r<52;r++){Card cr{(uint8_t)r};if(!dead.contains(cr))riv.push_back(r);}}
    for(int i=0;i<nc;i++)for(int j=0;j<nc;j++){if(j==i)continue;
        if(combos[i][0].idx==combos[j][0].idx||combos[i][0].idx==combos[j][1].idx||combos[i][1].idx==combos[j][0].idx||combos[i][1].idx==combos[j][1].idx){eq[(size_t)i*nc+j]=0.5f;continue;}
        if(!turn){std::vector<Card> hi={combos[i][0],combos[i][1]},hj={combos[j][0],combos[j][1]};for(auto x:bd){hi.push_back(x);hj.push_back(x);}int vi=EVAL.evaluate(hi),vj=EVAL.evaluate(hj);eq[(size_t)i*nc+j]=(vi>vj)?1.0f:(vi==vj?0.5f:0.0f);}
        else{double acc=0;int tot=0;for(int rc:riv){Card cr{(uint8_t)rc};if(cr.idx==combos[i][0].idx||cr.idx==combos[i][1].idx||cr.idx==combos[j][0].idx||cr.idx==combos[j][1].idx)continue;std::vector<Card> hi={combos[i][0],combos[i][1]},hj={combos[j][0],combos[j][1]};for(auto x:bd){hi.push_back(x);hj.push_back(x);}hi.push_back(cr);hj.push_back(cr);int vi=EVAL.evaluate(hi),vj=EVAL.evaluate(hj);acc+=(vi>vj)?1.0:(vi==vj?0.5:0.0);tot++;}eq[(size_t)i*nc+j]=tot?(float)(acc/tot):0.5f;}
    }
    auto conflict=[&](int i,int j){return combos[i][0].idx==combos[j][0].idx||combos[i][0].idx==combos[j][1].idx||combos[i][1].idx==combos[j][0].idx||combos[i][1].idx==combos[j][1].idx;};
    for(long t=1;t<=iters;t++) for(int i=0;i<nc;i++)for(int j=0;j<nc;j++){if(j==i||conflict(i,j))continue;cfr("",i,j,1.0,1.0);}
    // вывод GTO-действия для руки героя в корневом узле (его позиция)
    int act=(pos=="oop")?0:1;
    std::string root = (act==0)? "" : "x"; // OOP в "", IP — после чека (узел "x"); для простоты покажем оба
    int n; const char* L; std::array<double,3> s;
    printf("\nGTO-стратегия героя в корне:\n");
    if(act==0){ L=LEG("",n); Avg(0,"",heroIdx,n,s);
        printf("  OOP узел \"\": "); for(int k=0;k<n;k++)printf("%c=%.2f ",L[k],s[k]); printf("\n"); }
    else { L=LEG("x",n); Avg(1,"x",heroIdx,n,s);
        printf("  IP после чека \"x\": "); for(int k=0;k<n;k++)printf("%c=%.2f ",L[k],s[k]); printf("\n");
        L=LEG("b",n); std::array<double,3> s2; Avg(1,"b",heroIdx,n,s2);
        printf("  IP против бета \"b\": "); for(int k=0;k<n;k++)printf("%c=%.2f ",L[k],s2[k]); printf("\n"); }
    double e_vs_field=0;int cnt=0; for(int j=0;j<nc;j++){if(j==heroIdx||conflict(heroIdx,j))continue;e_vs_field+=eq[(size_t)heroIdx*nc+j];cnt++;}
    printf("  (эквити руки против диапазона ~%.0f%%)\n",cnt?100.0*e_vs_field/cnt:50.0);
    printf("\nЭто живой re-solve: агент вызывает это на спобе и берёт GTO-действие.\n");
    return 0;
}
