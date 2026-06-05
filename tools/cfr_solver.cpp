// tools/cfr_solver.cpp
// CFR-движок, ДОКАЗУЕМО сходящийся к точному GTO (равновесию Нэша).
// Бенчмарк: Kuhn poker — игра, где равновесие известно аналитически.
// Замеряем exploitability (NashConv) — она должна -> 0. Это проверяемое ядро
// «GTO-силы»: тот же алгоритм масштабируется на Leduc/NLHE (см. README, раздел CFR).
//   g++ -std=c++20 -O3 cfr_solver.cpp -o cfr_solver && ./cfr_solver --iters 200000
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <unordered_map>
#include <array>
#include <algorithm>
#include <cmath>
struct Node{ std::array<double,2> regret{{0,0}},strat{{0,0}}; }; // 0=pass,1=bet
static std::unordered_map<std::string,Node> N;
static double term(const std::string& h,int c0,int c1,bool& is){
    int hi=(c0>c1)?0:1; is=true;
    if(h=="pp") return (hi==0)?1:-1;
    if(h=="bp") return 1;
    if(h=="pbp")return -1;
    if(h=="bb") return (hi==0)?2:-2;
    if(h=="pbb")return (hi==0)?2:-2;
    is=false; return 0;
}
static double cfr(const std::string& h,int c0,int c1,double p0,double p1){
    bool is; double tv=term(h,c0,c1,is); if(is)return tv;
    int actor=(int)h.size()%2, mycard=(actor==0)?c0:c1;
    std::string key=std::to_string(mycard)+h; Node& nd=N[key];
    std::array<double,2> s; double sum=0;
    for(int a=0;a<2;a++){ s[a]=std::max(0.0,nd.regret[a]); sum+=s[a]; }
    for(int a=0;a<2;a++) s[a]=(sum>0)?s[a]/sum:0.5;
    double rm=(actor==0)?p0:p1; for(int a=0;a<2;a++) nd.strat[a]+=rm*s[a];
    std::array<double,2> u{{0,0}}; double nu=0;
    for(int a=0;a<2;a++){ std::string h2=h+(a?'b':'p');
        u[a]=(actor==0)?cfr(h2,c0,c1,p0*s[a],p1):cfr(h2,c0,c1,p0,p1*s[a]); nu+=s[a]*u[a]; }
    double cf=(actor==0)?p1:p0;
    for(int a=0;a<2;a++){ double r=(actor==0)?(u[a]-nu):(nu-u[a]); nd.regret[a]+=cf*r; }
    return nu;
}
static double As(const std::string& key,int a){ auto it=N.find(key); if(it==N.end())return 0.5;
    double s=it->second.strat[0]+it->second.strat[1]; return s>0?it->second.strat[a]/s:0.5; }
// Корректный best-response: BR-игрок НЕ видит карту соперника; максимизирует
// ожидание по распределению чужих карт (reach соперника по его стратегии).
static double payoff_brp(const std::string& h,int brcard,int oc,int brp){
    int c0=(brp==0)?brcard:oc, c1=(brp==0)?oc:brcard; bool is; double u0=term(h,c0,c1,is);
    return (brp==0)?u0:-u0;
}
static double brrec(const std::string& h,int brp,int brcard,const std::array<double,3>& reach){
    bool is; { int c0=(brp==0)?brcard:0,c1=(brp==0)?0:brcard; (void)c0;(void)c1; }
    // терминал?
    bool term_any=false; { bool t; term(h,brcard,(brcard+1)%3,t); term_any=t; }
    if(term_any){ double v=0; for(int oc=0;oc<3;oc++) if(reach[oc]>0&&oc!=brcard) v+=reach[oc]*payoff_brp(h,brcard,oc,brp); return v; }
    int actor=(int)h.size()%2;
    if(actor==brp){ double best=-1e18;
        for(int a=0;a<2;a++){ double v=brrec(h+(a?'b':'p'),brp,brcard,reach); best=std::max(best,v); } return best; }
    // ход соперника: расщепляем reach по его средней стратегии (для каждой его карты oc)
    double total=0;
    for(int a=0;a<2;a++){ std::array<double,3> r2{{0,0,0}}; bool any=false;
        for(int oc=0;oc<3;oc++){ if(oc==brcard||reach[oc]<=0)continue;
            double pr=As(std::to_string(oc)+h,a); r2[oc]=reach[oc]*pr; if(r2[oc]>0)any=true; }
        if(any) total+=brrec(h+(a?'b':'p'),brp,brcard,r2);
    }
    return total;
}
static double br_value(int brp){
    double tot=0; // E по своей карте (1/3) и карте соперника (1/2 из оставшихся)
    for(int bc=0;bc<3;bc++){ std::array<double,3> reach{{0,0,0}};
        for(int oc=0;oc<3;oc++) if(oc!=bc) reach[oc]=0.5;
        tot += (1.0/3.0)*brrec("",brp,bc,reach);
    }
    return tot;
}
int main(int argc,char**argv){
    long iters=200000; for(int i=1;i<argc;i++) if(!strcmp(argv[i],"--iters")&&i+1<argc)iters=atol(argv[++i]);
    printf("CFR на Kuhn poker, %ld итераций. exploitability (NashConv) -> 0 = точный GTO.\n",iters);
    for(long t=1;t<=iters;t++){
        for(int c0=0;c0<3;c0++)for(int c1=0;c1<3;c1++) if(c0!=c1) cfr("",c0,c1,1,1);
        if(t==1||t%(iters/10)==0){ double e=br_value(0)+br_value(1); // NashConv = BR0+BR1 (u0+u1=0)
            printf("  итер %8ld: exploitability(NashConv) = %.6f\n",t,e); }
    }
    printf("\nравновесная стратегия (открытие P0): J bet=%.3f  Q bet=%.3f  K bet=%.3f\n",As("0",1),As("1",1),As("2",1));
    printf("эталон GTO: K бетит ровно 3x от блефа J; Q всегда чек. NashConv->0 => точный GTO.\n");
    return 0;
}
