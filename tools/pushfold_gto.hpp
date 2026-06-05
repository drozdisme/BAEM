#pragma once
// tools/pushfold_gto.hpp — ЖИВОЙ GTO push/fold модуль для агента (шаг 4b).
// Самодостаточно: 169 классов, матрица эквити (кэш data/eq169.bin), быстрый CFR
// push/fold по требуемой глубине стека. Агент на коротком HU-префлопе играет
// ТОЧНУЮ GTO push/fold (а не выученный приор).
#include "../poker_core/cards.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include "../poker_core/deck_rng.hpp"
#include <vector>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <unordered_map>

namespace pfgto {

struct PushFold {
    poker::HandEvaluator ev; poker::Xoshiro256ss rng{2024};
    int N=169; std::vector<float> eq;     // 169x169
    std::vector<double> w;                 // веса классов (combos)
    bool ready=false;
    std::unordered_map<int,std::vector<char>> pushCache, callCache; // по стеку(bb)

    // индекс класса руки (порядок: для hi от A до 2: пара, затем suited(lo), затем offsuit(lo))
    static int classidx(poker::Card a,poker::Card b){
        int r1=a.rank(),r2=b.rank(),s1=a.suit(),s2=b.suit();
        int hi=std::max(r1,r2),lo=std::min(r1,r2); bool suited=(s1==s2),pair=(r1==r2);
        int idx=0;
        for(int H=12;H>=0;H--){
            if(H==hi){ if(pair)return idx; idx++; // пара этого ранга — позиция 0 блока
                // suited блок: lo от H-1..0
                for(int L=H-1;L>=0;L--){ if(suited&&lo==L)return idx; idx++; }
                for(int L=H-1;L>=0;L--){ if(!suited&&!pair&&lo==L)return idx; idx++; }
                return idx; // не должно
            } else { idx++; for(int L=H-1;L>=0;L--)idx++; for(int L=H-1;L>=0;L--)idx++; }
        }
        return 0;
    }
    // построить combos класса (для эквити) — представитель
    static void rep(int hi,int lo,bool suited,bool pair,int var,poker::Card&a,poker::Card&b){
        int s1=var%4,s2=(var/4)%4;
        if(pair){ if(s2==s1)s2=(s1+1)%4; a=poker::Card{(uint8_t)(s1*13+hi)};b=poker::Card{(uint8_t)(s2*13+lo)};}
        else if(suited){ a=poker::Card{(uint8_t)(s1*13+hi)};b=poker::Card{(uint8_t)(s1*13+lo)};}
        else { if(s2==s1)s2=(s1+1)%4; a=poker::Card{(uint8_t)(s1*13+hi)};b=poker::Card{(uint8_t)(s2*13+lo)};}
    }
    struct Cls{int hi,lo;bool suited,pair;int combos;};
    std::vector<Cls> classes(){ std::vector<Cls> v;
        for(int hi=12;hi>=0;hi--){ v.push_back({hi,hi,false,true,6});
            for(int lo=hi-1;lo>=0;lo--)v.push_back({hi,lo,true,false,4});
            for(int lo=hi-1;lo>=0;lo--)v.push_back({hi,lo,false,false,12}); }
        return v; }

    float eq_ij(const Cls&ci,const Cls&cj,int samp){
        int tot=0; double acc=0;
        for(int s=0;s<samp;s++){ poker::Card a,b,c,d; rep(ci.hi,ci.lo,ci.suited,ci.pair,s,a,b); rep(cj.hi,cj.lo,cj.suited,cj.pair,s/4+1,c,d);
            std::array<int,4> id={a.idx,b.idx,c.idx,d.idx}; bool cl=false;
            for(int x=0;x<4&&!cl;x++)for(int y=x+1;y<4;y++)if(id[x]==id[y])cl=true; if(cl)continue;
            bool used[52]={false}; for(int z:id)used[z]=true; int board[5],g=0;
            while(g<5){int r=(int)rng.next_bounded(52); if(!used[r]){used[r]=true;board[g++]=r;}}
            std::vector<poker::Card> h1={a,b},h2={c,d};
            for(int k=0;k<5;k++){h1.push_back(poker::Card{(uint8_t)board[k]});h2.push_back(poker::Card{(uint8_t)board[k]});}
            int v1=ev.evaluate(h1),v2=ev.evaluate(h2); acc+=(v1>v2)?1.0:(v1==v2?0.5:0.0); tot++; }
        return tot?(float)(acc/tot):0.5f;
    }
    void init(const char* cache="data/eq169.bin",int samp=300){
        if(ready)return; auto C=classes(); eq.assign((size_t)N*N,0.5f); w.assign(N,0);
        for(int i=0;i<N;i++)w[i]=C[i].combos;
        FILE* f=std::fopen(cache,"rb");
        if(f){ size_t rd=std::fread(eq.data(),sizeof(float),(size_t)N*N,f); std::fclose(f); if(rd==(size_t)N*N){ready=true;return;} }
        for(int i=0;i<N;i++)for(int j=i;j<N;j++){ float e=eq_ij(C[i],C[j],samp); eq[(size_t)i*N+j]=e; eq[(size_t)j*N+i]=1-e; }
        std::system("mkdir -p data"); FILE* wf=std::fopen(cache,"wb"); if(wf){std::fwrite(eq.data(),sizeof(float),(size_t)N*N,wf);std::fclose(wf);}
        ready=true;
    }
    // решить push/fold для стека S (bb), кэшировать булевы диапазоны
    void solve(int Sbb,int iters=3000){
        if(pushCache.count(Sbb))return; if(!ready)init();
        double S=Sbb, wsum=0; for(int i=0;i<N;i++)wsum+=w[i]; std::vector<double> wn(N); for(int i=0;i<N;i++)wn[i]=w[i]/wsum;
        std::vector<std::array<double,2>> rS(N,{0,0}),sS(N,{0,0}),rB(N,{0,0}),sB(N,{0,0});
        auto strat=[&](std::array<double,2>&r){double a=std::max(0.0,r[0]),b=std::max(0.0,r[1]),s=a+b;return std::array<double,2>{s>0?a/s:0.5,s>0?b/s:0.5};};
        for(int t=0;t<iters;t++){ std::vector<std::array<double,2>> stS(N),stB(N);
            for(int i=0;i<N;i++){stS[i]=strat(rS[i]);stB[i]=strat(rB[i]);}
            for(int j=0;j<N;j++){ double num=0,den=0; for(int i=0;i<N;i++){double pp=wn[i]*stS[i][0];if(pp<=0)continue;double e=eq[(size_t)j*N+i];num+=pp*((2*e-1)*S);den+=pp;}
                double evC=den>0?num/den:0,evF=-1.0,c0=stB[j][0],c1=stB[j][1],node=c0*evC+c1*evF;
                rB[j][0]+=den*(evC-node);rB[j][1]+=den*(evF-node);sB[j][0]+=c0;sB[j][1]+=c1; }
            for(int i=0;i<N;i++){ double evP=0; for(int j=0;j<N;j++){double pc=stB[j][0];double e=eq[(size_t)i*N+j];evP+=wn[j]*(pc*((2*e-1)*S)+(1-pc)*1.0);}
                double evF=-0.5,c0=stS[i][0],c1=stS[i][1],node=c0*evP+c1*evF;
                rS[i][0]+=(evP-node);rS[i][1]+=(evF-node);sS[i][0]+=c0;sS[i][1]+=c1; } }
        std::vector<char> push(N),call(N);
        for(int i=0;i<N;i++){double s=sS[i][0]+sS[i][1];push[i]=(s>0&&sS[i][0]/s>0.5)?1:0;}
        for(int j=0;j<N;j++){double s=sB[j][0]+sB[j][1];call[j]=(s>0&&sB[j][0]/s>0.5)?1:0;}
        pushCache[Sbb]=push; callCache[Sbb]=call;
    }
    bool sb_push(poker::Card a,poker::Card b,int Sbb){ solve(Sbb); return pushCache[Sbb][classidx(a,b)]!=0; }
    bool bb_call(poker::Card a,poker::Card b,int Sbb){ solve(Sbb); return callCache[Sbb][classidx(a,b)]!=0; }
    double sb_push_pct(int Sbb){ solve(Sbb); auto&p=pushCache[Sbb]; auto C=classes(); double tot=0,push=0;
        for(int i=0;i<N;i++){tot+=C[i].combos; if(p[i])push+=C[i].combos;} return 100.0*push/tot; }
    double bb_call_pct(int Sbb){ solve(Sbb); auto&p=callCache[Sbb]; auto C=classes(); double tot=0,c=0;
        for(int i=0;i<N;i++){tot+=C[i].combos; if(p[i])c+=C[i].combos;} return 100.0*c/tot; }
};

} // namespace pfgto
