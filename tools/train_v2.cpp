// tools/train_v2.cpp — обучение политики с сайзингом (8 классов) и фичами рука×борд.
//   ./train_v2 --data prepared_v2.txt --out policy_v2.bin --epochs 40
#include "poker_v2.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <chrono>

int main(int argc,char**argv){
    std::string data="prepared_v2.txt",out="policy_v2.bin";
    int epochs=40,batch=128,patience=6; double lr=6e-4,val_frac=0.1; uint64_t seed=1;
    for(int i=1;i<argc;i++){std::string a=argv[i];auto nx=[&]{return (i+1<argc)?std::string(argv[++i]):std::string();};
        if(a=="--data")data=nx();else if(a=="--out")out=nx();else if(a=="--epochs")epochs=atoi(nx().c_str());
        else if(a=="--lr")lr=atof(nx().c_str());else if(a=="--batch")batch=atoi(nx().c_str());
        else if(a=="--val-frac")val_frac=atof(nx().c_str());else if(a=="--patience")patience=atoi(nx().c_str());
        else if(a=="--seed")seed=strtoull(nx().c_str(),0,10);}

    poker::HandEvaluator ev; pv2::EncoderV2 enc(&ev);
    printf("[load] %s ...\n",data.c_str());
    std::ifstream f(data); if(!f){fprintf(stderr,"no file %s\n",data.c_str());return 1;}
    std::vector<std::array<float,pv2::FV2_DIM>> X; std::vector<int> Y;
    std::string line; long bad=0; std::array<long,pv2::NUM_ACT> cc{};
    while(std::getline(f,line)){ if(line.empty())continue; pv2::Row r;
        if(!pv2::parse_v2(line,r)){bad++;continue;}
        X.push_back(enc.encode(r.board,r.c1,r.c2,r.street,r.pot,r.tocall,r.hero_ip,r.agg));
        Y.push_back(r.klabel); cc[r.klabel]++; }
    printf("[load] samples=%zu bad=%ld\n",X.size(),bad);
    if(X.size()<100){fprintf(stderr,"too few\n");return 1;}
    printf("[load] class dist:\n");
    for(int c=0;c<pv2::NUM_ACT;c++) printf("   %-6s %8ld (%.1f%%)\n",pv2::ANAME[c],cc[c],100.0*cc[c]/X.size());

    // веса классов (обратная частота, среднее≈1, кап 4x чтобы редкие классы не доминировали)
    std::array<float,pv2::NUM_ACT> cw; { float m=0;int nz=0;
        for(int c=0;c<pv2::NUM_ACT;c++){ cw[c]=cc[c]?(float)X.size()/(pv2::NUM_ACT*cc[c]):0.0f; if(cc[c]){m+=cw[c];nz++;} }
        m/=std::max(1,nz); for(int c=0;c<pv2::NUM_ACT;c++) cw[c]= cc[c]? std::min(4.0f, cw[c]/m) : 0.0f; }
    printf("[weights] class weights: ");
    for(int c=0;c<pv2::NUM_ACT;c++) printf("%s=%.2f ",pv2::ANAME[c],cw[c]); printf("\n");

    std::mt19937_64 rng(seed);
    std::vector<size_t> idx(X.size()); for(size_t i=0;i<idx.size();i++)idx[i]=i;
    std::shuffle(idx.begin(),idx.end(),rng);
    size_t nval=(size_t)(X.size()*val_frac), ntr=X.size()-nval;
    printf("[split] train=%zu val=%zu\n",ntr,nval);

    pv2::PolicyNet net; net.lr=lr;
    auto eval_val=[&](double&acc){ long ok=0,ck_ok=0,ck_n=0,bt_ok=0,bt_n=0; float p[pv2::NUM_ACT];
        for(size_t k=ntr;k<X.size();k++){ net.infer(X[idx[k]],p);
            int pr=(int)(std::max_element(p,p+pv2::NUM_ACT)-p), y=Y[idx[k]]; if(pr==y)ok++;
            // CHECK vs BET discrimination — самое важное
            if(y==pv2::CHECK){ ck_n++; if(pr==pv2::CHECK)ck_ok++; }
            else if(y>=pv2::R33){ bt_n++; if(pr>=pv2::R33)bt_ok++; } }
        acc=nval?(double)ok/nval:0;
        if(nval){ printf("  check-acc=%.2f(%ld) bet-acc=%.2f(%ld)",
            ck_n?(double)ck_ok/ck_n:0,ck_n, bt_n?(double)bt_ok/bt_n:0,bt_n); } };

    std::vector<std::array<float,pv2::FV2_DIM>> bx; std::vector<int> by; std::vector<float> bw;
    double best=-1; int noimp=0; auto t0=std::chrono::steady_clock::now();
    std::vector<size_t> ord(idx.begin(),idx.begin()+ntr);
    for(int ep=1;ep<=epochs;ep++){
        std::shuffle(ord.begin(),ord.end(),rng);
        double el=0;int nb=0;
        for(size_t k=0;k<ntr;k++){ size_t s=ord[k]; bx.push_back(X[s]); by.push_back(Y[s]); bw.push_back(cw[Y[s]]);
            if((int)bx.size()==batch||k+1==ntr){ el+=net.train_batch(bx,by,bw); nb++; bx.clear();by.clear();bw.clear(); } }
        el/=std::max(1,nb); double acc; eval_val(acc);
        printf("[ep %3d/%d] loss=%.4f val_acc=%.4f",ep,epochs,el,acc);
        if(acc>best+1e-4){best=acc;noimp=0;net.save(out.c_str());printf("  <-best\n");}
        else{noimp++;printf("  (%d/%d)\n",noimp,patience);}
        if(noimp>=patience){printf("[early-stop]\n");break;}
    }
    double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
    net.load(out.c_str()); double acc; eval_val(acc);
    printf("\n=== DONE === best val_acc=%.4f  time=%.1fs  weights=%s\n",acc,sec,out.c_str());
    return 0;
}