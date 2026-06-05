#pragma once
// tools/poker_v2.hpp
// Единый модуль для обучения И арены (один энкодер → нет рассинхрона train/inference).
//   • encode_v2(): богатые признаки, включая взаимодействие руки с бордом (через HandEvaluator),
//     дро, топ-пара/оверпара, pot-odds, SPR, позицию.
//   • Сайзинг: 8 классов действий с размерами ставок.
//   • PolicyNet: 3-слойный MLP (быстрый, под нашим контролем) с Adam, save/load.
//
// Формат v2-строки данных (целые числа):
//   klabel street pot tocall hero_ip c1 c2 nboard [board...] nhist [type amt player]...
//   klabel ∈ 0..7 (см. ниже), tocall — сколько доставить (BB×100), pot — банк (BB×100).

#include "../poker_core/cards.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include "../poker_core/game_state.hpp"
#include <array>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>
#include <cstdio>
#include <string>
#include <sstream>

namespace pv2 {

// ─── Пространство действий с сайзингом ────────────────────────────────────────
enum Act { FOLD=0, CHECK=1, CALL=2, R33=3, R66=4, R100=5, R150=6, ALLIN=7, NUM_ACT=8 };
inline constexpr float RAISE_FRAC[NUM_ACT] = {0,0,0, 0.33f,0.66f,1.0f,1.5f, 0};
inline const char* ANAME[NUM_ACT] = {"fold","check","call","r33","r66","r100","r150","allin"};

inline constexpr int FV2_DIM = 80;

// ─── Энкодер v2 ───────────────────────────────────────────────────────────────
// board: CardSet борда; h1,h2: карманные; street 0..3; pot,tocall в BB×100;
// hero_ip 0/1; agg[4]: частоты fold/check/call/raise в истории (опц., можно нули).
struct EncoderV2 {
    const poker::HandEvaluator* ev;
    explicit EncoderV2(const poker::HandEvaluator* e) : ev(e) {}

    std::array<float,FV2_DIM> encode(const poker::CardSet& board,
                                     poker::Card h1, poker::Card h2,
                                     int street, int pot, int tocall, int hero_ip,
                                     const std::array<float,4>& agg = {}) const
    {
        std::array<float,FV2_DIM> v{};
        // [0..3] улица
        if(street>=0&&street<4) v[street]=1.0f;
        // [4] банк норм
        v[4] = std::min(1.0f, pot/10000.0f);
        // [5] pot-odds = tocall/(pot+tocall)
        v[5] = (pot+tocall>0)? (float)tocall/(float)(pot+tocall) : 0.0f;
        // [6] facing bet
        v[6] = (tocall>0)?1.0f:0.0f;
        // [7] SPR прокси (100bb стек) = stack/pot
        v[7] = std::min(2.0f, (pot>0)? 10000.0f/pot : 2.0f)/2.0f;
        // [8] позиция
        v[8] = (float)hero_ip;
        // [9..21] ранги борда (гистограмма)
        int board_ranks[13]={0}; int suit_cnt[4]={0};
        int nb=0; int maxbr=-1;
        board.for_each([&](poker::Card c){ board_ranks[c.rank()]++; suit_cnt[c.suit()]++; nb++; maxbr=std::max(maxbr,c.rank()); });
        for(int r=0;r<13;r++) v[9+r]= board_ranks[r]?1.0f:0.0f;
        // [22..25] масти борда
        for(int s=0;s<4;s++) v[22+s]= suit_cnt[s]/5.0f;
        // [26] число карт борда
        v[26]= nb/5.0f;

        // ── Карманные карты [27..36] ──
        int r1=h1.rank(), r2=h2.rank(), s1=h1.suit(), s2=h2.suit();
        int hi=std::max(r1,r2), lo=std::min(r1,r2), gap=hi-lo;
        bool pair=(r1==r2), suited=(s1==s2);
        v[27]=r1/12.0f; v[28]=r2/12.0f; v[29]=suited?1:0; v[30]=pair?1:0;
        v[31]=hi/12.0f; v[32]=lo/12.0f; v[33]=gap/12.0f;
        v[34]=(gap<=1)?1:0; v[35]=(lo>=8)?1:0;
        v[36]=std::clamp((hi+ (pair?6.0f:0)+(suited?2.0f:0)-gap*0.5f)/24.0f,0.0f,1.0f);

        // ── Рука × борд [37..58] (когда есть борд) ──
        if(nb>=3 && ev){
            std::vector<poker::Card> cs; cs.reserve(7);
            cs.push_back(h1); cs.push_back(h2);
            board.for_each([&](poker::Card c){cs.push_back(c);});
            poker::HandStrength st = ev->evaluate(cs);
            int cat = (int)poker::category(st);   // 1..9
            // [37..45] one-hot категории (1..9 → idx 0..8)
            if(cat>=1&&cat<=9) v[37+(cat-1)]=1.0f;
            // [46] сила нормирована
            v[46]= st/7462.0f;
            // топ-пара / оверпара (для OnePair)
            bool top_pair=false, overpair=false;
            if(cat==2){ // OnePair
                if(pair && lo>maxbr) overpair=true;
                if(r1==maxbr||r2==maxbr) top_pair=true;
            }
            v[47]=top_pair?1:0; v[48]=overpair?1:0;
            // оверкарты (обе выше борда)
            v[49]=(lo>maxbr)?1:0;
            // ── дро ──
            // флеш-дро: какая-то масть встречается ровно 4 раза среди 2+борд
            int sc2[4]={0}; sc2[s1]++; sc2[s2]++;
            for(int s=0;s<4;s++) sc2[s]+=suit_cnt[s];
            bool fd=false, fmade=false;
            for(int s=0;s<4;s++){ if(sc2[s]==4)fd=true; if(sc2[s]>=5)fmade=true; }
            v[50]=fd?1:0; v[51]=fmade?1:0;
            // стрит-дро: среди уникальных рангов (вкл. колесо) есть окно из 5, где 4 присутствуют
            bool ranks[15]={false};
            ranks[r1]=ranks[r2]=true; for(int r=0;r<13;r++) if(board_ranks[r])ranks[r]=true;
            if(ranks[12]) ranks[ -1 +1 ]=ranks[12]; // ace low помечаем как индекс? упрощённо ниже
            bool sd=false, smade=false;
            for(int start=0;start<=8;start++){
                int c4=0; for(int k=0;k<5;k++) if(ranks[start+k])c4++;
                if(c4>=5)smade=true; if(c4==4)sd=true;
            }
            // колесо A-2-3-4-5
            { int c4=0; if(ranks[12])c4++; for(int k=0;k<4;k++) if(ranks[k])c4++;
              if(c4>=5)smade=true; if(c4==4)sd=true; }
            v[52]=sd?1:0; v[53]=smade?1:0;
            // [54] есть ли пара с бордом (любая моя карта совпала с рангом борда)
            v[54]=(board_ranks[r1]>0||board_ranks[r2]>0)?1:0;
            // [55] борд парный (есть пара на борде)
            { bool bp=false; for(int r=0;r<13;r++) if(board_ranks[r]>=2)bp=true; v[55]=bp?1:0; }
        } else if(ev) {
            // префлоп: грубая сила (нормированный proxy уже в [36])
            v[46]=v[36];
        }

        // ── агрессия истории [70..73] ──
        for(int i=0;i<4;i++) v[70+i]=agg[i];
        return v;
    }
};

// ─── Сайзинг класса → действие движка + целевой размер (raise-to в BB×100) ────
// pot,tocall,street_inv,stack — в BB×100 для героя. Возвращает (ActionType, raise_to).
inline std::pair<poker::ActionType,int> act_from_class(int k, int pot, int tocall,
                                                       int street_inv, int stack, int cur_bet)
{
    auto allin_to=[&]{ return street_inv+stack; };
    switch(k){
        case FOLD:  return {poker::ActionType::Fold,0};
        case CHECK: return {poker::ActionType::Check,0};
        case CALL:  return {poker::ActionType::Call,0};
        case ALLIN: return {poker::ActionType::AllIn, allin_to()};
        default: {
            float frac = RAISE_FRAC[k];
            int add = (int)((pot + tocall) * frac);          // bet/raise размером frac*(банк после колла)
            int raise_to = cur_bet + std::max(add, 100);     // минимум 1bb сверху
            raise_to = std::min(raise_to, allin_to());
            if(raise_to >= allin_to()) return {poker::ActionType::AllIn, allin_to()};
            return {poker::ActionType::Raise, raise_to};
        }
    }
}

// классификация размера рейза в бакет по доле банка
inline int raise_bucket(float frac){
    if(frac<0.45f) return R33;
    if(frac<0.85f) return R66;
    if(frac<1.25f) return R100;
    if(frac<2.2f)  return R150;
    return ALLIN;
}

// ─── PolicyNet ────────────────────────────────────────────────────────────────
// ПО УМОЛЧАНИЮ — MLP (быстрый, лучше работает на ручных фичах: val_acc ~0.75).
// Трансформер unified_ml — ОПЦИОНАЛЬНО, только при -DUSE_TF_PRIOR (эксперимент;
// на seq_len=1 он уступает MLP из-за layernorm над ручными фичами).
#if defined(HAVE_UNIFIED_ML) && defined(USE_TF_PRIOR)
} // close namespace pv2 to include external headers cleanly
#include "models/transformer/transformer_block.hpp"
#include "core/optimizers.hpp"
#include "autograd/tensor.h"
#include <memory>
namespace pv2 {
struct PolicyNet {
    static constexpr int IN=FV2_DIM, OUT=NUM_ACT;
    transformer::TransformerConfig cfg;
    std::unique_ptr<transformer::TransformerEncoder> enc;
    std::unique_ptr<core::Adam> opt;
    double lr=6e-4;

    PolicyNet(){ init(); }
    void init(){
        cfg.embed_dim=IN; cfg.ff_hidden_dim=192; cfg.num_heads=4; cfg.num_layers=2;
        cfg.num_classes=OUT; cfg.max_seq_len=1; cfg.causal=false;
        enc=std::make_unique<transformer::TransformerEncoder>(cfg);
        opt=std::make_unique<core::Adam>(enc->parameters(), lr);
    }
    void infer(const std::array<float,FV2_DIM>& fv, float* probs) const {
        std::vector<double> in(IN); for(int i=0;i<IN;i++)in[i]=fv[i];
        autograd::Tensor x(in,{1,1,(size_t)IN},false);
        autograd::Tensor logits=enc->classify(x);
        const double* lp=logits.data_ptr();
        double mx=lp[0]; for(int i=1;i<OUT;i++)mx=std::max(mx,lp[i]);
        double s=0,e[OUT]; for(int i=0;i<OUT;i++){e[i]=std::exp(lp[i]-mx);s+=e[i];}
        for(int i=0;i<OUT;i++)probs[i]=(float)(e[i]/s);
    }
    float train_batch(const std::vector<std::array<float,FV2_DIM>>&X,
                      const std::vector<int>&Y, const std::vector<float>&W){
        int B=(int)X.size(); if(!B)return 0.0f;
        std::vector<double> in((size_t)B*IN,0.0);
        for(int b=0;b<B;b++)for(int i=0;i<IN;i++)in[(size_t)b*IN+i]=X[b][i];
        autograd::Tensor x(in,{(size_t)B,1,(size_t)IN},false);
        autograd::Tensor logits=enc->classify(x);                 // [B,OUT]
        autograd::Tensor logp=autograd::log(autograd::softmax(logits));
        std::vector<double> oh((size_t)B*OUT,0.0); double wsum=0;
        for(int b=0;b<B;b++){int y=Y[b]; double w=(b<(int)W.size())?W[b]:1.0; oh[(size_t)b*OUT+y]=w; wsum+=w;}
        autograd::Tensor onehot(oh,{(size_t)B,(size_t)OUT},false);
        autograd::Tensor prod=logp*onehot;
        autograd::Tensor s=autograd::sum(prod);
        double norm=wsum>0?wsum:B;
        autograd::Tensor loss=s*(-1.0/norm);
        opt->set_learning_rate(lr);
        opt->zero_grad(); loss.backward(); opt->step();
        return (float)loss.data_ptr()[0];
    }
    bool save(const char* path) const {
        FILE* f=std::fopen(path,"wb"); if(!f)return false;
        const char tag[4]={'P','V','2','T'}; std::fwrite(tag,1,4,f);
        int hdr[5]={cfg.embed_dim,cfg.ff_hidden_dim,cfg.num_heads,cfg.num_layers,cfg.num_classes};
        std::fwrite(hdr,sizeof(int),5,f);
        auto ps=enc->parameters(); size_t np=ps.size(); std::fwrite(&np,sizeof(np),1,f);
        for(auto*p:ps){ size_t n=p->numel(); std::fwrite(&n,sizeof(n),1,f); std::fwrite(p->data_ptr(),sizeof(double),n,f); }
        std::fclose(f); return true;
    }
    bool load(const char* path){
        FILE* f=std::fopen(path,"rb"); if(!f)return false;
        char tag[4]={0}; if(std::fread(tag,1,4,f)!=4||tag[0]!='P'||tag[1]!='V'||tag[2]!='2'||tag[3]!='T'){std::fclose(f);return false;}
        int hdr[5]; size_t g=std::fread(hdr,sizeof(int),5,f); (void)g;
        auto ps=enc->parameters(); size_t np=0; size_t g2=std::fread(&np,sizeof(np),1,f);(void)g2;
        if(np!=ps.size()){std::fclose(f);return false;}
        for(auto*p:ps){ size_t n=0; size_t g3=std::fread(&n,sizeof(n),1,f);(void)g3;
            if(n!=p->numel()){std::fclose(f);return false;} size_t g4=std::fread(p->data_ptr(),sizeof(double),n,f);(void)g4; }
        std::fclose(f); return true;
    }
};
#else
// ─── PolicyNet: 3-слойный MLP (FV2→H1→H2→NUM_ACT) с Adam (fallback) ──────────
struct PolicyNet {
    static constexpr int IN=FV2_DIM, H1=128, H2=64, OUT=NUM_ACT;
    std::vector<double> W1,b1,W2,b2,W3,b3;
    std::vector<double> mW1,vW1,mb1,vb1,mW2,vW2,mb2,vb2,mW3,vW3,mb3,vb3;
    int t=0; double lr=6e-4,beta1=0.9,beta2=0.999,eps=1e-8,wd=1e-5;

    PolicyNet(){ init(); }
    void init(){
        auto mk=[&](std::vector<double>&w,int n,int fan){ w.resize(n);
            std::mt19937_64 rng(987654321ull + (uint64_t)n); std::normal_distribution<double> nd(0,std::sqrt(2.0/fan));
            for(auto&x:w)x=nd(rng); };
        mk(W1,H1*IN,IN); b1.assign(H1,0); mk(W2,H2*H1,H1); b2.assign(H2,0);
        mk(W3,OUT*H2,H2); b3.assign(OUT,0);
        auto z=[&](std::vector<double>&m,size_t n){m.assign(n,0);};
        z(mW1,W1.size());z(vW1,W1.size());z(mb1,H1);z(vb1,H1);
        z(mW2,W2.size());z(vW2,W2.size());z(mb2,H2);z(vb2,H2);
        z(mW3,W3.size());z(vW3,W3.size());z(mb3,OUT);z(vb3,OUT);
    }
    static void relu(std::vector<double>&z){ for(auto&x:z)if(x<0)x=0; }

    void forward(const float* x, double* h1,double* a1,double* h2,double* a2,double* logits) const {
        for(int i=0;i<H1;i++){ double s=b1[i]; const double*w=&W1[i*IN]; for(int j=0;j<IN;j++)s+=w[j]*x[j]; h1[i]=s; a1[i]=s>0?s:0; }
        for(int i=0;i<H2;i++){ double s=b2[i]; const double*w=&W2[i*H1]; for(int j=0;j<H1;j++)s+=w[j]*a1[j]; h2[i]=s; a2[i]=s>0?s:0; }
        for(int i=0;i<OUT;i++){ double s=b3[i]; const double*w=&W3[i*H2]; for(int j=0;j<H2;j++)s+=w[j]*a2[j]; logits[i]=s; }
    }
    void infer(const std::array<float,FV2_DIM>& fv, float* probs) const {
        double h1[H1],a1[H1],h2[H2],a2[H2],lg[OUT];
        float xf[IN]; for(int i=0;i<IN;i++)xf[i]=fv[i];
        forward(xf,h1,a1,h2,a2,lg);
        double mx=lg[0]; for(int i=1;i<OUT;i++)mx=std::max(mx,lg[i]);
        double sum=0,e[OUT]; for(int i=0;i<OUT;i++){e[i]=std::exp(lg[i]-mx);sum+=e[i];}
        for(int i=0;i<OUT;i++)probs[i]=(float)(e[i]/sum);
    }

    static void adam(std::vector<double>&p,std::vector<double>&g,std::vector<double>&m,std::vector<double>&v,
                     int t,double lr,double b1,double b2,double eps,double wd){
        double bc1=1-std::pow(b1,t),bc2=1-std::pow(b2,t);
        for(size_t i=0;i<p.size();i++){ if(wd>0)p[i]*=(1-lr*wd);
            m[i]=b1*m[i]+(1-b1)*g[i]; v[i]=b2*v[i]+(1-b2)*g[i]*g[i];
            p[i]-=lr*(m[i]/bc1)/(std::sqrt(v[i]/bc2)+eps); }
    }

    // батч: средний loss, один Adam-шаг по сумме градиентов
    float train_batch(const std::vector<std::array<float,FV2_DIM>>& X,
                      const std::vector<int>& Y, const std::vector<float>& Wt){
        const int B=(int)X.size(); if(B==0)return 0;
        std::vector<double> gW1(W1.size(),0),gb1(H1,0),gW2(W2.size(),0),gb2(H2,0),gW3(W3.size(),0),gb3(OUT,0);
        double tot=0,wsum=0;
        double h1[H1],a1[H1],h2[H2],a2[H2],lg[OUT];
        for(int b=0;b<B;b++){
            float xf[IN]; for(int i=0;i<IN;i++)xf[i]=X[b][i];
            forward(xf,h1,a1,h2,a2,lg);
            double mx=lg[0]; for(int i=1;i<OUT;i++)mx=std::max(mx,lg[i]);
            double sum=0,p[OUT]; for(int i=0;i<OUT;i++){p[i]=std::exp(lg[i]-mx);sum+=p[i];}
            for(int i=0;i<OUT;i++)p[i]/=sum;
            int y=Y[b]; double w=(b<(int)Wt.size())?Wt[b]:1.0; wsum+=w;
            tot += -w*std::log(std::max(p[y],1e-9));
            double dlg[OUT]; for(int i=0;i<OUT;i++)dlg[i]=w*(p[i]-(i==y?1.0:0.0));
            // grad W3,b3 ; dh2
            double dh2[H2]={0};
            for(int i=0;i<OUT;i++){ gb3[i]+=dlg[i]; double*gw=&gW3[i*H2];
                for(int j=0;j<H2;j++){ gw[j]+=dlg[i]*a2[j]; dh2[j]+=dlg[i]*W3[i*H2+j]; } }
            double dz2[H2]; for(int j=0;j<H2;j++)dz2[j]=dh2[j]*(h2[j]>0?1:0);
            double dh1[H1]={0};
            for(int i=0;i<H2;i++){ gb2[i]+=dz2[i]; double*gw=&gW2[i*H1];
                for(int j=0;j<H1;j++){ gw[j]+=dz2[i]*a1[j]; dh1[j]+=dz2[i]*W2[i*H1+j]; } }
            double dz1[H1]; for(int j=0;j<H1;j++)dz1[j]=dh1[j]*(h1[j]>0?1:0);
            for(int i=0;i<H1;i++){ gb1[i]+=dz1[i]; double*gw=&gW1[i*IN];
                for(int j=0;j<IN;j++) gw[j]+=dz1[i]*xf[j]; }
        }
        double inv=1.0/B; for(auto&g:gW1)g*=inv; for(auto&g:gb1)g*=inv;
        for(auto&g:gW2)g*=inv; for(auto&g:gb2)g*=inv; for(auto&g:gW3)g*=inv; for(auto&g:gb3)g*=inv;
        t++;
        adam(W1,gW1,mW1,vW1,t,lr,beta1,beta2,eps,wd); adam(b1,gb1,mb1,vb1,t,lr,beta1,beta2,eps,0);
        adam(W2,gW2,mW2,vW2,t,lr,beta1,beta2,eps,wd); adam(b2,gb2,mb2,vb2,t,lr,beta1,beta2,eps,0);
        adam(W3,gW3,mW3,vW3,t,lr,beta1,beta2,eps,wd); adam(b3,gb3,mb3,vb3,t,lr,beta1,beta2,eps,0);
        return (float)(tot/std::max(1.0,wsum));
    }

    bool save(const char* path) const {
        FILE* f=std::fopen(path,"wb"); if(!f)return false;
        auto wv=[&](const std::vector<double>&v){ size_t n=v.size(); std::fwrite(&n,sizeof(n),1,f); std::fwrite(v.data(),sizeof(double),n,f); };
        const char tag[4]={'P','V','2','a'}; std::fwrite(tag,1,4,f);
        wv(W1);wv(b1);wv(W2);wv(b2);wv(W3);wv(b3); std::fclose(f); return true;
    }
    bool load(const char* path){
        FILE* f=std::fopen(path,"rb"); if(!f)return false;
        char tag[4]={0}; if(std::fread(tag,1,4,f)!=4||tag[0]!='P'||tag[1]!='V'||tag[2]!='2'){std::fclose(f);return false;}
        auto rv=[&](std::vector<double>&v){ size_t n=0; if(std::fread(&n,sizeof(n),1,f)!=1){v.clear();return;} v.resize(n); if(n){size_t g=std::fread(v.data(),sizeof(double),n,f);(void)g;} };
        rv(W1);rv(b1);rv(W2);rv(b2);rv(W3);rv(b3); std::fclose(f);
        // reinit Adam
        auto z=[&](std::vector<double>&m,size_t n){m.assign(n,0);};
        z(mW1,W1.size());z(vW1,W1.size());z(mb1,b1.size());z(vb1,b1.size());
        z(mW2,W2.size());z(vW2,W2.size());z(mb2,b2.size());z(vb2,b2.size());
        z(mW3,W3.size());z(vW3,W3.size());z(mb3,b3.size());z(vb3,b3.size()); t=0;
        return true;
    }
};
#endif // transformer prior (opt-in)

// ─── Парсер v2-строки → поля ──────────────────────────────────────────────────
struct Row {
    int klabel,street,pot,tocall,hero_ip;
    poker::Card c1,c2; poker::CardSet board;
    std::array<float,4> agg{};
};
inline bool parse_v2(const std::string& line, Row& r){
    std::istringstream is(line);
    int c1,c2,nb;
    if(!(is>>r.klabel>>r.street>>r.pot>>r.tocall>>r.hero_ip>>c1>>c2>>nb))return false;
    r.c1=(c1>=0&&c1<52)?poker::Card{(uint8_t)c1}:poker::Card{};
    r.c2=(c2>=0&&c2<52)?poker::Card{(uint8_t)c2}:poker::Card{};
    r.board=poker::CardSet{};
    for(int i=0;i<nb;i++){int b;if(!(is>>b))return false; if(b>=0&&b<52)r.board.add(poker::Card{(uint8_t)b});}
    int nh=0; int cnt[4]={0}; int tot=0;
    if(is>>nh) for(int i=0;i<nh;i++){int t,a,p; if(!(is>>t>>a>>p))break;
        if(t>=0&&t<4)cnt[t]++; else if(t==4)cnt[3]++; tot++; }
    if(tot>0) for(int k=0;k<4;k++) r.agg[k]=(float)cnt[k]/tot;
    return r.klabel>=0 && r.klabel<NUM_ACT;
}

} // namespace pv2