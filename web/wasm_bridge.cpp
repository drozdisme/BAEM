// web/wasm_bridge.cpp — BAEM v3 WebAssembly bridge.
// Exposes three C functions to JavaScript:
//   engine_new_hand()          — start a new hand
//   engine_action(type,amount) — hero acts (0=fold,1=call,2=raise)
//   engine_get_state()         — returns current state as JSON string
//
// Build (from baem/ root):
//   cd web && bash build_wasm.sh

#include <emscripten/emscripten.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <random>
#include <vector>
#include <string>

// ── poker_core ──
#include "../poker_core/cards.hpp"
#include "../poker_core/hand_evaluator.hpp"
#include "../poker_core/deck_rng.hpp"

// ── MLP PolicyNet from poker_v2.hpp (self-contained math, no I/O) ──
// We define BAEM_WASM so the header skips any training/save/load code
// that depends on fopen – we load from embedded bytes instead.
#define BAEM_WASM 1
// Only pull in the structs and inference we need.
// Manually reproduce the lightweight parts of poker_v2.hpp:
namespace pv2 {
  static constexpr int FV2_DIM = 80;
  static constexpr int NUM_ACT = 8;
  enum Act { FOLD=0, CHECK=1, CALL=2, R33=3, R66=4, R100=5, R150=6, ALLIN=7 };
  static constexpr float RAISE_FRAC[8] = {0,0,0,0.33f,0.66f,1.0f,1.5f,0};

  struct PolicyNet {
    static constexpr int IN=FV2_DIM, H1=128, H2=64, OUT=NUM_ACT;
    std::vector<double> W1,b1,W2,b2,W3,b3;
    PolicyNet(){ init(); }
    void init(){
      auto mk=[&](std::vector<double>&w,int n,int fan){
        w.resize(n); std::mt19937_64 r(987654321ull+(uint64_t)n);
        std::normal_distribution<double> nd(0,std::sqrt(2.0/fan));
        for(auto&x:w)x=nd(r);
      };
      mk(W1,H1*IN,IN); b1.assign(H1,0); mk(W2,H2*H1,H1); b2.assign(H2,0);
      mk(W3,OUT*H2,H2); b3.assign(OUT,0);
    }
    void infer(const float* fv, float* probs) const {
      double h1[H1],a1[H1],h2[H2],a2[H2],lg[OUT];
      for(int i=0;i<H1;i++){ double s=b1[i]; const double*w=&W1[i*IN];
        for(int j=0;j<IN;j++)s+=w[j]*fv[j]; h1[i]=s; a1[i]=s>0?s:0; }
      for(int i=0;i<H2;i++){ double s=b2[i]; const double*w=&W2[i*H1];
        for(int j=0;j<H1;j++)s+=w[j]*a1[j]; h2[i]=s; a2[i]=s>0?s:0; }
      for(int i=0;i<OUT;i++){ double s=b3[i]; const double*w=&W3[i*H2];
        for(int j=0;j<H2;j++)s+=w[j]*a2[j]; lg[i]=s; }
      double mx=lg[0]; for(int i=1;i<OUT;i++)mx=std::max(mx,lg[i]);
      double sm=0,e[OUT]; for(int i=0;i<OUT;i++){e[i]=std::exp(lg[i]-mx);sm+=e[i];}
      for(int i=0;i<OUT;i++)probs[i]=(float)(e[i]/sm);
    }
    // Load from in-memory bytes (format: "PV2a" tag, then size+data per vector)
    bool load_bytes(const uint8_t* data, size_t len){
      if(len<4||data[0]!='P'||data[1]!='V'||data[2]!='2') return false;
      size_t off=4;
      auto rd=[&](std::vector<double>&v)->bool{
        if(off+sizeof(size_t)>len) return false;
        size_t n; memcpy(&n,data+off,sizeof(n)); off+=sizeof(n);
        if(off+n*sizeof(double)>len) return false;
        v.resize(n); memcpy(v.data(),data+off,n*sizeof(double)); off+=n*sizeof(double);
        return true;
      };
      return rd(W1)&&rd(b1)&&rd(W2)&&rd(b2)&&rd(W3)&&rd(b3);
    }
  };
} // namespace pv2

#include "weights_data.h"  // BAEM_WEIGHTS[], BAEM_WEIGHTS_LEN

// ═══════════════════════════════════════════════════════════
//  GAME STATE
// ═══════════════════════════════════════════════════════════
static const int NPLAYERS = 6;
static const int HERO     = 5;  // BTN

struct Player {
  const char* name;
  const char* pos;
  int   chips;
  int   bet;
  bool  folded, allin;
  poker::Card cards[2];
  bool  hasCards;
  bool  isHero;
  int   posRank; // 0=UTG…5=BTN (higher = IP)
};

static Player     P[NPLAYERS];
static poker::Card BOARD[5];
static int  BOARD_SZ  = 0;
static int  POT       = 0;
static int  TO_CALL   = 0;
static int  MIN_RAISE = 0;
static int  ACT_IDX   = 0;
static int  HAND_NUM  = 0;
static char STREET[16]= "idle";

struct ActEntry { char name[16]; char action[12]; int amount; };
static ActEntry ACT_LOG[128];
static int      ACT_LOG_N = 0;

static unsigned ACTED_MASK = 0;  // bit i set when player i has acted this street
static poker::HandEvaluator EVAL;
static poker::Xoshiro256ss  RNG(42u);
static pv2::PolicyNet       NET;
static bool                 NET_LOADED = false;
static bool                 INITED     = false;

// ── card pool ──
static uint8_t DECK[52]; static int DECK_TOP=0;
static void shuffle_deck(){
  for(int i=0;i<52;i++) DECK[i]=(uint8_t)i;
  for(int i=51;i>0;i--){ int j=(int)RNG.next_bounded(i+1); uint8_t t=DECK[i];DECK[i]=DECK[j];DECK[j]=t; }
  DECK_TOP=0;
}
static poker::Card deal_card(){ return poker::Card{DECK[DECK_TOP++]}; }

static const char* RANK_STR[13] = {"2","3","4","5","6","7","8","9","T","J","Q","K","A"};
static const char* SUIT_STR[4]  = {"c","d","h","s"};

// ═══════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════
static void init_players(){
  const char* names[6]   = {"VIKING","GHOST","ORACLE","WRAITH","TITAN","HERO"};
  const char* positions[6]= {"UTG","UTG+1","HJ","CO","SB","BTN"};
  for(int i=0;i<NPLAYERS;i++){
    P[i].name=names[i]; P[i].pos=positions[i];
    P[i].chips=10000; P[i].bet=0;
    P[i].folded=false; P[i].allin=false;
    P[i].hasCards=false; P[i].isHero=(i==HERO);
    P[i].posRank=i;
  }
}

// ═══════════════════════════════════════════════════════════
//  FEATURE ENCODER  (matches EncoderV2 from poker_v2.hpp)
// ═══════════════════════════════════════════════════════════
static float fv[pv2::FV2_DIM];

static void encode_for_player(int pi){
  memset(fv,0,sizeof(fv));
  Player& pl=P[pi];
  poker::CardSet board{}; for(int i=0;i<BOARD_SZ;i++) board.add(BOARD[i]);
  poker::Card h1=pl.cards[0], h2=pl.cards[1];

  // [0..3] street one-hot
  int st=0;
  if(strcmp(STREET,"flop")==0)st=1; else if(strcmp(STREET,"turn")==0)st=2; else if(strcmp(STREET,"river")==0)st=3;
  if(st<4) fv[st]=1.f;
  // [4] pot / 20000
  fv[4]=(float)POT/20000.f;
  // [5] pot-odds
  int tc=std::max(0,TO_CALL-pl.bet);
  fv[5]=(POT+tc>0)?(float)tc/(float)(POT+tc):0.f;
  // [6] facing bet
  fv[6]=(tc>0)?1.f:0.f;
  // [7] stack / 20000
  fv[7]=(float)pl.chips/20000.f;
  // [8] hero ip (position-based: CO,SB,BTN = 1)
  fv[8]=(pl.posRank>=3)?1.f:0.f;

  // [9..74] hand+board features via HandEvaluator
  if(pl.hasCards){
    // made-hand strength (0..1)
    std::vector<poker::Card> held={h1,h2};
    board.for_each([&](poker::Card c){held.push_back(c);});
    int sv=EVAL.evaluate(held);
    fv[9]=(float)sv/32768.f;
    // rank classes [10..21]
    int r1=h1.rank(), r2=h2.rank();
    int hi=std::max(r1,r2), lo=std::min(r1,r2);
    bool suited=(h1.suit()==h2.suit()), pair=(r1==r2);
    if(pair&&hi<13) fv[10+hi]=1.f;
    if(suited&&hi>=0&&hi<13) fv[22+hi]=(float)lo/12.f;
    if(!suited&&!pair) fv[35]=(float)hi/12.f, fv[36]=(float)lo/12.f;
    // board texture [37..48]
    if(BOARD_SZ>=3){
      // flush draw / made flush
      int suits[4]={0,0,0,0};
      for(int i=0;i<BOARD_SZ;i++) suits[BOARD[i].suit()]++;
      int hsuitC=std::max({suits[0],suits[1],suits[2],suits[3]});
      fv[37]=(float)hsuitC/5.f;
      // hero suit count on board
      int mysuit1=0,mysuit2=0;
      for(int i=0;i<BOARD_SZ;i++){
        if(BOARD[i].suit()==h1.suit()) mysuit1++;
        if(BOARD[i].suit()==h2.suit()) mysuit2++;
      }
      fv[38]=(float)std::max(mysuit1,mysuit2)/5.f;
    }
  }
  // Remaining features [49..79] — zero (aggressor history not tracked per-hand)
}

// ═══════════════════════════════════════════════════════════
//  ACTION LOGIC
// ═══════════════════════════════════════════════════════════
static void push_log(const char* name, const char* action, int amount){
  if(ACT_LOG_N>=128) return;
  strncpy(ACT_LOG[ACT_LOG_N].name,   name,   15);
  strncpy(ACT_LOG[ACT_LOG_N].action, action, 11);
  ACT_LOG[ACT_LOG_N].amount=amount;
  ACT_LOG_N++;
}

static void do_fold(int pi){
  P[pi].folded=true;
  push_log(P[pi].name,"fold",0);
  ACTED_MASK |= (1u<<pi);
}
static void do_call(int pi){
  int owe=std::max(0,TO_CALL-P[pi].bet);
  int pay=std::min(owe,P[pi].chips);
  P[pi].chips-=pay; P[pi].bet+=pay; POT+=pay;
  if(P[pi].chips==0) P[pi].allin=true;
  push_log(P[pi].name, owe==0?"check":"call", pay);
  ACTED_MASK |= (1u<<pi);
}
static void do_raise(int pi, int total){
  int maxTotal = P[pi].chips + P[pi].bet;
  total = std::min(total, maxTotal);
  int increment = std::max(0, total - TO_CALL);     // amount raised above the call
  int add = total - P[pi].bet;
  P[pi].chips -= add; POT += add; P[pi].bet = total;
  TO_CALL = total;
  // NLHE min-raise: next raise must be at least this increment above current bet
  MIN_RAISE = total + std::max(increment, 500);
  if(P[pi].chips==0) P[pi].allin=true;
  push_log(P[pi].name,"raise",total);
  ACTED_MASK = (1u<<pi);
}

static void award_pot(int winner){
  snprintf(STREET,sizeof(STREET),"showdown");
  int won=POT;
  P[winner].chips+=POT; POT=0;
  push_log(P[winner].name,"win",won);
}

// advance actIdx to next non-folded non-allin player
static bool advance_actor(){
  for(int step=0;step<NPLAYERS;step++){
    ACT_IDX=(ACT_IDX+1)%NPLAYERS;
    if(!P[ACT_IDX].folded&&!P[ACT_IDX].allin) return true;
  }
  return false;
}

static int count_active(){
  int n=0; for(int i=0;i<NPLAYERS;i++) if(!P[i].folded) n++;
  return n;
}
static bool street_complete(){
  // every active (non-folded, non-allin) player must have acted AND matched the bet
  int maxBet=0;
  for(int i=0;i<NPLAYERS;i++) if(!P[i].folded) maxBet=std::max(maxBet,P[i].bet);
  for(int i=0;i<NPLAYERS;i++){
    if(P[i].folded||P[i].allin) continue;
    if(!(ACTED_MASK & (1u<<i))) return false;  // hasn't acted yet
    if(P[i].bet<maxBet) return false;          // hasn't matched
  }
  return true;
}

static void next_street();
static void bot_act(int pi);

// ── Bot decision via PolicyNet ──
static float randu(){ return (float)RNG.next_bounded(10000)/10000.f; }

// ── Monte-Carlo equity using the REAL HandEvaluator ──
// Returns P(this player's hand wins vs random opponent hands at showdown).
static float mc_equity(int pi, int nsims){
  int nopp = count_active() - 1;
  if(nopp < 1) return 1.0f;
  if(nopp > 5) nopp = 5;

  bool used[52]={false};
  used[P[pi].cards[0].idx]=true;
  used[P[pi].cards[1].idx]=true;
  for(int i=0;i<BOARD_SZ;i++) used[BOARD[i].idx]=true;

  int avail[52], navail=0;
  for(int c=0;c<52;c++) if(!used[c]) avail[navail++]=c;

  int wins=0, ties=0;
  const int needBoard = 5-BOARD_SZ;

  for(int s=0;s<nsims;s++){
    int pool[52]; for(int i=0;i<navail;i++) pool[i]=avail[i];
    int top=navail;
    auto draw=[&](){ int j=(int)RNG.next_bounded(top); int c=pool[j]; pool[j]=pool[--top]; return c; };

    // complete the board
    std::vector<poker::Card> board5;
    for(int i=0;i<BOARD_SZ;i++) board5.push_back(BOARD[i]);
    for(int i=0;i<needBoard;i++) board5.push_back(poker::Card{(uint8_t)draw()});

    // hero strength
    std::vector<poker::Card> hero=board5;
    hero.push_back(P[pi].cards[0]); hero.push_back(P[pi].cards[1]);
    int heroVal=EVAL.evaluate(hero);

    bool heroBest=true, tied=false;
    for(int o=0;o<nopp;o++){
      std::vector<poker::Card> oh=board5;
      oh.push_back(poker::Card{(uint8_t)draw()});
      oh.push_back(poker::Card{(uint8_t)draw()});
      int ov=EVAL.evaluate(oh);
      if(ov>heroVal){ heroBest=false; break; }
      if(ov==heroVal) tied=true;
    }
    if(heroBest){ if(tied) ties++; else wins++; }
  }
  return (wins + 0.5f*ties)/(float)nsims;
}

// ── Bot decision: equity + pot-odds + position, with mixing ──
static void bot_act(int pi){
  if(P[pi].folded||P[pi].allin) return;
  int tc       = std::max(0, TO_CALL - P[pi].bet);
  int maxTotal = P[pi].chips + P[pi].bet;
  int callTo   = P[pi].bet + tc;

  float eq = mc_equity(pi, 160);
  float r  = randu();
  bool  ip = (P[pi].posRank >= 3);   // in position → slightly more aggressive

  // helper: make a pot-relative raise, never auto-shoving unless it must
  auto raiseFrac=[&](float frac){
    int sz = (int)((POT + tc) * frac);
    int to = callTo + sz;
    if(to < MIN_RAISE) to = MIN_RAISE;
    if(to >= maxTotal) to = maxTotal;        // becomes all-in only if pot is huge vs stack
    if(to <= callTo){ do_call(pi); return; }
    do_raise(pi, to);
  };

  if(tc > 0){
    // ── facing a bet ──
    float potOdds = (float)tc / (float)(POT + tc);
    float foldLine = potOdds + (ip?0.02f:0.06f);

    if(eq < foldLine){
      // usually fold; occasionally float/bluff-catch in position
      if(ip && r < 0.12f){ do_call(pi); return; }
      do_fold(pi); return;
    }
    // strong → value raise sometimes
    if(eq > 0.78f && r < 0.45f){ raiseFrac(0.6f + 0.3f*randu()); return; }
    if(eq > 0.62f && r < 0.18f){ raiseFrac(0.5f); return; }
    // otherwise call
    do_call(pi); return;

  } else {
    // ── no bet: check or bet ──
    if(eq > 0.66f){                       // value bet strong hands
      if(r < (ip?0.72f:0.6f)){ raiseFrac(0.55f + 0.25f*randu()); return; }
    } else if(eq < 0.34f && r < (ip?0.16f:0.08f)){  // occasional bluff
      raiseFrac(0.5f); return;
    }
    do_call(pi); return;                  // check
  }
}

// ── Run all bots until hero's turn or end ──
static void run_bots(){
  for(int guard=0;guard<NPLAYERS*4;guard++){
    if(strcmp(STREET,"showdown")==0) return;
    if(count_active()<=1){
      for(int i=0;i<NPLAYERS;i++) if(!P[i].folded){ award_pot(i); return; }
      return;
    }
    if(P[ACT_IDX].isHero) return;   // hero's turn — stop
    if(P[ACT_IDX].folded||P[ACT_IDX].allin){ advance_actor(); continue; }
    bot_act(ACT_IDX);
    if(strcmp(STREET,"showdown")==0) return;
    if(street_complete()){
      if(count_active()<=1){
        for(int i=0;i<NPLAYERS;i++) if(!P[i].folded){ award_pot(i); return; }
        return;
      }
      next_street(); return;
    }
    advance_actor();
  }
}

static void next_street(){
  ACTED_MASK=0;  // reset for new street
  for(int i=0;i<NPLAYERS;i++) P[i].bet=0;
  TO_CALL=0; MIN_RAISE=1000;

  if(strcmp(STREET,"preflop")==0){
    snprintf(STREET,sizeof(STREET),"flop");
    BOARD[0]=deal_card(); BOARD[1]=deal_card(); BOARD[2]=deal_card();
    BOARD_SZ=3;
  } else if(strcmp(STREET,"flop")==0){
    snprintf(STREET,sizeof(STREET),"turn");
    BOARD[3]=deal_card(); BOARD_SZ=4;
  } else if(strcmp(STREET,"turn")==0){
    snprintf(STREET,sizeof(STREET),"river");
    BOARD[4]=deal_card(); BOARD_SZ=5;
  } else if(strcmp(STREET,"river")==0){
    // Showdown: best hand wins
    poker::CardSet bd{};
    for(int i=0;i<BOARD_SZ;i++) bd.add(BOARD[i]);
    int best=-1, winner=-1;
    for(int i=0;i<NPLAYERS;i++){
      if(P[i].folded||!P[i].hasCards) continue;
      std::vector<poker::Card> h={P[i].cards[0],P[i].cards[1]};
      bd.for_each([&](poker::Card c){h.push_back(c);});
      int v=EVAL.evaluate(h);
      if(v>best){ best=v; winner=i; }
    }
    if(winner>=0) award_pot(winner);
    return;
  }
  // if <=1 player can still act (rest all-in), run out the board and showdown
  int canAct=0;
  for(int i=0;i<NPLAYERS;i++) if(!P[i].folded&&!P[i].allin) canAct++;
  if(canAct<=1){
    while(BOARD_SZ<5) BOARD[BOARD_SZ++]=deal_card();
    poker::CardSet bd{}; for(int i=0;i<BOARD_SZ;i++) bd.add(BOARD[i]);
    int best=-1, winner=-1;
    for(int i=0;i<NPLAYERS;i++){
      if(P[i].folded||!P[i].hasCards) continue;
      std::vector<poker::Card> h={P[i].cards[0],P[i].cards[1]};
      bd.for_each([&](poker::Card c){h.push_back(c);});
      int v=EVAL.evaluate(h);
      if(v>best){ best=v; winner=i; }
    }
    if(winner>=0) award_pot(winner);
    return;
  }
  // SB acts first post-flop (seat 4), find first active from SB
  ACT_IDX=4;
  for(int step=0;step<NPLAYERS;step++){
    if(!P[ACT_IDX].folded&&!P[ACT_IDX].allin) break;
    ACT_IDX=(ACT_IDX+1)%NPLAYERS;
  }
  run_bots();
}

// ═══════════════════════════════════════════════════════════
//  JSON SERIALISATION
// ═══════════════════════════════════════════════════════════
static char JSON_BUF[65536];

static int jstr(char* p, const char* s){
  int n=0; p[n++]='"';
  for(;*s;s++) p[n++]=*s;
  p[n++]='"'; return n;
}
static int jint(char* p, int v){
  return sprintf(p,"%d",v);
}
static int jbool(char* p, bool v){
  return sprintf(p,"%s",v?"true":"false");
}

static char* build_json(){
  char* p=JSON_BUF; int n=0;
  auto w=[&](const char* s){ int l=strlen(s); memcpy(p+n,s,l); n+=l; };
  auto wi=[&](int v){ char tmp[16]; int l=sprintf(tmp,"%d",v); memcpy(p+n,tmp,l); n+=l; };
  auto wb=[&](bool v){ w(v?"true":"false"); };
  auto wq=[&](const char* s){ w("\""); w(s); w("\""); };

  w("{");
  w("\"handNum\":"); wi(HAND_NUM); w(",");
  w("\"street\":"); wq(STREET); w(",");
  w("\"pot\":"); wi(POT); w(",");
  w("\"toCall\":"); wi(TO_CALL); w(",");
  w("\"minRaise\":"); wi(MIN_RAISE); w(",");
  w("\"actIdx\":"); wi(ACT_IDX); w(",");
  w("\"heroIdx\":"); wi(HERO); w(",");
  // board
  w("\"board\":[");
  for(int i=0;i<BOARD_SZ;i++){
    if(i) w(",");
    w("{\"r\":"); wq(RANK_STR[BOARD[i].rank()]);
    w(",\"s\":"); wq(SUIT_STR[BOARD[i].suit()]); w("}");
  }
  w("],");
  // players
  w("\"players\":[");
  for(int i=0;i<NPLAYERS;i++){
    if(i) w(",");
    w("{");
    w("\"id\":"); char pid[8]={'p',(char)('1'+i),0,0}; if(i==HERO)memcpy(pid,"hero",5); wq(pid); w(",");
    w("\"name\":"); wq(P[i].name); w(",");
    w("\"pos\":"); wq(P[i].pos); w(",");
    w("\"chips\":"); wi(P[i].chips); w(",");
    w("\"bet\":"); wi(P[i].bet); w(",");
    w("\"folded\":"); wb(P[i].folded); w(",");
    w("\"allin\":"); wb(P[i].allin); w(",");
    w("\"isHero\":"); wb(P[i].isHero); w(",");
    // cards (always send, UI decides visibility)
    w("\"cards\":[");
    if(P[i].hasCards){
      w("{\"r\":"); wq(RANK_STR[P[i].cards[0].rank()]);
      w(",\"s\":"); wq(SUIT_STR[P[i].cards[0].suit()]); w("},");
      w("{\"r\":"); wq(RANK_STR[P[i].cards[1].rank()]);
      w(",\"s\":"); wq(SUIT_STR[P[i].cards[1].suit()]); w("}");
    }
    w("]}");
  }
  w("],");
  // action log
  w("\"actLog\":[");
  for(int i=0;i<ACT_LOG_N;i++){
    if(i) w(",");
    w("{\"name\":"); wq(ACT_LOG[i].name);
    w(",\"action\":"); wq(ACT_LOG[i].action);
    w(",\"amount\":"); wi(ACT_LOG[i].amount); w("}");
  }
  w("]}");
  p[n]=0;
  return JSON_BUF;
}

// ═══════════════════════════════════════════════════════════
//  EXPORTED FUNCTIONS
// ═══════════════════════════════════════════════════════════
extern "C" {

EMSCRIPTEN_KEEPALIVE
void engine_init(){
  // Load weights from embedded bytes
  if(!NET_LOADED){
    NET_LOADED = NET.load_bytes(BAEM_WEIGHTS, BAEM_WEIGHTS_LEN);
  }
  init_players();
  snprintf(STREET,sizeof(STREET),"idle");
}

EMSCRIPTEN_KEEPALIVE
void engine_new_hand(){
  if(!INITED){ engine_init(); INITED=true; }   // one-time only — chips persist between hands
  HAND_NUM++;
  shuffle_deck();
  BOARD_SZ=0; POT=0; ACT_LOG_N=0; ACTED_MASK=0;
  TO_CALL=500; MIN_RAISE=1000;
  snprintf(STREET,sizeof(STREET),"preflop");

  // any player who busted out gets a fresh stack so the table stays full
  for(int i=0;i<NPLAYERS;i++) if(P[i].chips<=0) P[i].chips=10000;

  for(int i=0;i<NPLAYERS;i++){
    P[i].bet=0; P[i].folded=false; P[i].allin=false;
    P[i].cards[0]=deal_card(); P[i].cards[1]=deal_card();
    P[i].hasCards=true;
  }
  // Post blinds: SB=seat4, BB=seat0 (clamped to stack)
  int sb=std::min(250,P[4].chips); P[4].bet=sb; P[4].chips-=sb; POT+=sb;
  int bb=std::min(500,P[0].chips); P[0].bet=bb; P[0].chips-=bb; POT+=bb;
  if(P[4].chips==0)P[4].allin=true;
  if(P[0].chips==0)P[0].allin=true;
  push_log(P[4].name,"sb",sb);
  push_log(P[0].name,"bb",bb);
  ACT_IDX=1;
  run_bots();
}

EMSCRIPTEN_KEEPALIVE
void engine_reset_game(){
  // full reset: everyone back to 10000, start a fresh session
  engine_init();
  INITED=true;
  HAND_NUM=0;
}

EMSCRIPTEN_KEEPALIVE
void engine_action(int type, int amount){
  // 0=fold, 1=call/check, 2=raise
  if(strcmp(STREET,"idle")==0||strcmp(STREET,"showdown")==0) return;
  if(ACT_IDX!=HERO) return;
  switch(type){
    case 0: do_fold(HERO); break;
    case 1: do_call(HERO); break;
    case 2: do_raise(HERO, amount); break;
  }
  if(strcmp(STREET,"showdown")==0) return;
  if(count_active()==1){
    for(int i=0;i<NPLAYERS;i++) if(!P[i].folded){ award_pot(i); return; }
    return;
  }
  if(street_complete()){ next_street(); return; }
  advance_actor();
  run_bots();
}

EMSCRIPTEN_KEEPALIVE
const char* engine_get_state(){
  return build_json();
}

} // extern "C"