#!/usr/bin/env python3
"""
prepare_pokerbench_v2.py — CSV PokerBench → v2-формат с САЙЗИНГ-метками (8 классов).

Формат строки: klabel street pot tocall hero_ip c1 c2 nboard [board...] nhist [..]
  klabel: 0 fold,1 check,2 call,3 r33,4 r66,5 r100,6 r150,7 allin

  python3 prepare_pokerbench_v2.py --csv preflop_*_game_scenario_information.csv  --out prepared_v2.txt
  python3 prepare_pokerbench_v2.py --csv postflop_*_game_scenario_information.csv --out prepared_v2.txt --append
"""
import argparse, re, sys

RANK={"2":0,"3":1,"4":2,"5":3,"6":4,"7":5,"8":6,"9":7,"t":8,"j":9,"q":10,"k":11,"a":12}
SUIT={"c":0,"d":1,"h":2,"s":3}
POS_ALL={"utg","hj","co","btn","sb","bb","mp","lj"}
IP_POS={"btn","co","hj","button","cutoff","ip"}
STREET={"flop":1,"turn":2,"river":3,"preflop":0}

# классы сайзинга
FOLD,CHECK,CALL,R33,R66,R100,R150,ALLIN=range(8)
def raise_bucket(frac):
    if frac<0.45: return R33
    if frac<0.85: return R66
    if frac<1.25: return R100
    if frac<2.2:  return R150
    return ALLIN

def cards(s):
    out=[]
    if not isinstance(s,str): return out
    for m in re.finditer(r"([2-9tjqkaTJQKA])([cdhs])",s):
        r=RANK.get(m.group(1).lower()); su=SUIT.get(m.group(2).lower())
        if r is not None and su is not None: out.append(su*13+r)
    return out
def ip(p): return 1 if str(p).strip().lower() in IP_POS else 0
def bb100(x):
    try:return int(round(float(x)*100))
    except: return 0
def num(tok):
    m=re.search(r"([0-9]+(?:\.[0-9]+)?)",str(tok)); return float(m.group(1)) if m else 0.0

def klass_from(dec, pot_bb):
    d=str(dec).strip().lower()
    if not d: return None
    if "all" in d and "in" in d or "shove" in d or "jam" in d: return ALLIN
    if d.startswith("fold"):  return FOLD
    if d.startswith("check"): return CHECK
    if d.startswith("call"):  return CALL
    amt=num(d)  # размер для bet/raise/bare
    if d.startswith("bet") or d.startswith("raise") or amt>0 or "bb" in d:
        if pot_bb<=0: return R66
        frac=(amt*100.0)/pot_bb
        return raise_bucket(frac)
    return None

def tocall_preflop(prev_line, hero_pos):
    # последняя ставка - то что герой уже вложил (SB=50,BB=100,иначе 0)
    if not isinstance(prev_line,str): return 0
    toks=[t for t in prev_line.split("/") if t]
    last=0.0
    for t in toks:
        if re.search(r"[0-9].*bb|^[0-9.]+$",t.lower()): last=num(t)
        if "allin" in t.lower(): last=100.0  # грубо: олл-ин = большой
    hp=str(hero_pos).strip().lower()
    posted = 50 if hp=="sb" else (100 if hp=="bb" else 0)
    return max(0, int(round(last*100))-posted)

def tocall_postflop(pf, avail):
    a=str(avail).lower()
    if "check" in a: return 0
    if not isinstance(pf,str): return 0
    last=0.0
    for tok in pf.split("/"):
        b=tok.upper()
        if "BET" in b or "RAISE" in b: last=num(b)
    return int(round(last*100))

def build_preflop(r):
    hole=cards(r.get("hero_holding") or r.get("holding"))
    if len(hole)<2: return None,"no_hole"
    pot=bb100(r.get("pot_size",0))
    k=klass_from(r.get("correct_decision"), pot)
    if k is None: return None,"bad_label"
    tc=tocall_preflop(r.get("prev_line",""), r.get("hero_pos") or r.get("hero_position"))
    parts=[k,0,pot,tc,ip(r.get("hero_pos") or r.get("hero_position")),hole[0],hole[1],0,0]
    return " ".join(map(str,parts)),"ok"

def build_postflop(r):
    hole=cards(r.get("holding") or r.get("hero_holding"))
    if len(hole)<2: return None,"no_hole"
    pot=bb100(r.get("pot_size",0))
    k=klass_from(r.get("correct_decision"), pot)
    if k is None: return None,"bad_label"
    board=[]
    for c in ("board_flop","board_turn","board_river"):
        v=r.get(c)
        if isinstance(v,str) and v.strip(): board+=cards(v)
    board=board[:5]; nb=len(board)
    ev=str(r.get("evaluation_at","")).strip().lower()
    street=STREET.get(ev, 1 if nb==3 else 2 if nb==4 else 3 if nb>=5 else 0)
    tc=tocall_postflop(r.get("postflop_action",""), r.get("available_moves",""))
    parts=[k,street,pot,tc,ip(r.get("hero_position") or r.get("hero_pos")),hole[0],hole[1],nb,*board,0]
    return " ".join(map(str,parts)),"ok"

def detect(cols):
    cl=set(c.lower() for c in cols)
    return "postflop" if ("board_flop" in cl or "postflop_action" in cl) else "preflop"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--csv",required=True); ap.add_argument("--out",required=True)
    ap.add_argument("--append",action="store_true"); ap.add_argument("--inspect",type=int,default=0)
    ap.add_argument("--limit",type=int,default=0)
    a=ap.parse_args()
    import pandas as pd
    df=pd.read_csv(a.csv)
    kind=detect(df.columns); build=build_postflop if kind=="postflop" else build_preflop
    print(f"[v2] kind={kind} rows={len(df)}",file=sys.stderr)
    if a.inspect:
        n=0
        for _,r in df.iterrows():
            line,st=build(r.to_dict()); print(st,"|",line); n+=1
            if n>=a.inspect:break
        return
    ok=0;reasons={}
    with open(a.out,"a" if a.append else "w") as fo:
        for i,(_,r) in enumerate(df.iterrows()):
            if a.limit and i>=a.limit:break
            line,st=build(r.to_dict())
            if line is None:reasons[st]=reasons.get(st,0)+1;continue
            fo.write(line+"\n");ok+=1
    print(f"[v2] wrote {ok}/{ok+sum(reasons.values())} → {a.out}",file=sys.stderr)
    if reasons:print("[skipped]",reasons,file=sys.stderr)

if __name__=="__main__": main()
