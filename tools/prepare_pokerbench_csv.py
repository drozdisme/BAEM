#!/usr/bin/env python3
"""
prepare_pokerbench_csv.py
ОСНОВНОЙ путь подготовки данных: парсит СТРУКТУРИРОВАННЫЕ CSV PokerBench
(preflop_*_game_scenario_information.csv и postflop_*_game_scenario_information.csv).
Это надёжнее, чем разбор текстовых промптов: точные поля holding/board/decision/pot.

Выход — тот же целочисленный формат, что читает C++-тренер:
  label street pot curbet hero_ip c1 c2 nboard [board...] nhist [type amt player]...

Использование:
  python3 prepare_pokerbench_csv.py --csv preflop_60k_train_set_game_scenario_information.csv  --out prepared_train.txt
  python3 prepare_pokerbench_csv.py --csv postflop_500k_train_set_game_scenario_information.csv --out prepared_train.txt --append
  # тип (preflop/postflop) определяется автоматически по колонкам; можно задать --kind
"""
import argparse, re, sys, ast

RANK = {"2":0,"3":1,"4":2,"5":3,"6":4,"7":5,"8":6,"9":7,"t":8,"j":9,"q":10,"k":11,"a":12}
SUIT = {"c":0,"d":1,"h":2,"s":3}
POS_ALL = {"utg","hj","co","btn","sb","bb","mp","lj","utg+1","utg+2"}
IP_POS  = {"btn","co","hj","button","cutoff","ip"}

def parse_cards(s):
    """'KdKc' / 'Ks7h2d' / '8h8c' → список индексов 0..51."""
    out = []
    if not isinstance(s, str): return out
    for m in re.finditer(r"([2-9tjqkaTJQKA])([cdhs])", s):
        r = RANK.get(m.group(1).lower()); su = SUIT.get(m.group(2).lower())
        if r is not None and su is not None: out.append(su*13 + r)
    return out

def hero_ip(pos):
    return 1 if str(pos).strip().lower() in IP_POS else 0

def to_bb100(x):
    try: return int(round(float(x) * 100))
    except (ValueError, TypeError): return 0

def label_from_decision(dec):
    d = str(dec).strip().lower()
    if not d: return None
    if "all" in d and "in" in d: return 4
    if d.startswith("fold"):  return 0
    if d.startswith("check"): return 1
    if d.startswith("call"):  return 2
    if d.startswith("bet") or d.startswith("raise"): return 3
    return None

def amt_bb100_from(token):
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)", token)
    return to_bb100(m.group(1)) if m else 0

# ─── preflop: prev_line = "UTG/2.0bb/BTN/call/SB/13.0bb/BB/allin/UTG/fold/..." ─
def parse_prev_line(prev_line, hero_pos):
    if not isinstance(prev_line, str): return []
    toks = [t for t in prev_line.split("/") if t != ""]
    hist = []
    i = 0
    cur_pos = None
    hp = str(hero_pos).strip().lower()
    while i < len(toks):
        t = toks[i].strip().lower()
        if t in POS_ALL:
            cur_pos = t; i += 1; continue
        # t — действие для cur_pos
        player = 1 if cur_pos == hp else 0
        if t in ("fold","folds"):       hist.append((0,0,player))
        elif t in ("check","checks"):   hist.append((1,0,player))
        elif t in ("call","calls"):     hist.append((2,0,player))
        elif "allin" in t or "all-in" in t or t=="all": hist.append((4,0,player))
        elif re.search(r"[0-9].*bb|^[0-9.]+$", t):  hist.append((3, amt_bb100_from(t), player))
        i += 1
    return hist[-8:]

# ─── postflop: postflop_action = "OOP_CHECK/IP_BET_5/dealcards/Jc/OOP_RAISE_14" ─
def parse_postflop_action(pf, preflop_action):
    hist = []
    # сначала префлоп-история (как в prev_line, позиции игнорируем для player)
    if isinstance(preflop_action, str):
        for tok in preflop_action.split("/"):
            tl = tok.strip().lower()
            if tl in ("fold","folds"): hist.append((0,0,0))
            elif tl in ("check","checks"): hist.append((1,0,0))
            elif tl in ("call","calls"): hist.append((2,0,0))
            elif re.search(r"[0-9].*bb", tl): hist.append((3, amt_bb100_from(tl), 0))
    if isinstance(pf, str):
        for tok in pf.split("/"):
            tl = tok.strip()
            if not tl or tl.lower()=="dealcards": continue
            if re.fullmatch(r"[2-9tjqkaTJQKA][cdhs]", tl): continue  # карта борда
            player = 1 if tl.upper().startswith("IP") else 0
            body = tl.upper()
            if "FOLD" in body:     hist.append((0,0,player))
            elif "CHECK" in body:  hist.append((1,0,player))
            elif "CALL" in body:   hist.append((2,0,player))
            elif "ALLIN" in body or "ALL_IN" in body: hist.append((4,0,player))
            elif "BET" in body or "RAISE" in body:
                hist.append((3, amt_bb100_from(body), player))
    return hist[-8:]

STREET = {"flop":1,"turn":2,"river":3,"preflop":0}

def build_line_preflop(row):
    hole = parse_cards(row.get("hero_holding") or row.get("holding"))
    if len(hole) < 2: return None, "no_hole"
    label = label_from_decision(row.get("correct_decision"))
    if label is None: return None, "bad_label"
    pot = to_bb100(row.get("pot_size", 0))
    ip  = hero_ip(row.get("hero_pos") or row.get("hero_position"))
    hist = parse_prev_line(row.get("prev_line",""), row.get("hero_pos") or row.get("hero_position"))
    parts = [label, 0, pot, 0, ip, hole[0], hole[1], 0, len(hist)]
    for (t,a,p) in hist: parts += [t,a,p]
    return " ".join(map(str,parts)), "ok"

def build_line_postflop(row):
    hole = parse_cards(row.get("holding") or row.get("hero_holding"))
    if len(hole) < 2: return None, "no_hole"
    label = label_from_decision(row.get("correct_decision"))
    if label is None: return None, "bad_label"
    board = []
    for col in ("board_flop","board_turn","board_river"):
        v = row.get(col)
        if isinstance(v,str) and v.strip(): board += parse_cards(v)
    board = board[:5]
    nboard = len(board)
    ev = str(row.get("evaluation_at","")).strip().lower()
    street = STREET.get(ev, 1 if nboard==3 else 2 if nboard==4 else 3 if nboard>=5 else 0)
    pot = to_bb100(row.get("pot_size",0))
    ip  = hero_ip(row.get("hero_position") or row.get("hero_pos"))
    hist = parse_postflop_action(row.get("postflop_action",""), row.get("preflop_action",""))
    parts = [label, street, pot, 0, ip, hole[0], hole[1], nboard, *board, len(hist)]
    for (t,a,p) in hist: parts += [t,a,p]
    return " ".join(map(str,parts)), "ok"

def detect_kind(cols):
    cl = set(c.lower() for c in cols)
    if "board_flop" in cl or "postflop_action" in cl: return "postflop"
    if "prev_line" in cl or "hero_holding" in cl:     return "preflop"
    return "preflop"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kind", choices=["preflop","postflop","auto"], default="auto")
    ap.add_argument("--append", action="store_true", help="дописать в --out, не перезаписывать")
    ap.add_argument("--inspect", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_csv(args.csv)
    # если в CSV нет заголовка с именами — назначим по позиции (по схемам README)
    cols_lower = [str(c).lower() for c in df.columns]
    if not any(c in cols_lower for c in ("hero_holding","holding","prev_line","board_flop","postflop_action")):
        pre = ["prev_line","hero_pos","hero_holding","correct_decision","num_players","num_bets","available_moves","pot_size"]
        post= ["preflop_action","board_flop","board_turn","board_river","aggressor_position","postflop_action","evaluation_at","available_moves","pot_size","hero_position","holding","correct_decision"]
        names = post if df.shape[1] >= 11 else pre
        df.columns = names[:df.shape[1]]

    kind = detect_kind(df.columns) if args.kind=="auto" else args.kind
    builder = build_line_postflop if kind=="postflop" else build_line_preflop
    print(f"[csv] kind={kind} rows={len(df)} cols={list(df.columns)}", file=sys.stderr)

    if args.inspect:
        n=0
        for _, r in df.iterrows():
            line, st = builder(r.to_dict())
            print("ROW:", {k:r[k] for k in df.columns[:6]})
            print("  ->", st, "|", line)
            n+=1
            if n>=args.inspect: break
        return

    mode = "a" if args.append else "w"
    ok=0; reasons={}
    with open(args.out, mode) as fo:
        for i,(_, r) in enumerate(df.iterrows()):
            if args.limit and i>=args.limit: break
            line, st = builder(r.to_dict())
            if line is None: reasons[st]=reasons.get(st,0)+1; continue
            fo.write(line+"\n"); ok+=1
    total = ok+sum(reasons.values())
    print(f"[done] wrote {ok}/{total} → {args.out} (mode={mode})", file=sys.stderr)
    if reasons: print("[skipped]", reasons, file=sys.stderr)

if __name__ == "__main__":
    main()
