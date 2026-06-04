#!/usr/bin/env python3
"""
prepare_pokerbench.py
Конвертирует датасет PokerBench (RZ412/PokerBench) в компактный целочисленный
формат, который читает C++-тренер (train_pokerbench).

Формат выходной строки (только целые числа, через пробел):
  label street pot curbet hero_ip c1 c2 nboard [board...] nhist [type amt player]...

Зачем так: всё разборщик текста делает здесь (Python, гибкие регэкспы),
а кодирование признаков — в C++ тем же HandHistoryEncoder, что и в рантайме.
Это гарантирует совместимость весов.

Использование:
  # 1) Посмотреть схему датасета (какие колонки/сплиты):
  python3 prepare_pokerbench.py --hf RZ412/PokerBench --dump-schema

  # 2) Проверить парсинг на 5 примерах ПЕРЕД полной конвертацией (важно!):
  python3 prepare_pokerbench.py --hf RZ412/PokerBench --split train --inspect 5

  # 3) Полная конвертация train и test:
  python3 prepare_pokerbench.py --hf RZ412/PokerBench --split train --out prepared_train.txt
  python3 prepare_pokerbench.py --hf RZ412/PokerBench --split test  --out prepared_test.txt

  # Вместо --hf можно дать локальный файл (.jsonl/.csv/.parquet):
  python3 prepare_pokerbench.py --local data.parquet --out prepared_train.txt
"""
import argparse, re, sys, json

# ─── Карты: имя → индекс 0..51 (rank-major: 2c=0..Ac=12, 2d=13.., 2h=26.., 2s=39..) ──
RANKS = {
    "two":0,"three":1,"four":2,"five":3,"six":4,"seven":5,"eight":6,"nine":7,
    "ten":8,"jack":9,"queen":10,"king":11,"ace":12,
    # короткие/числовые формы на всякий случай
    "2":0,"3":1,"4":2,"5":3,"6":4,"7":5,"8":6,"9":7,"10":8,"t":8,"j":9,"q":10,"k":11,"a":12,
}
SUITS = {"club":0,"clubs":0,"c":0,"diamond":1,"diamonds":1,"d":1,
         "heart":2,"hearts":2,"h":2,"spade":3,"spades":3,"s":3}

def card_to_idx(rank_word, suit_word):
    r = RANKS.get(rank_word.strip().lower())
    s = SUITS.get(suit_word.strip().lower())
    if r is None or s is None: return -1
    return s * 13 + r

# "King of Club", "Ten Of Heart", "Three of Spade"
CARD_RE = re.compile(r"\b(two|three|four|five|six|seven|eight|nine|ten|jack|queen|king|ace)\s+of\s+(club|diamond|heart|spade)s?\b", re.I)
# Короткая форма "Kc", "Th", "Ad" — запасной вариант
CARD_SHORT_RE = re.compile(r"\b([2-9TJQKA])([cdhs])\b")

def find_cards(text):
    """Вернёт список индексов карт в порядке появления."""
    out = []
    for m in CARD_RE.finditer(text):
        idx = card_to_idx(m.group(1), m.group(2))
        if idx >= 0: out.append(idx)
    if not out:
        for m in CARD_SHORT_RE.finditer(text):
            idx = card_to_idx(m.group(1), m.group(2))
            if idx >= 0: out.append(idx)
    return out

# ─── Позиция героя → грубая оценка IP (1) / OOP (0) ──────────────────────────
IP_POSITIONS = {"btn","button","co","cutoff","hj","hijack","bvb_ip"}
def hero_ip_from_text(prompt):
    m = re.search(r"your position is\s+([A-Za-z]+)", prompt, re.I)
    if not m: return 1
    pos = m.group(1).lower()
    return 1 if pos in IP_POSITIONS else 0

# ─── Размер пота (в "чипах", где BB=1) → bb100 ───────────────────────────────
def pot_bb100(prompt):
    m = re.search(r"pot size is\s+([0-9.]+)", prompt, re.I)
    if not m: return 200  # дефолт 2bb
    try: return int(round(float(m.group(1)) * 100))
    except ValueError: return 200

# ─── Street из текста борда ──────────────────────────────────────────────────
def street_from_board(nboard, prompt):
    # Число карт борда однозначно задаёт улицу; ключевые слова ненадёжны
    # ("turn" встречается в "your turn to act").
    if nboard >= 5: return 3
    if nboard == 4: return 2
    if nboard == 3: return 1
    return 0

# ─── История действий: ищем глаголы + суммы ──────────────────────────────────
ACTION_WORDS = [
    (re.compile(r"\b(folds?|folded)\b", re.I), 0, False),
    (re.compile(r"\b(checks?|checked)\b", re.I), 1, False),
    (re.compile(r"\b(calls?|called)\b", re.I), 2, False),
    (re.compile(r"\b(bets?|raises?|raised|bet)\b", re.I), 3, True),
    (re.compile(r"\ball[\s-]?in\b", re.I), 4, False),
]
def parse_history(prompt):
    """Очень грубо: разбиваем по запятым/точкам, для каждого фрагмента берём действие.
       player=1, если фрагмент содержит 'you', иначе 0."""
    hist = []
    # Сплит по запятым/точкам с запятой и границам предложений (точка+пробел+заглавная).
    # НЕ рвём десятичные числа вида "2.0 chips".
    frags = re.split(r"[;,]|\.\s+(?=[A-Z])", prompt)
    for fr in frags:
        low = fr.lower()
        if "all" in low and "in" in low and re.search(r"all[\s-]?in", low):
            atype, amt = 4, 0
        else:
            atype = None; amt = 0
            for rx, code, has_amt in ACTION_WORDS:
                if rx.search(fr):
                    atype = code
                    if has_amt:
                        mm = re.search(r"([0-9.]+)\s*chips?", fr)
                        if mm:
                            try: amt = int(round(float(mm.group(1)) * 100))
                            except ValueError: amt = 0
                    break
            if atype is None: continue
        player = 1 if re.search(r"\byou\b", fr, re.I) else 0
        hist.append((atype, amt, player))
    return hist[-8:]  # энкодер использует последние 8

# ─── Метка из ответа ──────────────────────────────────────────────────────────
def parse_label(answer):
    a = answer.strip().lower()
    if re.search(r"all[\s-]?in", a): return 4
    if a.startswith("fold")  or re.match(r"^\s*fold",  a): return 0
    if a.startswith("check") or re.match(r"^\s*check", a): return 1
    if a.startswith("call")  or re.match(r"^\s*call",  a): return 2
    if re.search(r"\b(bet|raise)\b", a): return 3
    return None  # не распознано → пропустить

# ─── Извлечение карманных карт и борда из промпта ────────────────────────────
def extract_hole_and_board(prompt):
    """Карманные карты — в фрагменте 'holding is [.. and ..]'.
       Борд — все прочие карты в промпте."""
    hole = []
    mh = re.search(r"holding is\s*\[?([^\].]*?)\]", prompt, re.I)
    if mh:
        hole = find_cards(mh.group(1))[:2]
    if len(hole) < 2:
        mh2 = re.search(r"h(?:and|olding)[^.]*?is\s*\[([^\]]*)\]", prompt, re.I)
        if mh2: hole = find_cards(mh2.group(1))[:2]
    all_cards = find_cards(prompt)
    hole_set = set(hole)
    board = [c for c in all_cards if c not in hole_set]
    # дедуп с сохранением порядка, максимум 5
    seen=set(); board2=[]
    for c in board:
        if c not in seen: seen.add(c); board2.append(c)
    return hole, board2[:5]

def row_to_line(prompt, answer):
    label = parse_label(answer)
    if label is None: return None, "bad_label"
    hole, board = extract_hole_and_board(prompt)
    if len(hole) < 2: return None, "no_hole"
    nboard = len(board)
    street = street_from_board(nboard, prompt)
    pot = pot_bb100(prompt)
    curbet = 0
    ip = hero_ip_from_text(prompt)
    hist = parse_history(prompt)
    parts = [label, street, pot, curbet, ip, hole[0], hole[1], nboard, *board,
             len(hist)]
    for (t, amt, pl) in hist: parts += [t, amt, pl]
    return " ".join(str(x) for x in parts), "ok"

# ─── Загрузка датасета ────────────────────────────────────────────────────────
def load_rows(args):
    if args.local:
        path = args.local
        if path.endswith(".jsonl"):
            rows = [json.loads(l) for l in open(path) if l.strip()]
        elif path.endswith(".json"):
            obj = json.load(open(path)); rows = obj if isinstance(obj, list) else obj.get("data", [])
        else:
            import pandas as pd
            df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
            rows = df.to_dict("records")
        return rows, None
    from datasets import load_dataset
    ds = load_dataset(args.hf)
    return None, ds

def detect_columns(sample_row):
    keys = list(sample_row.keys())
    prompt_cands = ["instruction","prompt","question","input","text","context"]
    answer_cands = ["output","answer","response","label","action","gold","completion","target"]
    pk = next((k for k in prompt_cands if k in sample_row), None)
    ak = next((k for k in answer_cands if k in sample_row), None)
    if pk is None or ak is None:
        # по длине строковых полей: самое длинное = промпт, короткое = ответ
        strs = [(k, len(str(sample_row[k]))) for k in keys if isinstance(sample_row[k], str)]
        strs.sort(key=lambda x: x[1])
        if len(strs) >= 2:
            ak = ak or strs[0][0]
            pk = pk or strs[-1][0]
    return pk, ak

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf", default="RZ412/PokerBench")
    ap.add_argument("--local", default="")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default="prepared_train.txt")
    ap.add_argument("--prompt-col", default="")
    ap.add_argument("--answer-col", default="")
    ap.add_argument("--dump-schema", action="store_true")
    ap.add_argument("--inspect", type=int, default=0, help="распечатать N разобранных примеров и выйти")
    ap.add_argument("--limit", type=int, default=0, help="ограничить число строк (0=все)")
    args = ap.parse_args()

    rows, ds = load_rows(args)

    # Получаем итератор строк выбранного сплита
    def iter_rows():
        if rows is not None:
            for r in rows: yield r
        else:
            split = args.split if args.split in ds else list(ds.keys())[0]
            for r in ds[split]: yield r

    first = next(iter_rows())
    if args.dump_schema:
        print("splits:", list(ds.keys()) if ds is not None else "(local)")
        print("columns:", list(first.keys()))
        for k, v in first.items():
            sv = str(v); print(f"  {k}: {sv[:200]}{'...' if len(sv)>200 else ''}")
        return

    pk = args.prompt_col or detect_columns(first)[0]
    ak = args.answer_col or detect_columns(first)[1]
    if pk is None or ak is None:
        print("ERROR: не удалось определить колонки prompt/answer. "
              "Запустите --dump-schema и задайте --prompt-col/--answer-col.", file=sys.stderr)
        sys.exit(1)
    print(f"[cols] prompt='{pk}'  answer='{ak}'", file=sys.stderr)

    if args.inspect > 0:
        n = 0
        for r in iter_rows():
            prompt, answer = str(r[pk]), str(r[ak])
            line, status = row_to_line(prompt, answer)
            print("="*70)
            print("PROMPT:", prompt[:400].replace("\n"," "))
            print("ANSWER:", answer[:80])
            print("STATUS:", status, "| LINE:", line)
            n += 1
            if n >= args.inspect: break
        return

    ok = 0; reasons = {}
    with open(args.out, "w") as fo:
        for i, r in enumerate(iter_rows()):
            if args.limit and i >= args.limit: break
            prompt, answer = str(r[pk]), str(r[ak])
            line, status = row_to_line(prompt, answer)
            if line is None:
                reasons[status] = reasons.get(status, 0) + 1
                continue
            fo.write(line + "\n"); ok += 1
    total = ok + sum(reasons.values())
    print(f"[done] wrote {ok}/{total} samples to {args.out}", file=sys.stderr)
    if reasons:
        print("[skipped reasons]", reasons, file=sys.stderr)
    if total and ok / total < 0.5:
        print("WARNING: покрытие <50%. Запустите --inspect 5 и проверьте регэкспы "
              "под точный шаблон вашего сплита PokerBench.", file=sys.stderr)

if __name__ == "__main__":
    main()
