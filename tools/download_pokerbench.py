#!/usr/bin/env python3
"""
download_pokerbench.py — скачивает CSV-файлы датасета PokerBench с HuggingFace
и печатает путь к локальной папке снапшота (последняя строка stdout).

  python3 download_pokerbench.py --repo RZ412/PokerBench
"""
import argparse, sys, glob, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="RZ412/PokerBench")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    kw = dict(repo_id=args.repo, repo_type="dataset",
              allow_patterns=["*game_scenario_information.csv", "*.csv"])
    if args.out_dir:
        kw["local_dir"] = args.out_dir
    path = snapshot_download(**kw)

    csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)
    print("[download] CSV файлов:", len(csvs), file=sys.stderr)
    for c in sorted(csvs):
        print("   ", os.path.relpath(c, path), file=sys.stderr)
    # последняя строка stdout — путь (его читает run_all.sh)
    print(path)

if __name__ == "__main__":
    main()
