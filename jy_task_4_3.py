import os, glob, csv
from collections import defaultdict, Counter

def newest(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

def load_actions(csv_path):
    acts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            acts.append(row["raw_action"])
    return acts

def try_int(x):
    try:
        return int(x)
    except:
        return None

def decode_from_log(csv_path):
    acts_raw = load_actions(csv_path)
    ids = [try_int(x) for x in acts_raw]
    ids = [x for x in ids if x is not None]

    print("Loaded actions:", len(ids))
    print("Unique IDs:", sorted(set(ids)))
    print("Counts:", dict(Counter(ids)))
    print()
    print("下一步：用 Task4 的 env 对比法推断每个ID含义（见下面第2步脚本）")

if __name__ == "__main__":
    # 自动找最新的 actions_*.csv
    p = newest(r"out_task4_surface\actions_*.csv")
    if p is None:
        print("找不到 out_task4_surface/actions_*.csv")
    else:
        print("Using:", p)
        decode_from_log(p)
