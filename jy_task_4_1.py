import numpy as np
import csv
import glob
from collections import Counter

# 找最新输出
frames_path = sorted(glob.glob("out_task4_surface/frames_*.npy"))[-1]
actions_path = sorted(glob.glob("out_task4_surface/actions_*.csv"))[-1]

frames = np.load(frames_path)

print("frames:", frames.shape, frames.dtype)
print("value range:", int(frames.min()), "..", int(frames.max()))

# 检查棋盘是否全黑（全0）
non_empty_ratio = float((frames.sum(axis=(1, 2)) > 0).mean())
print("non-empty frame ratio:", non_empty_ratio)

# 读 actions csv
names = []
with open(actions_path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        names.append(row["action_name"])

print("actions rows:", len(names))
cnt = Counter(names)
print("top actions:", cnt.most_common(10))

# 额外：前5帧方块数
print("first 5 frame filled cells:", [int(frames[i].sum()) for i in range(min(5, len(frames)))])
