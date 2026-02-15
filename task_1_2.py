import numpy as np
import csv
from collections import Counter

data = np.load("frames.npz")
frames = data["frames"]

print("frames shape:", frames.shape)
print("frames dtype:", frames.dtype)
print("value range:", int(frames.min()), int(frames.max()))

# 看有多少帧是全空/全满（质量检查）
empty = int((frames.sum(axis=(1,2)) == 0).sum())
full = int((frames.sum(axis=(1,2)) == frames.shape[1]*frames.shape[2]).sum())
print("empty frames:", empty, "/", frames.shape[0])
print("full frames:", full, "/", frames.shape[0])

with open("actions.csv", "r", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
print("actions rows:", len(rows))
cnt = Counter(r["action"] for r in rows)
print("action counts:", cnt)

none_ratio = cnt.get("NONE", 0) / max(1, len(rows))
print("NONE ratio:", round(none_ratio, 3))
