import numpy as np
import csv
import time

frames = np.load("frames.npz")["frames"]  # (N,18,14)

with open("frames.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    header = ["frame_idx", "timestamp"] + [f"c{i}" for i in range(18*14)]
    w.writerow(header)

    t0 = time.time()
    for i in range(frames.shape[0]):
        flat = frames[i].reshape(-1).tolist()  # 252
        w.writerow([i, t0 + i * 0.01] + flat)  # timestamp 这里用近似递增即可
print("Saved frames.csv rows:", frames.shape[0])
