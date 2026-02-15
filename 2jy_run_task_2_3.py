import numpy as np
frames = np.load("task2_frames.npy")[:80]   # 前80帧
col_counts = (frames != 0).sum(axis=(0,1))  # 对帧和行求和 -> 每列次数
print("nonzero counts per column:", col_counts)

