import numpy as np

path = r"data\task6_imitation_dataset.npz"   # 如果你训练脚本用的不是这个，就改成训练脚本里的路径
d = np.load(path)
print("NPZ:", path)
print("keys:", d.files)
for k in d.files:
    arr = d[k]
    print(k, arr.shape, arr.dtype)
