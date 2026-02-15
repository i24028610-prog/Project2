import numpy as np

# 兼容 frames.npz / frames.npy
def load_frames():
    try:
        data = np.load("frames.npz")
        if isinstance(data, np.lib.npyio.NpzFile):
            return data["frames"]
    except Exception:
        pass
    return np.load("frames.npy")

frames = load_frames()   # (N,18,14)
N, H, W = frames.shape

topmost = np.full(N, -1, dtype=int)     # 每帧：最靠上的非0行（最小row）
bottommost = np.full(N, -1, dtype=int)  # 每帧：最靠下的非0行（最大row）
nonzero_cnt = np.zeros(N, dtype=int)

for i in range(N):
    ys, xs = np.nonzero(frames[i])
    nonzero_cnt[i] = len(ys)
    if len(ys) > 0:
        topmost[i] = int(ys.min())
        bottommost[i] = int(ys.max())

# 只看“有方块”的帧
idx = np.where(nonzero_cnt > 0)[0]
print("frames:", frames.shape)
print("non-empty frames:", len(idx), "/", N)

if len(idx) == 0:
    print("All frames are empty (all zeros). Nothing to analyze.")
    raise SystemExit

# 统计分布
tm = topmost[idx]
bm = bottommost[idx]

print("\nTopmost nonzero row statistics (smaller = closer to row=0):")
print("  min:", tm.min(), "median:", int(np.median(tm)), "max:", tm.max())

print("\nBottommost nonzero row statistics:")
print("  min:", bm.min(), "median:", int(np.median(bm)), "max:", bm.max())

# 判断“更像出生在顶部还是底部”
# 如果出生在顶部：在早期帧里 topmost 会很小（比如 0~3）
# 如果出生在底部：在早期帧里 topmost 会很大（比如 14~17）
K = min(200, len(idx))   # 看最早200个非空帧
early = idx[:K]
early_tm = topmost[early]

print(f"\nEarly {K} non-empty frames: topmost rows summary:")
print("  min:", early_tm.min(), "median:", int(np.median(early_tm)), "max:", early_tm.max())

# 粗判定（你也可以自己看 early median）
if np.median(early_tm) <= 5:
    print("\nConclusion: non-zero cells appear near TOP when pieces spawn (row=0 likely TOP).")
elif np.median(early_tm) >= H - 1 - 5:
    print("\nConclusion: non-zero cells appear near BOTTOM when pieces spawn (row=0 likely BOTTOM).")
else:
    print("\nConclusion: unclear/mixed; check sample frames below.")

# 打印几帧示例：最早的5个非空帧
print("\nSample (first 5 non-empty frames):")
for j in early[:5]:
    print(f"\n--- frame {j} nonzeros={nonzero_cnt[j]} topmost={topmost[j]} bottommost={bottommost[j]} ---")
    # 直接打印18x14（更直观）
    print(frames[j].astype(int))
