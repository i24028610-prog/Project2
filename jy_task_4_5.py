# check_task4_dataset.py
import os
import sys
import glob
import csv
import numpy as np
from collections import Counter

# 可选：保存抽样帧图片（不依赖 opencv）
try:
    from PIL import Image
    PIL_OK = True
except Exception:
    PIL_OK = False


def read_actions_csv(path: str):
    actions = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return actions

    # 兼容有无表头
    start = 1 if rows[0] and rows[0][0].strip().lower() in ("action", "actions") else 0
    for r in rows[start:]:
        if not r:
            continue
        actions.append(r[0].strip())
    return actions


def non_empty_ratio(frames: np.ndarray) -> float:
    if frames.size == 0:
        return 0.0
    # 认为 frame 里任何非零格就是非空
    filled = (frames.reshape(frames.shape[0], -1) != 0).any(axis=1)
    return float(filled.mean())


def sample_frame_images(frames: np.ndarray, out_dir: str, tag: str, k: int = 6, cell=16):
    """
    把若干帧保存成图片，便于肉眼验收（0=黑，非0=白）
    """
    if not PIL_OK:
        return

    os.makedirs(out_dir, exist_ok=True)
    n = frames.shape[0]
    if n == 0:
        return

    idxs = []
    if n <= k:
        idxs = list(range(n))
    else:
        # 均匀抽样
        step = max(1, n // k)
        idxs = list(range(0, n, step))[:k]

    for i in idxs:
        grid = frames[i]
        # 二值化显示：非0 -> 255
        img = (grid != 0).astype(np.uint8) * 255
        # 放大像素块
        img = np.repeat(np.repeat(img, cell, axis=0), cell, axis=1)
        im = Image.fromarray(img, mode="L")
        im.save(os.path.join(out_dir, f"{tag}_frame_{i:05d}.png"))


def pair_files(base_dir: str):
    """
    根据时间戳 tag，把 frames_*.npy 与 actions_*.csv 配对
    """
    frames_files = glob.glob(os.path.join(base_dir, "frames_*.npy"))
    actions_files = glob.glob(os.path.join(base_dir, "actions_*.csv"))

    def extract_tag(p, prefix, suffix):
        name = os.path.basename(p)
        if not name.startswith(prefix) or not name.endswith(suffix):
            return None
        return name[len(prefix):-len(suffix)]

    frames_map = {extract_tag(p, "frames_", ".npy"): p for p in frames_files}
    actions_map = {extract_tag(p, "actions_", ".csv"): p for p in actions_files}

    tags = sorted(set(frames_map.keys()) | set(actions_map.keys()))
    pairs = []
    for t in tags:
        fp = frames_map.get(t)
        ap = actions_map.get(t)
        pairs.append((t, fp, ap))
    return pairs


def main():
    base_dir = sys.argv[1] if len(sys.argv) >= 2 else "out_task4_surface"
    if not os.path.isdir(base_dir):
        print(f"[ERROR] directory not found: {base_dir}")
        print("Usage: python check_task4_dataset.py [out_dir]")
        sys.exit(1)

    pairs = pair_files(base_dir)
    if not pairs:
        print(f"[ERROR] no frames_*.npy found in: {base_dir}")
        sys.exit(1)

    print(f"=== Task4 dataset check ===")
    print(f"dir: {os.path.abspath(base_dir)}")
    print(f"found tags: {len(pairs)}")
    print()

    total_frames = 0
    total_actions = 0
    total_non_empty = 0.0
    ok_alignment = 0
    bad_alignment = 0

    all_action_counter = Counter()

    img_dir = os.path.join(base_dir, "_check_images")

    for tag, frames_path, actions_path in pairs:
        print(f"--- tag: {tag} ---")

        if frames_path is None:
            print("  [MISS] frames file missing")
            continue
        if actions_path is None:
            print("  [MISS] actions file missing")
            continue

        # load frames
        try:
            frames = np.load(frames_path)
        except Exception as e:
            print(f"  [ERROR] cannot load frames: {e}")
            continue

        # shape check
        if frames.ndim != 3:
            print(f"  [ERROR] frames ndim != 3 : {frames.shape}")
            continue
        n, h, w = frames.shape
        print(f"  frames: shape={frames.shape} dtype={frames.dtype} range={frames.min()}..{frames.max()}")

        # load actions
        try:
            actions = read_actions_csv(actions_path)
        except Exception as e:
            print(f"  [ERROR] cannot read actions: {e}")
            continue

        m = len(actions)
        print(f"  actions: rows={m}")

        # alignment
        if n == m:
            ok_alignment += 1
            align_str = "OK (actions == frames)"
        elif n == m + 1:
            ok_alignment += 1
            align_str = "OK-ish (frames == actions + 1)"
        else:
            bad_alignment += 1
            align_str = f"BAD (frames={n}, actions={m})"
        print(f"  alignment: {align_str}")

        # non-empty
        ratio = non_empty_ratio(frames)
        print(f"  non-empty ratio: {ratio:.3f}")

        # action distribution
        c = Counter(actions)
        all_action_counter.update(c)
        top5 = c.most_common(5)
        print(f"  top actions: {top5}")

        # accumulate
        total_frames += n
        total_actions += m
        total_non_empty += ratio

        # save sample images
        sample_frame_images(frames, img_dir, tag, k=6, cell=16)

        print()

    # summary
    print("=== Summary ===")
    print(f"datasets checked: {ok_alignment + bad_alignment}")
    print(f"alignment OK: {ok_alignment} | BAD: {bad_alignment}")
    if (ok_alignment + bad_alignment) > 0:
        avg_non_empty = total_non_empty / (ok_alignment + bad_alignment)
    else:
        avg_non_empty = 0.0
    print(f"total frames: {total_frames}")
    print(f"total actions: {total_actions}")
    print(f"avg non-empty ratio: {avg_non_empty:.3f}")
    print(f"overall top actions: {all_action_counter.most_common(10)}")

    if PIL_OK:
        print(f"sample images saved to: {os.path.abspath(img_dir)}")
    else:
        print("PIL not installed, skipped saving sample images.")
        print("If you want images: pip install pillow")

    # exit code: bad alignment => nonzero
    if bad_alignment > 0:
        sys.exit(2)
    sys.exit(0)


if __name__ == "__main__":
    main()
