import os
import csv
import numpy as np

BOARD_W = 14

def read_actions_csv(path: str) -> np.ndarray:
    actions = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头 action
        for row in reader:
            if not row:
                continue
            actions.append(int(row[0]))
    return np.array(actions, dtype=np.int32)

def check_one(tag_dir: str):
    frames_path = os.path.join(tag_dir, "frames.npy")
    actions_path = os.path.join(tag_dir, "actions.csv")
    cur_path = os.path.join(tag_dir, "cur_piece_label.npy")
    tgt_path = os.path.join(tag_dir, "target_label.npy")

    assert os.path.exists(frames_path), f"missing {frames_path}"
    assert os.path.exists(actions_path), f"missing {actions_path}"
    assert os.path.exists(cur_path), f"missing {cur_path}"
    assert os.path.exists(tgt_path), f"missing {tgt_path}"

    frames = np.load(frames_path)
    cur = np.load(cur_path)
    tgt = np.load(tgt_path)
    actions = read_actions_csv(actions_path)

    print(f"dir: {tag_dir}")
    print(f"frames: shape={frames.shape} dtype={frames.dtype} range={frames.min()}..{frames.max()}")
    print(f"actions: rows={actions.shape[0]}")
    print(f"cur_piece_label: shape={cur.shape} range={cur.min()}..{cur.max()}")
    print(f"target_label: shape={tgt.shape} range={tgt.min()}..{tgt.max()}")

    # 1) 对齐检查
    ok_align = (actions.shape[0] == frames.shape[0] == cur.shape[0] == tgt.shape[0])
    print("alignment:", "OK" if ok_align else "BAD")

    # 2) 范围检查
    # cur: 7*4*14 = 392 -> 0..391
    cur_ok = (cur.min() >= 0) and (cur.max() <= 7 * 4 * BOARD_W - 1)
    # tgt: 4*14 = 56 -> 0..55
    tgt_ok = (tgt.min() >= 0) and (tgt.max() <= 4 * BOARD_W - 1)
    print("cur label in [0..391]:", "OK" if cur_ok else "BAD")
    print("tgt label in [0..55] :", "OK" if tgt_ok else "BAD")

    # 3) 分布统计
    cur_unique = len(np.unique(cur))
    tgt_unique = len(np.unique(tgt))
    print(f"cur unique states: {cur_unique} / 392")
    print(f"tgt unique states: {tgt_unique} / 56")

    def topk(arr, k=10):
        vals, cnts = np.unique(arr, return_counts=True)
        idx = np.argsort(cnts)[::-1][:k]
        return list(zip(vals[idx].tolist(), cnts[idx].tolist()))

    print("top cur labels:", topk(cur, 10))
    print("top tgt labels:", topk(tgt, 10))

    # 4) 抽查反解
    sample_idx = np.linspace(0, len(cur) - 1, num=5, dtype=int)
    print("\nsamples:")
    for i in sample_idx:
        c = int(cur[i])
        pid = c // (4 * BOARD_W)
        rem = c % (4 * BOARD_W)
        rot = rem // BOARD_W
        x = rem % BOARD_W

        t = int(tgt[i])
        trot = t // BOARD_W
        tx = t % BOARD_W

        print(f"  idx={i:4d} cur(pid,rot,x)=({pid},{rot},{x})  tgt(x,rot)=({tx},{trot})  action={actions[i]}")

def main():
    base = "out_task5_detect"
    if not os.path.isdir(base):
        print("missing folder:", base)
        return

    tags = [d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
    tags.sort()
    if not tags:
        print("no tags found in out_task5_detect/")
        return

    print("available tags:")
    for t in tags:
        print("  ", t)

    required = ["frames.npy", "actions.csv", "cur_piece_label.npy", "target_label.npy"]

    valid = []
    for t in tags:
        ok = True
        for fn in required:
            if not os.path.exists(os.path.join(base, t, fn)):
                ok = False
                break
        if ok:
            valid.append(t)

    if not valid:
        print("\nNo COMPLETE tag found. (Some folders exist but missing required files.)")
        print("Fix: rerun collect, or delete incomplete tag folders.")
        return

    latest = valid[-1]
    print("\n=== Task5 detect dataset check ===")
    check_one(os.path.join(base, latest))


if __name__ == "__main__":
    main()
