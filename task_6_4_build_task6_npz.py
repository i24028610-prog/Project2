import os
from pathlib import Path
import numpy as np

def read_meta(meta_path: Path) -> dict:
    """
    解析 meta.txt（如果存在）。即使解析失败也不影响训练。
    """
    info = {}
    if not meta_path.exists():
        return info
    try:
        txt = meta_path.read_text(encoding="utf-8", errors="ignore").splitlines()
        for line in txt:
            if ":" in line:
                k, v = line.split(":", 1)
                info[k.strip()] = v.strip()
    except Exception:
        pass
    return info

def main():
    # TODO: 改成你的任务5输出根目录（里面有很多tag文件夹）
    ROOT = Path(r"C:\Users\25361\PycharmProjects\pythonProject2\out_task5_detect")
    OUT = Path(r"C:\Users\25361\PycharmProjects\pythonProject2\data\task6_imitation_dataset.npz")
    OUT.parent.mkdir(parents=True, exist_ok=True)

    assert ROOT.exists(), f"ROOT not found: {ROOT}"

    all_boards = []
    all_piece = []
    all_target = []

    tags = [p for p in ROOT.iterdir() if p.is_dir()]
    tags.sort(key=lambda p: p.name)

    kept = 0
    skipped = 0

    for tag_dir in tags:
        frames_p = tag_dir / "frames.npy"
        cur_piece_p = tag_dir / "cur_piece_label.npy"
        target_p = tag_dir / "target_label.npy"

        if not (frames_p.exists() and cur_piece_p.exists() and target_p.exists()):
            skipped += 1
            continue

        frames = np.load(frames_p)              # (T,18,14)
        cur_piece = np.load(cur_piece_p)        # (T,) or (T,k)
        target = np.load(target_p)              # (T,) or (T,k)

        # 基本一致性检查：长度要对齐
        T = frames.shape[0]
        if cur_piece.shape[0] != T or target.shape[0] != T:
            print(f"[SKIP] {tag_dir.name}: length mismatch "
                  f"frames={T}, cur_piece={cur_piece.shape[0]}, target={target.shape[0]}")
            skipped += 1
            continue

        # 强制 frames 变成 int8，防止占用过大
        frames = frames.astype(np.int8)

        all_boards.append(frames)
        all_piece.append(cur_piece)
        all_target.append(target)
        kept += 1

    assert kept > 0, "No valid tag folders found. Check ROOT path."

    boards = np.concatenate(all_boards, axis=0)     # (N,18,14)
    piece = np.concatenate(all_piece, axis=0)       # (N,) or (N,k)
    target = np.concatenate(all_target, axis=0)     # (N,) or (N,k)

    print("=== merged ===")
    print("boards:", boards.shape, boards.dtype, "range", int(boards.min()), int(boards.max()))
    print("piece :", piece.shape, piece.dtype)
    print("target:", target.shape, target.dtype)

    # ---- 关键：把标签整理成我们训练需要的形式 ----
    # 1) 当前方块：期望 piece_ids (N,) ，rots (N,)
    #    你的 cur_piece_label.npy 可能是：
    #    - (N,) 单整数编码
    #    - (N,2) [piece_id, rot]
    #
    # 2) 目标位置：期望 target_xs (N,) target_rots (N,)
    #    你的 target_label.npy 可能是：
    #    - (N,) 单整数编码 placement_id
    #    - (N,2) [target_x, target_rot] 或 [rot, x]
    #
    # 我们做“自适应”解析：先猜测，再给出可调参数。

    # --- 解析 cur_piece_label ---
    if piece.ndim == 1:
        # 先假设：低3位=piece_id(1..7)，高位=rot(0..3) 这种编码（常见做法）
        # 若不符合，你后面我会告诉你怎么根据实际改两行。
        piece_ids = (piece % 8).astype(np.int64)
        rots = ((piece // 8) % 4).astype(np.int64)
    else:
        # 假设第0列是piece_id，第1列是rot
        piece_ids = piece[:, 0].astype(np.int64)
        rots = piece[:, 1].astype(np.int64) if piece.shape[1] > 1 else np.zeros(len(piece), dtype=np.int64)

    # --- 解析 target_label ---
    if target.ndim == 1:
        # 假设：placement_id = rot*14 + x
        placement = target.astype(np.int64)
        target_rots = (placement // 14).astype(np.int64)
        target_xs = (placement % 14).astype(np.int64)
    else:
        # 这里假设 target[:,0]=x, target[:,1]=rot （若顺序相反，后面改一下即可）
        target_xs = target[:, 0].astype(np.int64)
        target_rots = target[:, 1].astype(np.int64) if target.shape[1] > 1 else np.zeros(len(target), dtype=np.int64)

    # 约束范围（避免脏数据）
    target_xs = np.clip(target_xs, 0, 13)
    target_rots = np.clip(target_rots, 0, 3)
    rots = np.clip(rots, 0, 3)
    # piece_id 通常是 1..7；若你的编码是 0..6，也没问题，后续可统一
    # 这里不强行clip到1..7，先保留原值
    # piece_ids = np.clip(piece_ids, 0, 7)

    # 可选：把缺失的 x,y,next_piece 都置0（保持接口统一）
    xs = np.zeros((boards.shape[0],), dtype=np.float32)
    ys = np.zeros((boards.shape[0],), dtype=np.float32)
    next_piece_ids = np.zeros((boards.shape[0],), dtype=np.int64)

    np.savez_compressed(
        OUT,
        boards=boards,
        piece_ids=piece_ids,
        rots=rots,
        xs=xs,
        ys=ys,
        next_piece_ids=next_piece_ids,
        target_xs=target_xs,
        target_rots=target_rots,
    )
    print(f"\nSaved -> {OUT}")

if __name__ == "__main__":
    main()
