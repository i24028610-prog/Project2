# task_3_1.py
# Pierre Dellacherie heuristic agent for Tetris (Task 3)
# - Enumerate all placements (rotation + x), drop to bottom
# - Evaluate resulting board with PD features
# - Output action sequence: ROTATE -> LEFT/RIGHT -> HARD_DROP
#
# This file includes a robust _extract_env_state(env) that tries to adapt
# to different TetrisEnv implementations and prints debug info if missing fields.

import numpy as np

# ---------------------------
# Import env constants if available
# ---------------------------
try:
    from task_2_1 import (
        BOARD_H, BOARD_W,
        A_NONE, A_LEFT, A_RIGHT, A_ROTATE, A_SOFT_DROP, A_HARD_DROP
    )
except Exception:
    # fallback defaults
    BOARD_H, BOARD_W = 18, 14
    A_NONE, A_LEFT, A_RIGHT, A_ROTATE, A_SOFT_DROP, A_HARD_DROP = 0, 1, 2, 3, 4, 5


# ---------------------------
# Tetromino rotation states
# piece id: 1..7 => I,O,T,S,Z,J,L
# Each rotation is list of (x,y) blocks relative to piece local origin (0,0)
# ---------------------------
PIECES = {
    1: [  # I
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    2: [  # O
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (2, 1)],
    ],
    3: [  # T
        [(1, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (0, 1), (1, 1), (1, 2)],
    ],
    4: [  # S
        [(1, 0), (2, 0), (0, 1), (1, 1)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(0, 0), (0, 1), (1, 1), (1, 2)],
    ],
    5: [  # Z
        [(0, 0), (1, 0), (1, 1), (2, 1)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(1, 0), (0, 1), (1, 1), (0, 2)],
    ],
    6: [  # J
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    7: [  # L
        [(2, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
}


# ---------------------------
# Robust state extraction helpers
# ---------------------------
def _get_attr_any(obj, names, default=None):
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return default


def _get_dict_any(d, keys, default=None):
    if not isinstance(d, dict):
        return default
    for k in keys:
        if k in d:
            return d[k]
    return default


LETTER_TO_ID = {"I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7}


def _normalize_piece_id(pid):
    if pid is None:
        return None
    if isinstance(pid, (int, np.integer)):
        return int(pid)
    if isinstance(pid, str):
        s = pid.strip().upper()
        if s in LETTER_TO_ID:
            return LETTER_TO_ID[s]
        # e.g., "TETROMINO_T" / "piece_T"
        for ch in ["I", "O", "T", "S", "Z", "J", "L"]:
            if ch in s:
                return LETTER_TO_ID[ch]
    return None


def _dump_env_debug(env, piece_obj=None):
    print("\n========== [DEBUG] Env attributes ==========")
    try:
        keys = sorted(list(env.__dict__.keys()))
        print("env.__dict__.keys =", keys)
        # also show some values (short)
        for k in keys[:30]:
            v = env.__dict__.get(k, None)
            t = type(v).__name__
            if isinstance(v, (int, float, str, bool, type(None))):
                print(f"  {k}: ({t}) {v}")
            elif isinstance(v, (list, tuple)) and len(v) <= 5:
                print(f"  {k}: ({t}) {v}")
            elif isinstance(v, np.ndarray):
                print(f"  {k}: (ndarray) shape={v.shape} dtype={v.dtype}")
            else:
                print(f"  {k}: ({t}) ...")
    except Exception as e:
        print("cannot read env.__dict__", e)

    if piece_obj is not None:
        print("\n========== [DEBUG] Piece object ==========")
        print("piece_obj type =", type(piece_obj))
        if isinstance(piece_obj, dict):
            print("piece_obj keys =", sorted(list(piece_obj.keys())))
            # show some values
            for k in sorted(list(piece_obj.keys()))[:30]:
                v = piece_obj.get(k, None)
                print(f"  {k}: {v}")
        else:
            try:
                names = [x for x in dir(piece_obj) if not x.startswith("_")]
                print("dir(piece_obj) =", names[:80])
            except Exception as e:
                print("cannot dir(piece_obj)", e)
    print("==========================================\n")


def _extract_env_state(env):
    """
    适配你当前的 TetrisEnv 字段：
    - board: env.locked (18x14)
    - pid: env.cur_id
    - px: env.cur_col
    - py: env.cur_row
    - prot: 由 env.cur_cells 在 PIECES[pid] 里匹配得到(0..3)
    """

    # 1) board
    if hasattr(env, "locked"):
        board = np.array(env.locked, dtype=np.int8)
    else:
        # 兜底：尝试其他名字
        board = _get_attr_any(env, ["board", "grid", "state", "matrix"], None)
        if board is None:
            raise ValueError("找不到棋盘：你的 env 应该有 locked (18x14)。")
        board = np.array(board, dtype=np.int8)

    # 2) piece id & position
    pid = _get_attr_any(env, ["cur_id", "piece_id", "current_id"], None)
    px  = _get_attr_any(env, ["cur_col", "piece_x", "cur_x", "current_x"], None)
    py  = _get_attr_any(env, ["cur_row", "piece_y", "cur_y", "current_y"], None)

    if pid is None or px is None or py is None:
        _dump_env_debug(env, None)
        raise ValueError("找不到 cur_id / cur_col / cur_row，请确认 env.__dict__ 中存在它们。")

    pid = int(pid)
    px, py = int(px), int(py)

    # 3) rotation: match cur_cells to PIECES[pid][rot]
    cur_cells = _get_attr_any(env, ["cur_cells"], None)
    if cur_cells is None:
        _dump_env_debug(env, None)
        raise ValueError("找不到 cur_cells（用于反推旋转）。")

    # 把 cur_cells 规范化成“相对坐标”（减去最小x/y）
    cells = [(int(a), int(b)) for a, b in cur_cells]
    minx = min(a for a, _ in cells)
    miny = min(b for _, b in cells)
    norm = sorted([(a - minx, b - miny) for a, b in cells])

    if pid not in PIECES:
        raise ValueError(f"piece_id={pid} 不在 PIECES 映射中。")

    prot = None
    for r in range(4):
        shape = sorted(PIECES[pid][r])
        # 同样标准化一下 shape（避免不同定义原点导致匹配不上）
        sx_min = min(a for a, _ in shape)
        sy_min = min(b for _, b in shape)
        shape_norm = sorted([(a - sx_min, b - sy_min) for a, b in shape])
        if shape_norm == norm:
            prot = r
            break

    # 如果匹配失败（极少数情况：你 env 的 PIECES 定义原点跟我不同）
    # 先给个默认 rot=0，让程序至少能继续跑，并打印提示
    if prot is None:
        print("[WARN] 无法从 cur_cells 匹配 rot，暂用 rot=0。请把 cur_cells 和 pid 发我，我会修正 PIECES 对齐方式。")
        prot = 0

    return board, pid, px, py, int(prot)



# ---------------------------
# Board simulation helpers
# ---------------------------
def _fits(board, cells, x, y):
    H, W = board.shape
    for cx, cy in cells:
        gx, gy = x + cx, y + cy
        if gx < 0 or gx >= W or gy < 0 or gy >= H:
            return False
        if board[gy, gx] != 0:
            return False
    return True


def _drop_y(board, cells, x, start_y):
    y = start_y
    while _fits(board, cells, x, y + 1):
        y += 1
    return y


def _place_and_clear(board, cells, x, y, pid):
    """
    Return:
      new_board, lines_cleared, piece_cells_in_cleared
    """
    H, W = board.shape
    newb = board.copy()
    placed = []
    for cx, cy in cells:
        gx, gy = x + cx, y + cy
        newb[gy, gx] = pid
        placed.append((gx, gy))

    full_rows = [r for r in range(H) if np.all(newb[r, :] != 0)]
    lines = len(full_rows)

    piece_cells_in_cleared = 0
    if lines > 0:
        full_set = set(full_rows)
        for gx, gy in placed:
            if gy in full_set:
                piece_cells_in_cleared += 1

        kept = [r for r in range(H) if r not in full_rows]
        cleared = np.zeros((lines, W), dtype=newb.dtype)
        newb = np.vstack([cleared, newb[kept, :]])

    return newb, lines, piece_cells_in_cleared


# ---------------------------
# Pierre Dellacherie features
# ---------------------------
def _landing_height(y, cells):
    max_cy = max(cy for _, cy in cells)
    bottom_y = y + max_cy
    return (BOARD_H - 1) - bottom_y


def _row_transitions(board):
    H, W = board.shape
    t = 0
    for r in range(H):
        prev = 1  # left wall filled
        for c in range(W):
            cur = 1 if board[r, c] != 0 else 0
            if cur != prev:
                t += 1
            prev = cur
        if prev == 0:  # right wall filled
            t += 1
    return t


def _col_transitions(board):
    H, W = board.shape
    t = 0
    for c in range(W):
        prev = 1  # top wall filled
        for r in range(H):
            cur = 1 if board[r, c] != 0 else 0
            if cur != prev:
                t += 1
            prev = cur
        if prev == 0:  # bottom wall filled
            t += 1
    return t


def _holes(board):
    H, W = board.shape
    holes = 0
    for c in range(W):
        block_seen = False
        for r in range(H):
            if board[r, c] != 0:
                block_seen = True
            elif block_seen:
                holes += 1
    return holes


def _well_sums(board):
    H, W = board.shape
    wells = 0
    for c in range(W):
        r = 0
        while r < H:
            if board[r, c] == 0:
                left_filled = (c == 0) or (board[r, c - 1] != 0)
                right_filled = (c == W - 1) or (board[r, c + 1] != 0)
                if left_filled and right_filled:
                    depth = 0
                    while r < H and board[r, c] == 0:
                        lf = (c == 0) or (board[r, c - 1] != 0)
                        rf = (c == W - 1) or (board[r, c + 1] != 0)
                        if not (lf and rf):
                            break
                        depth += 1
                        r += 1
                    wells += depth * (depth + 1) // 2
                    continue
            r += 1
    return wells


# ---------------------------
# Heuristic Agent
# ---------------------------
class HeuristicAgent:
    """
    PD heuristic agent.
    Each time a new piece appears, plan a sequence of actions:
      rotate -> move -> hard drop
    """

    def __init__(self, piece_id_map=None, weights=None):
        # env piece_id may not be 1..7; optionally map: {env_id: 1..7}
        self.piece_id_map = piece_id_map or {}

        # widely used PD weights
        default_w = {
            "landing_height": -4.500158825082766,
            "rows_eliminated": 3.4181268101392694,
            "row_trans": -3.2178882868487753,
            "col_trans": -9.348695305445199,
            "holes": -7.899265427351652,
            "wells": -3.3855972247263626,
        }
        self.w = weights or default_w

        self._plan = []
        self._last_piece_sig = None

    def _std_pid(self, pid):
        return self.piece_id_map.get(pid, pid)

    def _evaluate(self, after_board, landing_h, lines, piece_cells_in_cleared):
        # common PD: rows_eliminated = lines * piece_cells_in_cleared
        rows_elim = lines * piece_cells_in_cleared

        score = 0.0
        score += self.w["landing_height"] * landing_h
        score += self.w["rows_eliminated"] * rows_elim
        score += self.w["row_trans"] * _row_transitions(after_board)
        score += self.w["col_trans"] * _col_transitions(after_board)
        score += self.w["holes"] * _holes(after_board)
        score += self.w["wells"] * _well_sums(after_board)
        return score

    def _best_move(self, board, pid, px, py, prot):
        pid = self._std_pid(pid)
        if pid not in PIECES:
            raise ValueError(f"未知 piece_id={pid}，请检查 env 的编号或在 HeuristicAgent(piece_id_map=...) 里映射到 1..7")

        best = None  # (score, target_rot, target_x)
        for r in range(4):
            cells = PIECES[pid][r]

            min_cx = min(cx for cx, _ in cells)
            max_cx = max(cx for cx, _ in cells)
            x_min = -min_cx
            x_max = (BOARD_W - 1) - max_cx

            for tx in range(x_min, x_max + 1):
                start_y = 0
                if not _fits(board, cells, tx, start_y):
                    continue
                ty = _drop_y(board, cells, tx, start_y)

                after, lines, p_in_clear = _place_and_clear(board, cells, tx, ty, pid)
                lh = _landing_height(ty, cells)
                score = self._evaluate(after, lh, lines, p_in_clear)

                if best is None or score > best[0]:
                    best = (score, r, tx)

        if best is None:
            return None
        return best[1], best[2]

    def _build_plan(self, px, prot, target_rot, target_x):
        plan = []
        rot_steps = (target_rot - prot) % 4
        for _ in range(rot_steps):
            plan.append(A_ROTATE)

        dx = target_x - px
        if dx < 0:
            plan += [A_LEFT] * (-dx)
        elif dx > 0:
            plan += [A_RIGHT] * dx

        plan.append(A_HARD_DROP)
        return plan

    def act(self, env):
        board, pid, px, py, prot = _extract_env_state(env)

        # detect new piece (simple but effective):
        # if piece id changed and y is small => likely a new spawn
        occ = int(np.sum(board != 0))
        sig = (pid, px, py, prot, occ)
        is_new_piece = (
            self._last_piece_sig is None or
            (pid != self._last_piece_sig[0] and py <= 2)
        )

        if (not self._plan) or is_new_piece:
            self._last_piece_sig = sig
            best = self._best_move(board, pid, px, py, prot)
            if best is None:
                return A_NONE
            target_rot, target_x = best
            self._plan = self._build_plan(px, prot, target_rot, target_x)

        return self._plan.pop(0) if self._plan else A_NONE
