# task3_pd_agent.py
import numpy as np

BOARD_H, BOARD_W = 18, 14

# piece id: 1..7 => I,O,T,S,Z,J,L
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


def _find_action(env, names, default=None):
    for n in names:
        if hasattr(env, n):
            return getattr(env, n)
    return default


def _get_actions(env):
    a_none = _find_action(env, ["A_NONE", "NONE"], 0)
    a_left = _find_action(env, ["A_LEFT", "LEFT"], 1)
    a_right = _find_action(env, ["A_RIGHT", "RIGHT"], 2)
    a_rotate = _find_action(env, ["A_ROTATE", "ROTATE"], 3)
    a_hard = _find_action(env, ["A_HARD_DROP", "HARD_DROP"], 5)
    return a_none, a_left, a_right, a_rotate, a_hard


def _extract_env(env):
    locked = np.array(getattr(env, "locked", np.zeros((BOARD_H, BOARD_W), dtype=np.int8)), dtype=np.int8)
    pid = int(getattr(env, "cur_id", 0))
    px = int(getattr(env, "cur_col", 0))
    py = int(getattr(env, "cur_row", 0))
    cur_cells = [(int(a), int(b)) for a, b in getattr(env, "cur_cells", [])]
    # infer current rotation index by matching shape
    prot = 0
    if pid in PIECES and cur_cells:
        minx = min(a for a, _ in cur_cells)
        miny = min(b for _, b in cur_cells)
        norm = sorted([(a - minx, b - miny) for a, b in cur_cells])
        for r in range(4):
            shp = sorted(PIECES[pid][r])
            sx = min(a for a, _ in shp)
            sy = min(b for _, b in shp)
            shp_norm = sorted([(a - sx, b - sy) for a, b in shp])
            if shp_norm == norm:
                prot = r
                break
    return locked, pid, px, py, prot


def _fits(board, cells, x, y):
    H, W = board.shape
    for cx, cy in cells:
        gx, gy = x + cx, y + cy
        if gx < 0 or gx >= W or gy < 0 or gy >= H:
            return False
        if board[gy, gx] != 0:
            return False
    return True


def _drop_y(board, cells, x):
    y = 0
    if not _fits(board, cells, x, y):
        return None
    while _fits(board, cells, x, y + 1):
        y += 1
    return y


def _place_and_clear(board, cells, x, y, pid):
    H, W = board.shape
    b = board.copy()
    placed = []
    for cx, cy in cells:
        gx, gy = x + cx, y + cy
        b[gy, gx] = pid
        placed.append((gx, gy))

    full_rows = [r for r in range(H) if np.all(b[r, :] != 0)]
    lines = len(full_rows)

    piece_cells_in_cleared = 0
    if lines > 0:
        full = set(full_rows)
        for gx, gy in placed:
            if gy in full:
                piece_cells_in_cleared += 1
        keep = [r for r in range(H) if r not in full_rows]
        b = np.vstack([np.zeros((lines, W), dtype=b.dtype), b[keep, :]])

    return b, lines, piece_cells_in_cleared


def _landing_height(y, cells):
    max_cy = max(cy for _, cy in cells)
    bottom_y = y + max_cy
    return (BOARD_H - 1) - bottom_y


def _row_transitions(board):
    H, W = board.shape
    t = 0
    for r in range(H):
        prev = 1
        for c in range(W):
            cur = 1 if board[r, c] != 0 else 0
            if cur != prev:
                t += 1
            prev = cur
        if prev == 0:
            t += 1
    return t


def _col_transitions(board):
    H, W = board.shape
    t = 0
    for c in range(W):
        prev = 1
        for r in range(H):
            cur = 1 if board[r, c] != 0 else 0
            if cur != prev:
                t += 1
            prev = cur
        if prev == 0:
            t += 1
    return t


def _holes(board):
    H, W = board.shape
    holes = 0
    for c in range(W):
        seen = False
        for r in range(H):
            if board[r, c] != 0:
                seen = True
            elif seen:
                holes += 1
    return holes


def _well_sums(board):
    H, W = board.shape
    wells = 0
    for c in range(W):
        r = 0
        while r < H:
            if board[r, c] == 0:
                lf = (c == 0) or (board[r, c - 1] != 0)
                rf = (c == W - 1) or (board[r, c + 1] != 0)
                if lf and rf:
                    d = 0
                    while r < H and board[r, c] == 0:
                        lf2 = (c == 0) or (board[r, c - 1] != 0)
                        rf2 = (c == W - 1) or (board[r, c + 1] != 0)
                        if not (lf2 and rf2):
                            break
                        d += 1
                        r += 1
                    wells += d * (d + 1) // 2
                    continue
            r += 1
    return wells


class PDHeuristicAgent:
    """
    Task3 agent that DOES NOT require env to implement scoring/line-clear correctly.
    It evaluates moves on a shadow board (copy of env.locked), simulates placement+clear,
    and outputs action sequence to env.
    """

    def __init__(self, weights=None, debug_actions=False):
        self.w = weights or {
            "landing_height": -4.500158825082766,
            "rows_eliminated": 3.4181268101392694,
            "row_trans": -3.2178882868487753,
            "col_trans": -9.348695305445199,
            "holes": -7.899265427351652,
            "wells": -3.3855972247263626,
        }
        self.debug_actions = debug_actions
        self._actions = None
        self._plan = []

    def _score(self, after_board, landing_h, lines, piece_cells_in_cleared):
        rows_elim = lines * piece_cells_in_cleared
        return (
            self.w["landing_height"] * landing_h
            + self.w["rows_eliminated"] * rows_elim
            + self.w["row_trans"] * _row_transitions(after_board)
            + self.w["col_trans"] * _col_transitions(after_board)
            + self.w["holes"] * _holes(after_board)
            + self.w["wells"] * _well_sums(after_board)
        )

    def _best_move(self, board, pid):
        best = None  # (score, rot, x)
        for r in range(4):
            cells = PIECES[pid][r]
            min_cx = min(cx for cx, _ in cells)
            max_cx = max(cx for cx, _ in cells)
            x_min = -min_cx
            x_max = (BOARD_W - 1) - max_cx

            for x in range(x_min, x_max + 1):
                y = _drop_y(board, cells, x)
                if y is None:
                    continue
                after, lines, p_in_clear = _place_and_clear(board, cells, x, y, pid)
                lh = _landing_height(y, cells)
                s = self._score(after, lh, lines, p_in_clear)
                if best is None or s > best[0]:
                    best = (s, r, x)
        if best is None:
            return 0, 0
        return best[1], best[2]

    def _build_plan(self, cur_x, cur_rot, target_rot, target_x):
        a_none, a_left, a_right, a_rotate, a_hard = self._actions
        plan = []
        rot_steps = (target_rot - cur_rot) % 4
        plan += [a_rotate] * rot_steps
        dx = target_x - cur_x
        if dx < 0:
            plan += [a_left] * (-dx)
        elif dx > 0:
            plan += [a_right] * dx
        plan.append(a_hard)
        return plan

    def act(self, env):
        if self._actions is None:
            self._actions = _get_actions(env)
            if self.debug_actions:
                print("[agent] actions =", self._actions)

        a_none, *_ = self._actions

        board, pid, px, py, prot = _extract_env(env)
        if pid == 0:
            return a_none

        # If env "pollutes" locked with the falling piece, we clean it in shadow board:
        # remove current piece cells from shadow board (only if they match pid).
        shadow = board.copy()
        for cx, cy in getattr(env, "cur_cells", []):
            gx = int(getattr(env, "cur_col", 0)) + int(cx)
            gy = int(getattr(env, "cur_row", 0)) + int(cy)
            if 0 <= gy < BOARD_H and 0 <= gx < BOARD_W and shadow[gy, gx] == pid:
                shadow[gy, gx] = 0

        if not self._plan:
            tr, tx = self._best_move(shadow, pid)
            self._plan = self._build_plan(px, prot, tr, tx)

        return self._plan.pop(0) if self._plan else a_none
