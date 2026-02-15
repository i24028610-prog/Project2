# task_2_1.py
import random
import numpy as np

# Board size
BOARD_H, BOARD_W = 18, 14

# Actions (must match Task4 import)
A_NONE = 0
A_LEFT = 1
A_RIGHT = 2
A_ROTATE = 3
A_SOFT_DROP = 4
A_HARD_DROP = 5

# 7 tetrominoes in 4x4 (rotation generated)
# Represent each piece as a list of (r,c) cells in a 4x4 grid for rot=0
PIECES = {
    1: [(1, 0), (1, 1), (1, 2), (1, 3)],              # I
    2: [(1, 1), (1, 2), (2, 1), (2, 2)],              # O
    3: [(1, 1), (2, 0), (2, 1), (2, 2)],              # T
    4: [(1, 1), (1, 2), (2, 0), (2, 1)],              # S
    5: [(1, 0), (1, 1), (2, 1), (2, 2)],              # Z
    6: [(1, 0), (2, 0), (2, 1), (2, 2)],              # J
    7: [(1, 2), (2, 0), (2, 1), (2, 2)],              # L
}

def _rot90(cells):
    # rotate inside 4x4: (r,c) -> (c, 3-r)
    return [(c, 3 - r) for (r, c) in cells]

def _normalize(cells):
    # shift to keep within 4x4-ish, but we actually place by x/y so keep as-is.
    return cells

# Precompute rotations 0..3 for each piece id
PIECE_ROTS = {}
for pid, base in PIECES.items():
    rots = []
    cur = base
    for _ in range(4):
        rots.append(_normalize(cur))
        cur = _rot90(cur)
    PIECE_ROTS[pid] = rots


class TetrisEnv:
    """
    Minimal, stable env for Task2/Task4.

    Required attributes used by Task4:
      - locked : np.ndarray shape (18,14) with 0/1..7
      - cur_cells : list[(r,c)] for current falling piece
      - cur_id : int 1..7
      - step(action) -> obs, reward, done, info
      - reset() -> obs
    """
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.locked = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)

        self.cur_id = None
        self.cur_rot = 0
        self.cur_x = 0
        self.cur_y = 0
        self.cur_cells = None

        self.done = False
        self.score = 0
        self.lines = 0

        # gravity control (optional)
        self._tick = 0
        self.gravity_every = None  # every N steps auto-drop by 1 (set None to disable)

    def reset(self):
        self.locked[:] = 0
        self.done = False
        self.score = 0
        self.lines = 0
        self._tick = 0
        self._spawn_new()
        return self._obs()

    def _obs(self):
        # For Task2, returning locked board is enough
        return self.locked.copy()

    def _spawn_new(self):
        self.cur_id = self.rng.randint(1, 7)
        self.cur_rot = 0
        # spawn near center
        self.cur_x = BOARD_W // 2 - 2
        self.cur_y = 0
        self._update_cur_cells()

        if self._collides(self.cur_x, self.cur_y, self.cur_rot):
            self.done = True

    def _cells_world(self, x, y, rot):
        cells = PIECE_ROTS[self.cur_id][rot]
        return [(y + r, x + c) for (r, c) in cells]

    def _update_cur_cells(self):
        if self.cur_id is None:
            self.cur_cells = None
            return
        self.cur_cells = self._cells_world(self.cur_x, self.cur_y, self.cur_rot)

    def _collides(self, x, y, rot):
        for (r, c) in self._cells_world(x, y, rot):
            if c < 0 or c >= BOARD_W or r < 0 or r >= BOARD_H:
                return True
            if self.locked[r, c] != 0:
                return True
        return False

    def _try_move(self, dx, dy):
        nx, ny = self.cur_x + dx, self.cur_y + dy
        if not self._collides(nx, ny, self.cur_rot):
            self.cur_x, self.cur_y = nx, ny
            self._update_cur_cells()
            return True
        return False

    def _try_rotate(self):
        nr = (self.cur_rot + 1) % 4
        # basic wall-kick attempts
        kicks = [(0, 0), (-1, 0), (1, 0), (-2, 0), (2, 0)]
        for kx, ky in kicks:
            nx, ny = self.cur_x + kx, self.cur_y + ky
            if not self._collides(nx, ny, nr):
                self.cur_x, self.cur_y, self.cur_rot = nx, ny, nr
                self._update_cur_cells()
                return True
        return False

    def _lock_piece(self):
        for (r, c) in self.cur_cells:
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                self.locked[r, c] = int(self.cur_id)

        cleared = self._clear_lines()
        self.lines += cleared
        self.score += cleared * 100

        self._spawn_new()
        return cleared

    def _clear_lines(self):
        full = np.where(np.all(self.locked != 0, axis=1))[0]
        if len(full) == 0:
            return 0
        new_board = np.zeros_like(self.locked)
        keep_rows = [r for r in range(BOARD_H) if r not in set(full)]
        # shift down
        dst = BOARD_H - 1
        for r in reversed(keep_rows):
            new_board[dst, :] = self.locked[r, :]
            dst -= 1
        self.locked = new_board
        return int(len(full))

    def step(self, action):
        if self.done:
            return self._obs(), 0.0, True, {}

        # normalize action if someone passes string
        if isinstance(action, str):
            action = action.upper()
            action = {
                "NONE": A_NONE,
                "LEFT": A_LEFT,
                "RIGHT": A_RIGHT,
                "ROTATE": A_ROTATE,
                "SOFT_DROP": A_SOFT_DROP,
                "HARD_DROP": A_HARD_DROP,
            }.get(action, A_NONE)

        reward = 0.0

        # --- handle action ---
        if action == A_LEFT:
            self._try_move(-1, 0)
        elif action == A_RIGHT:
            self._try_move(1, 0)
        elif action == A_ROTATE:
            self._try_rotate()
        elif action == A_SOFT_DROP:
            moved = self._try_move(0, 1)
            if not moved:
                reward += float(self._lock_piece())
        elif action == A_HARD_DROP:
            # drop until collision
            while self._try_move(0, 1):
                pass
            reward += float(self._lock_piece())
        else:
            # A_NONE
            pass

        # --- gravity (optional) ---
        self._tick += 1
        if self.gravity_every is not None and self._tick % self.gravity_every == 0:
            # if already locked by soft/hard drop, cur_cells may be new piece; still OK
            if not self._try_move(0, 1):
                # if can't fall, lock
                reward += float(self._lock_piece())

        done = bool(self.done)
        return self._obs(), reward, done, {"score": self.score, "lines": self.lines}
