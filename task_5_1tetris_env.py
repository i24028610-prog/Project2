# tetris_env.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import random

BOARD_H = 18
BOARD_W = 14

# actions
A_NONE = 0
A_LEFT = 1
A_RIGHT = 2
A_ROTATE = 3
A_SOFT_DROP = 4
A_HARD_DROP = 5

ACTION_NAMES = {
    A_NONE: "NONE",
    A_LEFT: "LEFT",
    A_RIGHT: "RIGHT",
    A_ROTATE: "ROTATE",
    A_SOFT_DROP: "SOFT_DROP",
    A_HARD_DROP: "HARD_DROP",
}

# 7 tetromino ids: 0..6
# I,O,T,S,Z,J,L
PIECE_NAMES = ["I", "O", "T", "S", "Z", "J", "L"]

# 4 rotations each, stored as list of (r,c) offsets in a 4x4 local grid
# Anchor/pivot is top-left of the 4x4 bounding box; we store blocks positions in that.
PIECES = {
    0: [  # I
        [(1,0),(1,1),(1,2),(1,3)],
        [(0,2),(1,2),(2,2),(3,2)],
        [(2,0),(2,1),(2,2),(2,3)],
        [(0,1),(1,1),(2,1),(3,1)],
    ],
    1: [  # O
        [(1,1),(1,2),(2,1),(2,2)],
        [(1,1),(1,2),(2,1),(2,2)],
        [(1,1),(1,2),(2,1),(2,2)],
        [(1,1),(1,2),(2,1),(2,2)],
    ],
    2: [  # T
        [(1,1),(1,0),(1,2),(2,1)],
        [(1,1),(0,1),(2,1),(1,2)],
        [(1,1),(1,0),(1,2),(0,1)],
        [(1,1),(0,1),(2,1),(1,0)],
    ],
    3: [  # S
        [(1,1),(1,2),(2,0),(2,1)],
        [(0,1),(1,1),(1,2),(2,2)],
        [(1,1),(1,2),(2,0),(2,1)],
        [(0,1),(1,1),(1,2),(2,2)],
    ],
    4: [  # Z
        [(1,0),(1,1),(2,1),(2,2)],
        [(0,2),(1,1),(1,2),(2,1)],
        [(1,0),(1,1),(2,1),(2,2)],
        [(0,2),(1,1),(1,2),(2,1)],
    ],
    5: [  # J
        [(1,0),(2,0),(2,1),(2,2)],
        [(0,1),(0,2),(1,1),(2,1)],
        [(1,2),(1,1),(1,0),(2,2)],
        [(0,1),(1,1),(2,1),(2,0)],
    ],
    6: [  # L
        [(1,2),(2,0),(2,1),(2,2)],
        [(0,1),(1,1),(2,1),(0,2)],
        [(1,0),(1,1),(1,2),(2,0)],
        [(0,1),(1,1),(2,1),(2,2)],
    ],
}


@dataclass
class PieceState:
    pid: int
    rot: int
    x: int
    y: int


class TetrisEnv:
    """
    - board: (18,14), 0 empty, 1..7 for blocks
    - current piece: PieceState(pid, rot, x, y) where (x,y) is top-left of 4x4 box
    """
    BOARD_H = BOARD_H
    BOARD_W = BOARD_W

    A_NONE = A_NONE
    A_LEFT = A_LEFT
    A_RIGHT = A_RIGHT
    A_ROTATE = A_ROTATE
    A_SOFT_DROP = A_SOFT_DROP
    A_HARD_DROP = A_HARD_DROP

    def __init__(self, seed: int | None = None):
        self.rng = random.Random(seed)
        self.board = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)
        self.score = 0
        self.lines = 0
        self.game_over = False
        self.cur: PieceState | None = None

        self.spawn_new_piece()

    # ---------- public helpers for Task5 ----------
    @property
    def cur_piece_id(self) -> int:
        return self.cur.pid

    @property
    def cur_rot(self) -> int:
        return self.cur.rot

    @property
    def cur_x(self) -> int:
        return self.cur.x

    @property
    def cur_y(self) -> int:
        return self.cur.y

    def get_frame_matrix(self) -> np.ndarray:
        """
        Return the 18x14 matrix representing the CURRENT board with the active piece painted on top.
        Value range: 0..7 (1..7 for piece colors)
        """
        frame = self.board.copy()
        if self.cur is not None and not self.game_over:
            for (dr, dc) in PIECES[self.cur.pid][self.cur.rot]:
                rr = self.cur.y + dr
                cc = self.cur.x + dc
                if 0 <= rr < BOARD_H and 0 <= cc < BOARD_W:
                    frame[rr, cc] = self.cur.pid + 1
        return frame

    # ---------- core env ----------
    def reset(self):
        self.board[:] = 0
        self.score = 0
        self.lines = 0
        self.game_over = False
        self.cur = None
        self.spawn_new_piece()

    def spawn_new_piece(self):
        pid = self.rng.randrange(7)
        rot = 0
        # spawn roughly centered
        x = (BOARD_W // 2) - 2
        y = -2  # allow spawn above top
        self.cur = PieceState(pid=pid, rot=rot, x=x, y=y)
        if self._collides(self.cur):
            self.game_over = True

    def step(self, action: int):
        """
        Applies action once, then gravity (soft drop by 1).
        Returns (reward, done).
        """
        if self.game_over:
            return 0.0, True

        # handle action
        if action == A_LEFT:
            self._try_move(dx=-1, dy=0)
        elif action == A_RIGHT:
            self._try_move(dx=1, dy=0)
        elif action == A_ROTATE:
            self._try_rotate()
        elif action == A_SOFT_DROP:
            self._try_move(dx=0, dy=1)
        elif action == A_HARD_DROP:
            self._hard_drop_and_lock()
            return 0.0, self.game_over
        elif action == A_NONE:
            pass

        # gravity
        moved = self._try_move(dx=0, dy=1)
        if not moved:
            # lock piece
            cleared = self._lock_piece()
            # simple reward: cleared lines
            reward = float(cleared)
            return reward, self.game_over

        return 0.0, self.game_over

    def clone_board(self) -> np.ndarray:
        return self.board.copy()

    def simulate_drop(self, pid: int, rot: int, x: int):
        """
        Returns (new_board, cleared_lines) if valid.
        Returns (None, None) if invalid placement (out of bounds / immediate collision / locks above top).
        """
        rot = int(rot) % 4
        x = int(x)

        test = PieceState(pid=pid, rot=rot, x=x, y=-2)

        # 出生点就碰撞/出界：非法
        if self._collides(test, board=self.board):
            return None, None

        # 下落到不能再下
        while True:
            nxt = PieceState(pid=test.pid, rot=test.rot, x=test.x, y=test.y + 1)
            if self._collides(nxt, board=self.board):
                break
            test = nxt

        # 锁定：任何格子出界 / 顶上方(rr<0) 都算非法（禁止裁剪）
        newb = self.board.copy()
        for (dr, dc) in PIECES[pid][rot]:
            rr = test.y + dr
            cc = test.x + dc
            if cc < 0 or cc >= BOARD_W or rr >= BOARD_H:
                return None, None
            if rr < 0:
                return None, None
            newb[rr, cc] = pid + 1

        newb, cleared = self._clear_lines_board(newb)
        return newb, cleared

    # ---------- internal movement ----------
    def _try_move(self, dx: int, dy: int) -> bool:
        nxt = PieceState(self.cur.pid, self.cur.rot, self.cur.x + dx, self.cur.y + dy)
        if not self._collides(nxt):
            self.cur = nxt
            return True
        return False

    def _try_rotate(self) -> bool:
        nxt = PieceState(self.cur.pid, (self.cur.rot + 1) % 4, self.cur.x, self.cur.y)
        # simple wall-kick: try small shifts
        for kickx in [0, -1, 1, -2, 2]:
            cand = PieceState(nxt.pid, nxt.rot, nxt.x + kickx, nxt.y)
            if not self._collides(cand):
                self.cur = cand
                return True
        return False

    def _hard_drop_and_lock(self):
        while self._try_move(dx=0, dy=1):
            pass
        self._lock_piece()

    def _lock_piece(self) -> int:
        # paint current onto board
        for (dr, dc) in PIECES[self.cur.pid][self.cur.rot]:
            rr = self.cur.y + dr
            cc = self.cur.x + dc
            if rr < 0:
                # locked above top => game over
                self.game_over = True
                return 0
            if 0 <= rr < BOARD_H and 0 <= cc < BOARD_W:
                self.board[rr, cc] = self.cur.pid + 1

        self.board, cleared = self._clear_lines_board(self.board)
        self.lines += cleared
        self.score += cleared * 100

        self.spawn_new_piece()
        return cleared

    def _clear_lines_board(self, board: np.ndarray) -> tuple[np.ndarray, int]:
        full = np.all(board > 0, axis=1)
        cleared = int(np.sum(full))
        if cleared == 0:
            return board, 0
        remain = board[~full]
        newb = np.zeros_like(board)
        newb[-remain.shape[0]:] = remain
        return newb, cleared

    def _collides(self, piece: PieceState, board: np.ndarray | None = None) -> bool:
        b = self.board if board is None else board
        for (dr, dc) in PIECES[piece.pid][piece.rot]:
            rr = piece.y + dr
            cc = piece.x + dc
            if cc < 0 or cc >= BOARD_W:
                return True
            if rr >= BOARD_H:
                return True
            if rr >= 0 and b[rr, cc] > 0:
                return True
        return False
