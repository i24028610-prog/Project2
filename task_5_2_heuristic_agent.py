# heuristic_agent.py
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from task_5_1tetris_env import BOARD_H, BOARD_W, TetrisEnv, A_LEFT, A_RIGHT, A_ROTATE, A_HARD_DROP, A_NONE


@dataclass
class Plan:
    target_x: int
    target_rot: int
    actions: list[int]


class HeuristicAgent:
    """
    Pierre Dellacherie-style heuristic (classic 4 features version):
      - aggregate height
      - complete lines
      - holes
      - bumpiness

    Task5 requires:
      - last_target: (best_x, best_rot) for current piece
    """
    def __init__(self):
        self.last_target = None  # (x, rot)
        self._plan: Plan | None = None
        self._last_piece_signature = None  # (pid, spawn_counter-like) use (pid, x,y,rot) at spawn

    def reset(self):
        self.last_target = None
        self._plan = None
        self._last_piece_signature = None

    def act(self, env: TetrisEnv) -> int:
        """
        Return ONE action per step.
        When a new piece appears, build a plan (sequence of actions) to reach best target, cache last_target.
        """
        sig = (env.cur_piece_id, env.cur_x, env.cur_y, env.cur_rot, env.lines, env.score)
        new_piece = self._is_new_piece(env)

        if new_piece or self._plan is None or len(self._plan.actions) == 0:
            best_x, best_rot = self._compute_best_target(env)
            self.last_target = (best_x, best_rot)
            self._plan = self._build_plan(env, best_x, best_rot)

        # pop next planned action
        if self._plan.actions:
            return self._plan.actions.pop(0)
        return A_NONE

    def _is_new_piece(self, env: TetrisEnv) -> bool:
        # crude but reliable enough: when y is small (spawn area) and signature differs, treat as new
        sig = (env.cur_piece_id, env.cur_rot, env.cur_x, env.cur_y)
        if self._last_piece_signature is None:
            self._last_piece_signature = sig
            return True

        # if the piece just spawned, y usually near -2..0
        is_spawn_zone = env.cur_y <= 0
        changed = sig != self._last_piece_signature
        if is_spawn_zone and changed:
            self._last_piece_signature = sig
            return True
        self._last_piece_signature = sig
        return False

    # ---------- PD search ----------
    def _compute_best_target(self, env: TetrisEnv) -> tuple[int, int]:
        pid = env.cur_piece_id

        best_score = -1e18
        best_x, best_rot = 0, 0

        for rot in range(4):
            # try all x positions; some will be invalid due to collision/walls
            for x in range(BOARD_W):
                # quick reject: if spawn collides immediately at this x/rot, skip
                try_board, cleared = env.simulate_drop(pid, rot, x)
                if try_board is None:
                    continue
                score = self._evaluate_board(try_board, cleared)

                if score > best_score:
                    best_score = score
                    best_x, best_rot = x, rot

        return best_x, best_rot

    def _evaluate_board(self, board: np.ndarray, cleared_lines: int) -> float:
        heights = self._column_heights(board)
        agg_height = float(np.sum(heights))
        holes = float(self._count_holes(board, heights))
        bump = float(np.sum(np.abs(np.diff(heights))))

        # lines is positive (want more), others negative (want less)
        # classic-ish weights (you can tune)
        return (
            + 0.76 * cleared_lines
            - 0.51 * agg_height
            - 0.36 * holes
            - 0.18 * bump
        )

    def _column_heights(self, board: np.ndarray) -> np.ndarray:
        h = np.zeros((BOARD_W,), dtype=np.int32)
        for c in range(BOARD_W):
            col = board[:, c]
            nz = np.where(col > 0)[0]
            h[c] = 0 if len(nz) == 0 else (BOARD_H - int(nz[0]))
        return h

    def _count_holes(self, board: np.ndarray, heights: np.ndarray) -> int:
        holes = 0
        for c in range(BOARD_W):
            if heights[c] == 0:
                continue
            top = BOARD_H - heights[c]
            col = board[:, c]
            # holes: empty cells below top that have a filled above somewhere
            holes += int(np.sum(col[top:] == 0))
        return holes

    # ---------- build action plan ----------
    def _build_plan(self, env: TetrisEnv, target_x: int, target_rot: int) -> Plan:
        actions: list[int] = []

        # rotate
        rot_steps = (target_rot - env.cur_rot) % 4
        for _ in range(rot_steps):
            actions.append(A_ROTATE)

        # move horizontally
        dx = target_x - env.cur_x
        if dx < 0:
            actions.extend([A_LEFT] * (-dx))
        elif dx > 0:
            actions.extend([A_RIGHT] * dx)

        # hard drop
        actions.append(A_HARD_DROP)
        return Plan(target_x=target_x, target_rot=target_rot, actions=actions)
