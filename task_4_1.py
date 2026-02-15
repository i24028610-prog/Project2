# task_4_1.py
import os
import csv
import time
import copy
import pygame
import numpy as np

from task_2_1_new import (
    TetrisEnv,
    BOARD_H, BOARD_W,
    A_NONE, A_LEFT, A_RIGHT, A_ROTATE, A_SOFT_DROP, A_HARD_DROP
)

# ================= 可调参数 =================
FPS = 20
OUT_DIR = "out_task4_surface"

# 没有计划时的动作（如果你的 env 自带重力，这里用 A_NONE；否则用 A_SOFT_DROP）
IDLE_ACTION = A_NONE

# 动作慢放：每个动作持续多少帧
ACTION_HOLD_FRAMES = 1

# plan 队列最长保护
MAX_PLAN_STEPS = 120

# 不贴边
EDGE_MARGIN = 1

# ★关键：HARD_DROP 后“结算帧”数量（用 SOFT_DROP 更稳）
SETTLE_FRAMES_AFTER_HARDDROP = 8
SETTLE_ACTION = A_SOFT_DROP

CELL = 30
MARGIN = 20
WIDTH = MARGIN * 2 + BOARD_W * CELL
HEIGHT = MARGIN * 2 + BOARD_H * CELL

BG = (20, 20, 20)
GRID = (45, 45, 45)
BLOCK = (160, 0, 240)
BORDER = (220, 200, 40)

NAME2ACTION = {
    "NONE": A_NONE,
    "LEFT": A_LEFT,
    "RIGHT": A_RIGHT,
    "ROTATE": A_ROTATE,
    "SOFT_DROP": A_SOFT_DROP,
    "HARD_DROP": A_HARD_DROP,
}
ACTION2NAME = {v: k for k, v in NAME2ACTION.items()}


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def draw_board(screen, board):
    screen.fill(BG)
    pygame.draw.rect(
        screen, BORDER,
        (MARGIN - 2, MARGIN - 2, BOARD_W * CELL + 4, BOARD_H * CELL + 4),
        2
    )
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL
            pygame.draw.rect(screen, GRID, (x, y, CELL, CELL), 1)
            if board[r, c] != 0:
                pygame.draw.rect(screen, BLOCK, (x + 1, y + 1, CELL - 2, CELL - 2), 0)


# ==========================================================
#   可靠获取“锁定棋盘”
# ==========================================================
def detect_locked_board_accessor(env):
    candidates = []

    for name in ["get_board", "get_locked_board", "get_locked", "get_state", "state"]:
        if hasattr(env, name) and callable(getattr(env, name)):
            candidates.append(("method", name))

    for name in ["locked", "locked_board", "static_board", "board", "_board", "grid", "field", "matrix"]:
        if hasattr(env, name):
            candidates.append(("attr", name))

    def normalize(arr):
        arr = np.array(arr, copy=True)
        if arr.shape == (BOARD_H, BOARD_W):
            return arr.astype(np.int8)
        if arr.ndim == 3 and arr.shape[0] == BOARD_H and arr.shape[1] == BOARD_W:
            return arr[:, :, 0].astype(np.int8)
        if arr.ndim == 3 and arr.shape[1] == BOARD_H and arr.shape[2] == BOARD_W:
            return arr[0].astype(np.int8)
        return None

    for kind, name in candidates:
        try:
            raw = getattr(env, name)() if kind == "method" else getattr(env, name)
            out = normalize(raw)
            if out is not None:
                print(f"[DETECT] locked board accessor = {kind}:{name} shape={out.shape}")
                return (lambda e, _k=kind, _n=name: normalize(getattr(e, _n)() if _k == "method" else getattr(e, _n)))
        except Exception:
            pass

    raise RuntimeError("Cannot detect locked board accessor from env. Check task_2_1.py board storage.")


LOCKED_BOARD = None


def extract_static_board(env):
    global LOCKED_BOARD
    if LOCKED_BOARD is None:
        LOCKED_BOARD = detect_locked_board_accessor(env)
    board = LOCKED_BOARD(env)
    if board is None:
        raise RuntimeError("LOCKED_BOARD accessor returned None")
    return np.array(board, copy=True).astype(np.int8)


def get_board_with_piece(env):
    board = extract_static_board(env).astype(np.int16)
    cells = getattr(env, "cur_cells", None)
    pid = getattr(env, "cur_id", 1)
    if cells is not None:
        for (r, c) in cells:
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                board[r, c] = pid
    return board.astype(np.int8)


# ==========================================================
#   Dellacherie Planner（稳定版：评估+中心偏好）
# ==========================================================
class DellacheriePlanner:
    def __init__(self):
        self.w_lines = 1.10
        self.w_holes = -2.50
        self.w_agg_height = -0.45
        self.w_max_height = -0.90
        self.w_bump = -0.35
        self.center_weight = 0.15

    @staticmethod
    def _col_heights(board):
        h, w = board.shape
        heights = []
        for c in range(w):
            col = board[:, c]
            nz = np.where(col != 0)[0]
            heights.append(0 if len(nz) == 0 else h - nz[0])
        return heights

    @staticmethod
    def _count_holes(board):
        h, w = board.shape
        holes = 0
        for c in range(w):
            col = board[:, c]
            nz = np.where(col != 0)[0]
            if len(nz) == 0:
                continue
            top = nz[0]
            holes += int(np.sum(col[top:] == 0))
        return holes

    @staticmethod
    def _bumpiness(heights):
        return sum(abs(heights[i] - heights[i + 1]) for i in range(len(heights) - 1))

    def evaluate(self, board):
        board = np.array(board, copy=False)
        lines = int(np.sum(np.all(board != 0, axis=1)))
        holes = self._count_holes(board)
        heights = self._col_heights(board)
        agg_h = float(sum(heights))
        max_h = float(max(heights)) if heights else 0.0
        bump = float(self._bumpiness(heights))
        return (self.w_lines * lines +
                self.w_holes * holes +
                self.w_agg_height * agg_h +
                self.w_max_height * max_h +
                self.w_bump * bump)

    def _settle_sim(self, sim):
        # ★关键：用 SOFT_DROP 推进结算，比 NONE 更像“时间流逝”
        before_locked = extract_static_board(sim)
        before_id = getattr(sim, "cur_id", None)
        for _ in range(4):
            sim.step(A_SOFT_DROP)
            after_locked = extract_static_board(sim)
            after_id = getattr(sim, "cur_id", None)
            if not np.array_equal(after_locked, before_locked):
                return
            if before_id is not None and after_id is not None and after_id != before_id:
                return

    def plan(self, env):
        if getattr(env, "cur_cells", None) is None:
            return ["SOFT_DROP"]

        best = None
        best_key = None
        center_target = (BOARD_W - 1) / 2.0

        for rot_k in range(4):
            sim0 = copy.deepcopy(env)
            for _ in range(rot_k):
                sim0.step(A_ROTATE)

            cells0 = getattr(sim0, "cur_cells", None)
            if not cells0:
                continue

            cols = [c for _, c in cells0]
            minc, maxc = min(cols), max(cols)
            dx_min = -(minc)
            dx_max = (BOARD_W - 1 - maxc)
            all_dx = list(range(dx_min, dx_max + 1))

            safe_dx = []
            for dx in all_dx:
                left_col = minc + dx
                right_col = maxc + dx
                if left_col < EDGE_MARGIN:
                    continue
                if right_col > (BOARD_W - 1 - EDGE_MARGIN):
                    continue
                safe_dx.append(dx)
            if not safe_dx:
                safe_dx = all_dx

            def dx_order_key(d):
                center_after = (minc + d + maxc + d) / 2.0
                return (abs(center_after - center_target), abs(d))
            dx_list = sorted(safe_dx, key=dx_order_key)

            for dx in dx_list:
                sim = copy.deepcopy(sim0)

                if dx < 0:
                    for _ in range(-dx):
                        sim.step(A_LEFT)
                elif dx > 0:
                    for _ in range(dx):
                        sim.step(A_RIGHT)

                sim.step(A_HARD_DROP)
                self._settle_sim(sim)

                board_after = extract_static_board(sim)
                base = self.evaluate(board_after)

                center_after = (minc + dx + maxc + dx) / 2.0
                center_dist = abs(center_after - center_target)
                score = base - self.center_weight * center_dist

                moves_len = rot_k + abs(dx) + 1
                key = (-score, center_dist, moves_len, rot_k, abs(dx))

                if best_key is None or key < best_key:
                    best_key = key
                    best = (rot_k, dx)

        if best is None:
            return ["SOFT_DROP"]

        rot_k, dx = best
        seq = []
        seq += ["ROTATE"] * rot_k
        if dx < 0:
            seq += ["LEFT"] * (-dx)
        elif dx > 0:
            seq += ["RIGHT"] * dx
        seq += ["HARD_DROP"]

        if len(seq) > MAX_PLAN_STEPS:
            seq = seq[:MAX_PLAN_STEPS - 1] + ["HARD_DROP"]
        return seq


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("TASK4_SURFACE_DRAWN")
    clock = pygame.time.Clock()

    env = TetrisEnv()
    planner = DellacheriePlanner()
    print("Using built-in DellacheriePlanner (Task4).")
    print("Task4 running. Press ESC to exit gracefully.")

    env.reset()

    frames, actions = [], []
    action_counts = {}

    plan_queue = []
    hold_left = 0
    holding_action_name = None

    # ★只在 HARD_DROP 后插入的结算帧（不会抢占计划）
    settle_left = 0

    # ★用 cur_id 变化判断“新块”，最稳
    last_piece_id = getattr(env, "cur_id", None)
    need_plan = True

    running = True
    while running:
        clock.tick(FPS)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        cur_id = getattr(env, "cur_id", None)
        if cur_id is not None and cur_id != last_piece_id:
            need_plan = True
            last_piece_id = cur_id

        board = get_board_with_piece(env)

        if need_plan and getattr(env, "cur_cells", None) is not None and settle_left == 0:
            plan_queue = planner.plan(env)
            print(f"[PLAN] len={len(plan_queue)} seq={plan_queue}")
            need_plan = False

        # ====== 动作选择：计划优先；结算只在 hard drop 后执行 ======
        if settle_left > 0:
            a = SETTLE_ACTION
            action_name = ACTION2NAME[a]
            settle_left -= 1
        else:
            if hold_left > 0 and holding_action_name is not None:
                action_name = holding_action_name
                hold_left -= 1
            else:
                if plan_queue:
                    action_name = plan_queue.pop(0)
                else:
                    action_name = ACTION2NAME[IDLE_ACTION]
                holding_action_name = action_name
                hold_left = max(0, ACTION_HOLD_FRAMES - 1)

            a = NAME2ACTION[action_name]

        # 记录对齐
        frames.append(board)
        actions.append(action_name)
        action_counts[action_name] = action_counts.get(action_name, 0) + 1

        obs, reward, done, info = env.step(a)

        if a == A_HARD_DROP:
            settle_left = SETTLE_FRAMES_AFTER_HARDDROP
            need_plan = True  # hard drop 后下一块必需重新规划

        draw_board(screen, get_board_with_piece(env))
        pygame.display.flip()

        if done:
            break

    pygame.quit()

    tag = now_tag()
    frames_np = np.stack(frames, axis=0) if frames else np.zeros((0, BOARD_H, BOARD_W), dtype=np.int8)
    frames_path = os.path.join(OUT_DIR, f"frames_{tag}.npy")
    actions_path = os.path.join(OUT_DIR, f"actions_{tag}.csv")

    np.save(frames_path, frames_np)
    with open(actions_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["action"])
        for a in actions:
            w.writerow([a])

    print("\nSaved:")
    print(f" - {frames_path} shape= {frames_np.shape}")
    print(f" - {actions_path} rows= {len(actions)}")
    print("Action counts:", action_counts)


if __name__ == "__main__":
    main()
