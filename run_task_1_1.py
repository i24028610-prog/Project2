import time
import csv
import numpy as np
import cv2
import pygame
import random
import sys
import pygame
pygame.display.set_caption("MY_TETRIS_REAL")
# =========================
# 任务一：pygame画面 + OpenCV识别 18x14 + 动作CSV
# =========================

H, W = 18, 14            # 棋盘：18行14列
CELL = 28                # 每格像素（你可改大/改小）
MARGIN_X = 80            # 棋盘左上角x
MARGIN_Y = 60            # 棋盘左上角y
FPS = 30

# 颜色（pygame画用）
BG = (25, 25, 25)
GRID_LINE = (45, 45, 45)
BLOCK = (40, 200, 255)
LOCKED = (220, 220, 220)
TEXT = (230, 230, 230)
YELLOW = (255, 255, 0)
RED = (255, 80, 80)
GREEN = (80, 255, 120)

# OpenCV识别阈值（不准再调）
S_TH = 40
V_TH = 50

A_NONE = "NONE"
A_LEFT = "LEFT"
A_RIGHT = "RIGHT"
A_ROTATE = "ROTATE"
A_SOFT_DROP = "SOFT_DROP"
A_HARD_DROP = "HARD_DROP"
A_QUIT = "QUIT"
A_GRAVITY = "GRAVITY"
# ============ 方块定义（最简版本，7种tetromino） ============
# 用 4x4 矩阵表示，每次旋转90度
SHAPES = {
    "I": [
        ["....",
         "####",
         "....",
         "...."],
    ],
    "O": [
        [".##.",
         ".##.",
         "....",
         "...."],
    ],
    "T": [
        [".#..",
         "###.",
         "....",
         "...."],
    ],
    "S": [
        [".##.",
         "##..",
         "....",
         "...."],
    ],
    "Z": [
        ["##..",
         ".##.",
         "....",
         "...."],
    ],
    "J": [
        ["#...",
         "###.",
         "....",
         "...."],
    ],
    "L": [
        ["..#.",
         "###.",
         "....",
         "...."],
    ]
}

def rotate_matrix(mat4):
    # mat4: list[str] 4 lines
    arr = [list(row) for row in mat4]
    rot = list(zip(*arr[::-1]))
    return ["".join(row) for row in rot]

# 预先生成每种方块的4个旋转态
ROTATIONS = {}
for k, mats in SHAPES.items():
    base = mats[0]
    rots = [base]
    for _ in range(3):
        rots.append(rotate_matrix(rots[-1]))
    # 去重（O方块会重复）
    uniq = []
    for r in rots:
        if r not in uniq:
            uniq.append(r)
    ROTATIONS[k] = uniq

def shape_cells(shape_name, rot_idx, top, left):
    """返回该方块当前旋转态占据的棋盘格子坐标列表[(r,c)...]"""
    mat = ROTATIONS[shape_name][rot_idx]
    cells = []
    for i in range(4):
        for j in range(4):
            if mat[i][j] == "#":
                cells.append((top + i, left + j))
    return cells

def in_bounds(r, c):
    return 0 <= r < H and 0 <= c < W

def collides(board, cells):
    """碰撞：越界或碰到已锁定块"""
    for r, c in cells:
        if c < 0 or c >= W or r >= H:
            return True
        if r >= 0 and board[r][c] == 1:
            return True
    return False

def clear_lines(board):
    """消行：返回(新board, 消除行数)"""
    new_rows = [row for row in board if np.sum(row) < W]
    cleared = H - len(new_rows)
    if cleared > 0:
        pad = [np.zeros(W, dtype=np.uint8) for _ in range(cleared)]
        board2 = np.array(pad + new_rows, dtype=np.uint8)
        return board2, cleared
    return board, 0

# ============ OpenCV 记录器 ============
class Task1Recorder:
    def __init__(self, board_rect, out_frames="frames.npz", out_csv="actions.csv"):
        self.x, self.y, self.w, self.h = board_rect
        self.out_frames = out_frames
        self.out_csv = out_csv
        self.frames = []
        self.rows = []

    def _surface_to_bgr(self, screen_surface) -> np.ndarray:
        rgb = pygame.surfarray.array3d(screen_surface)     # (W,H,3) RGB
        rgb = np.transpose(rgb, (1, 0, 2))                 # -> (H,W,3)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        return bgr

    def _bgr_to_grid(self, bgr: np.ndarray) -> np.ndarray:
        roi = bgr[self.y:self.y+self.h, self.x:self.x+self.w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        cell_h = self.h / H
        cell_w = self.w / W
        grid = np.zeros((H, W), dtype=np.uint8)

        for r in range(H):
            for c in range(W):
                cy = int((r + 0.5) * cell_h)
                cx = int((c + 0.5) * cell_w)

                # 中心5x5平均更稳
                y0, y1 = max(0, cy - 2), min(self.h, cy + 3)
                x0, x1 = max(0, cx - 2), min(self.w, cx + 3)
                patch = hsv[y0:y1, x0:x1]

                s_mean = float(patch[:, :, 1].mean())
                v_mean = float(patch[:, :, 2].mean())

                grid[r, c] = 1 if (s_mean > S_TH and v_mean > V_TH) else 0

        return grid

    def record_frame(self, screen_surface, action: str):
        bgr = self._surface_to_bgr(screen_surface)
        grid = self._bgr_to_grid(bgr)

        idx = len(self.frames)
        self.frames.append(grid)
        self.rows.append((idx, time.time(), action))
        return grid

    def save(self):
        if len(self.frames) == 0:
            print("No frames recorded, nothing saved.")
            return
        frames_np = np.stack(self.frames, axis=0).astype(np.uint8)
        np.savez_compressed(self.out_frames, frames=frames_np)

        with open(self.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["frame_idx", "timestamp", "action"])
            w.writerows(self.rows)

        print("Saved:", self.out_frames, frames_np.shape)
        print("Saved:", self.out_csv, len(self.rows), "rows")

def show_grid_debug(grid: np.ndarray, scale=20):
    vis = (grid * 255).astype(np.uint8)
    vis = cv2.resize(vis, (W * scale, H * scale), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("grid_18x14_debug (white=block)", vis)
    cv2.waitKey(1)

# ============ 取框（在真实游戏画面上点两次） ============
pick_mode = False
clicks = []
BOARD_RECT_USER = None
A_GRAVITY = "GRAVITY"
def handle_pick_rect(event):
    global pick_mode, clicks, BOARD_RECT_USER
    if event.type == pygame.KEYDOWN and event.key == pygame.K_p:
        pick_mode = not pick_mode
        clicks = []
        BOARD_RECT_USER = None
        print("Pick mode:", pick_mode, "| Click TOP-LEFT then BOTTOM-RIGHT of the BOARD")

    if pick_mode and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
        clicks.append(event.pos)
        print("click:", event.pos)
        if len(clicks) == 2:
            x1, y1 = clicks[0]
            x2, y2 = clicks[1]
            x, y = min(x1, x2), min(y1, y2)
            w, h = abs(x2 - x1), abs(y2 - y1)
            BOARD_RECT_USER = (x, y, w, h)
            print("BOARD_RECT =", BOARD_RECT_USER)
            pick_mode = False
            clicks = []

def draw_board(screen, board, active_cells):
    # 背景
    screen.fill(BG)

    # 棋盘区域（我们自己画，所以默认RECT可直接算出来）
    board_rect = pygame.Rect(MARGIN_X, MARGIN_Y, W * CELL, H * CELL)
    pygame.draw.rect(screen, (35, 35, 35), board_rect, border_radius=6)

    # 画网格与方块
    for r in range(H):
        for c in range(W):
            x = MARGIN_X + c * CELL
            y = MARGIN_Y + r * CELL
            pygame.draw.rect(screen, GRID_LINE, (x, y, CELL, CELL), 1)

            if board[r][c] == 1:
                pygame.draw.rect(screen, LOCKED, (x+2, y+2, CELL-4, CELL-4))

    # 画当前活动方块
    for r, c in active_cells:
        if r < 0:
            continue
        x = MARGIN_X + c * CELL
        y = MARGIN_Y + r * CELL
        pygame.draw.rect(screen, BLOCK, (x+2, y+2, CELL-4, CELL-4))

    # 如果在取框模式，画提示
    if pick_mode:
        font = pygame.font.SysFont("Arial", 18)
        tip = font.render("PICK MODE: click TOP-LEFT then BOTTOM-RIGHT of BOARD (press P to cancel)", True, YELLOW)
        screen.blit(tip, (10, 10))

def main():
    pygame.init()
    win_w = MARGIN_X * 2 + W * CELL + 240
    win_h = MARGIN_Y * 2 + H * CELL + 40
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption("MY_TETRIS_REAL")

    pygame.display.set_caption("Tetris Task1 (pygame + OpenCV capture) | Press P to pick BOARD_RECT")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont("Arial", 18)

    board = np.zeros((H, W), dtype=np.uint8)

    # 当前方块
    cur_shape = random.choice(list(ROTATIONS.keys()))
    rot = 0
    top = -2
    left = W // 2 - 2

    drop_timer = 0.0
    drop_interval = 0.5  # 自动下落速度
    score = 0

    # 默认棋盘rect（因为我们自己画出来的，最准）
    default_rect = (MARGIN_X, MARGIN_Y, W * CELL, H * CELL)
    recorder = Task1Recorder(board_rect=default_rect)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        gravity_moved = False  # 本帧是否发生自动下落
        locked_this_tick = False  # 本帧是否发生锁定/消行/生成新块
        gravity_moved = False  # 每一帧重置一次
        action = A_NONE
        instant_action = None  # ROTATE / HARD_DROP / QUIT 这种“一次性动作”优先

        for event in pygame.event.get():
            handle_pick_rect(event)

            if event.type == pygame.QUIT:
                instant_action = A_QUIT
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    instant_action = A_QUIT
                    running = False

                # 一次性动作：用 KEYDOWN 记录（只触发一次）
                if event.key == pygame.K_UP:
                    instant_action = A_ROTATE
                elif event.key == pygame.K_SPACE:
                    instant_action = A_HARD_DROP

        # 连续动作：用“按住状态”每帧采样
        keys = pygame.key.get_pressed()
        hold_action = A_NONE
        if keys[pygame.K_LEFT]:
            hold_action = A_LEFT
        elif keys[pygame.K_RIGHT]:
            hold_action = A_RIGHT
        elif keys[pygame.K_DOWN]:
            hold_action = A_SOFT_DROP

        # 最终本帧动作：一次性动作优先，其次连续动作
        action = instant_action if instant_action is not None else hold_action

        # 处理动作
        if action == A_LEFT:
            cells = shape_cells(cur_shape, rot, top, left - 1)
            if not collides(board, cells):
                left -= 1

        elif action == A_RIGHT:
            cells = shape_cells(cur_shape, rot, top, left + 1)
            if not collides(board, cells):
                left += 1

        elif action == A_ROTATE:
            new_rot = (rot + 1) % len(ROTATIONS[cur_shape])
            cells = shape_cells(cur_shape, new_rot, top, left)
            if not collides(board, cells):
                rot = new_rot

        elif action == A_SOFT_DROP:
            cells = shape_cells(cur_shape, rot, top + 1, left)
            if not collides(board, cells):
                top += 1

        elif action == A_HARD_DROP:
            while True:
                cells = shape_cells(cur_shape, rot, top + 1, left)
                if collides(board, cells):
                    break
                top += 1

        # 自动下落
        # 自动下落
        gravity_moved = False  # 每帧先重置（这一行位置没问题）

        drop_timer += dt
        if drop_timer >= drop_interval:
            drop_timer = 0.0
            cells = shape_cells(cur_shape, rot, top + 1, left)

            if not collides(board, cells):
                top += 1
                gravity_moved = True  # 这一帧发生了自动下落
            else:
                locked_this_tick = True
                # ✅ 碰撞才锁定：把下面“你原来的锁定/消行/新方块”整段挪到这里
                for r, c in shape_cells(cur_shape, rot, top, left):
                    if r < 0:
                        running = False
                        break
                    board[r][c] = 1

                board, cleared = clear_lines(board)
                if cleared > 0:
                    score += cleared * 100

                cur_shape = random.choice(list(ROTATIONS.keys()))
                rot = 0
                top = -2
                left = W // 2 - 2

        active = shape_cells(cur_shape, rot, top, left)
        draw_board(screen, board, active)

        # 侧边信息
        info_x = MARGIN_X + W * CELL + 30
        screen.blit(font.render("Task1 Recording ON", True, GREEN), (info_x, 80))
        screen.blit(font.render(f"Score: {score}", True, TEXT), (info_x, 110))
        screen.blit(font.render("Controls:", True, TEXT), (info_x, 150))
        screen.blit(font.render("LEFT/RIGHT: move", True, TEXT), (info_x, 175))
        screen.blit(font.render("UP: rotate", True, TEXT), (info_x, 200))
        screen.blit(font.render("DOWN: soft drop", True, TEXT), (info_x, 225))
        screen.blit(font.render("SPACE: hard drop", True, TEXT), (info_x, 250))
        screen.blit(font.render("P: pick BOARD_RECT (optional)", True, YELLOW), (info_x, 290))
        screen.blit(font.render("ESC: quit & save", True, RED), (info_x, 315))

        # 画出当前用于OpenCV识别的rect（可视化确认）
        cur_rect = pygame.Rect(recorder.x, recorder.y, recorder.w, recorder.h)
        pygame.draw.rect(screen, (255, 120, 0), cur_rect, 2)

        pygame.display.flip()

        # === 任务一：每帧用OpenCV识别并记录 ===
        if action == A_NONE and gravity_moved:
            action = A_GRAVITY
        should_log = (action != A_NONE) or gravity_moved
        # should_log 的判断你可以先保持不变（或用我下面的更强版本）
        if should_log:
            if action == A_NONE:
                # 这一帧你既然记录了，就给它一个默认动作，避免NONE继续占比过高
                action = A_GRAVITY  # 或者你定义 A_IDLE = "IDLE"
            grid = recorder.record_frame(pygame.display.get_surface(), action)
            show_grid_debug(grid)
        # 只在“有信息”的帧记录：按键动作 / 自动下落 / 锁定
        should_log = (action != A_NONE) or gravity_moved or locked_this_tick

        if should_log:
            # 如果没有按键动作，但这一帧你决定记录，就给它一个默认动作（避免NONE继续占比过高）
            if action == A_NONE:
                action = A_GRAVITY

            grid = recorder.record_frame(pygame.display.get_surface(), action)
            show_grid_debug(grid)

    # 退出保存
    recorder.save()
    cv2.destroyAllWindows()
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
