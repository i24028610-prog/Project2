# run_task_3_2.py
import time
import inspect
import pygame
import numpy as np

from task_2_1 import TetrisEnv
from task_3_1_new import PDHeuristicAgent

# board size
try:
    from task_2_1 import BOARD_H, BOARD_W
except Exception:
    BOARD_H, BOARD_W = 18, 14

# render config
CELL = 28
MARGIN = 12
WIDTH = MARGIN * 2 + BOARD_W * CELL
HEIGHT = MARGIN * 2 + BOARD_H * CELL

COLORS = {
    0: (20, 20, 20),
    1: (0, 255, 255),    # I
    2: (255, 255, 0),    # O
    3: (160, 0, 240),    # T
    4: (0, 255, 0),      # S
    5: (255, 0, 0),      # Z
    6: (0, 0, 255),      # J
    7: (255, 165, 0),    # L
}

def make_env():
    sig = inspect.signature(TetrisEnv.__init__)
    params = sig.parameters
    kwargs = {}
    if "render_mode" in params:
        kwargs["render_mode"] = "human"
    elif "render" in params:
        kwargs["render"] = True
    elif "use_pygame" in params:
        kwargs["use_pygame"] = True
    elif "enable_render" in params:
        kwargs["enable_render"] = True
    elif "is_render" in params:
        kwargs["is_render"] = True
    return TetrisEnv(**kwargs)

def step_unpack(ret):
    if isinstance(ret, tuple):
        if len(ret) == 4:
            return ret
        if len(ret) == 3:
            obs, reward, done = ret
            return obs, reward, done, {}
    return ret, 0.0, False, {}

def safe_reset(env):
    if hasattr(env, "reset"):
        return env.reset()
    return None

def _clean_locked(env):
    """
    有些 env 会把“下落中的方块”也写进 locked 导致拖影。
    这里做一个“清洁版 locked”，仅用于影子统计/渲染。
    """
    locked = np.array(getattr(env, "locked", np.zeros((BOARD_H, BOARD_W), dtype=np.int8)), dtype=np.int8)
    pid = int(getattr(env, "cur_id", 0))
    px = int(getattr(env, "cur_col", 0))
    py = int(getattr(env, "cur_row", 0))
    cur_cells = getattr(env, "cur_cells", [])
    # 把当前方块位置上与 pid 相同的格子清掉（避免污染统计）
    for cx, cy in cur_cells:
        gx = px + int(cx)
        gy = py + int(cy)
        if 0 <= gy < BOARD_H and 0 <= gx < BOARD_W and locked[gy, gx] == pid:
            locked[gy, gx] = 0
    return locked

def _shadow_clear_and_score(shadow_locked, shadow_lines, shadow_score):
    """
    影子消行 + 计分（不改 env，只用于统计）
    """
    full = np.where(np.all(shadow_locked != 0, axis=1))[0]
    k = int(len(full))
    if k > 0:
        keep = [r for r in range(BOARD_H) if r not in full.tolist()]
        shadow_locked = np.vstack([
            np.zeros((k, BOARD_W), dtype=shadow_locked.dtype),
            shadow_locked[keep, :]
        ])
        shadow_lines += k
        shadow_score += 100 * k  # 简单计分：每行100分（你也可按作业规则改）
    return shadow_locked, shadow_lines, shadow_score

def draw_board(screen, locked, env):
    screen.fill((0, 0, 0))

    # draw locked
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            v = int(locked[r, c])
            if v != 0:
                color = COLORS.get(v, (200, 200, 200))
                x = MARGIN + c * CELL
                y = MARGIN + r * CELL
                pygame.draw.rect(screen, color, (x, y, CELL - 1, CELL - 1))

    # draw current falling piece (always from env state)
    pid = int(getattr(env, "cur_id", 0))
    px = int(getattr(env, "cur_col", 0))
    py = int(getattr(env, "cur_row", 0))
    cur_cells = getattr(env, "cur_cells", [])
    color = COLORS.get(pid, (240, 240, 240))
    for cx, cy in cur_cells:
        gx = px + int(cx)
        gy = py + int(cy)
        if 0 <= gx < BOARD_W and 0 <= gy < BOARD_H:
            x = MARGIN + gx * CELL
            y = MARGIN + gy * CELL
            pygame.draw.rect(screen, color, (x, y, CELL - 1, CELL - 1))

    # grid
    grid_color = (40, 40, 40)
    for r in range(BOARD_H + 1):
        y = MARGIN + r * CELL
        pygame.draw.line(screen, grid_color, (MARGIN, y), (MARGIN + BOARD_W * CELL, y), 1)
    for c in range(BOARD_W + 1):
        x = MARGIN + c * CELL
        pygame.draw.line(screen, grid_color, (x, MARGIN), (x, MARGIN + BOARD_H * CELL), 1)

    pygame.display.flip()

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Tetris Task3 (ESC to quit)")

    env = make_env()
    agent = PDHeuristicAgent(debug_actions=True)

    safe_reset(env)

    clock = pygame.time.Clock()
    last_print = time.time()

    # shadow stats (does NOT affect env)
    shadow_lines = 0
    shadow_score = 0

    # 用 env.locked 的“清洁版”做影子棋盘基底
    shadow_locked = _clean_locked(env).copy()

    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    running = False

            if not running:
                break

            # step
            action = agent.act(env)
            obs, reward, done, info = step_unpack(env.step(action))

            # --- shadow update (stat only) ---
            # 用“清洁版 locked”同步 shadow_locked 的已锁定部分
            # 影子层面我们只关心“哪些格子最终锁定了”，所以每帧覆盖同步即可
            shadow_locked = _clean_locked(env).copy()

            # 如果出现满行，env 可能不清；我们影子清并计分
            shadow_locked, shadow_lines, shadow_score = _shadow_clear_and_score(
                shadow_locked, shadow_lines, shadow_score
            )

            # render: 用 shadow_locked 来画已落地方块（这样就算 env 拖影也不会显示怪）
            draw_board(screen, shadow_locked, env)

            # log every 1s
            now = time.time()
            if now - last_print >= 1.0:
                env_score = getattr(env, "score", None)
                env_lines = getattr(env, "lines", None)
                game_over = getattr(env, "game_over", False)
                print(
                    f"[running] env_score={env_score} env_lines={env_lines} "
                    f"| shadow_score={shadow_score} shadow_lines={shadow_lines} "
                    f"game_over={game_over} last_action={action}"
                )
                last_print = now

            if done or getattr(env, "game_over", False):
                safe_reset(env)
                shadow_locked = _clean_locked(env).copy()

            clock.tick(30)

    except KeyboardInterrupt:
        print("\n[quit] Ctrl+C -> exit gracefully.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
