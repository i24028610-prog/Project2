# run_task2_collect_human.py
import pygame
import numpy as np

from task_2_1 import (
    TetrisEnv, BOARD_H, BOARD_W,
    A_NONE, A_LEFT, A_RIGHT, A_ROTATE, A_SOFT_DROP, A_HARD_DROP
)
from task_2_2 import DatasetLogger

CELL = 30
MARGIN = 20
WIDTH = MARGIN * 2 + BOARD_W * CELL
HEIGHT = MARGIN * 2 + BOARD_H * CELL

# 颜色只用于显示，不影响矩阵语义
COLORS = {
    0: (25, 25, 25),
    1: (0, 255, 255),    # I
    2: (255, 255, 0),    # O
    3: (160, 0, 240),    # T
    4: (0, 255, 0),      # S
    5: (255, 0, 0),      # Z
    6: (0, 0, 255),      # J
    7: (255, 165, 0),    # L
}
GRID_LINE = (45, 45, 45)
TEXT = (230, 230, 230)

FPS = 5

def draw_board(screen, board):
    screen.fill((18, 18, 18))
    # board: row=0顶部，col=0左侧
    for r in range(BOARD_H):
        for c in range(BOARD_W):
            v = int(board[r, c])
            x = MARGIN + c * CELL
            y = MARGIN + r * CELL
            pygame.draw.rect(screen, COLORS.get(v, (200, 200, 200)), (x, y, CELL, CELL))
            pygame.draw.rect(screen, GRID_LINE, (x, y, CELL, CELL), 1)

def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Task2 Tetris (18x14) + Logger")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    env = TetrisEnv(seed=0)
    logger = DatasetLogger()

    board = env.reset()
    running = True

    # 用 KEYDOWN 捕捉“这一帧”的动作，没按键就 NONE
    while running:
        clock.tick(FPS)

        action = A_NONE

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = A_LEFT
                elif event.key == pygame.K_RIGHT:
                    action = A_RIGHT
                elif event.key == pygame.K_UP:
                    action = A_ROTATE
                elif event.key == pygame.K_DOWN:
                    action = A_SOFT_DROP
                elif event.key == pygame.K_SPACE:
                    action = A_HARD_DROP
                elif event.key == pygame.K_r:
                    # 重开：也记录一帧（按你的需要，可删）
                    board = env.reset()
                    action = A_NONE

        # 关键：每一帧都 step 一次，并记录(帧,动作)，保证对齐
        board, reward, done, info = env.step(action)
        logger.add(board, action)

        draw_board(screen, board)

        msg = f"score={info['score']} lines={info['lines']} frames={len(logger.frames)}"
        text = font.render(msg, True, TEXT)
        screen.blit(text, (10, 5))

        if done:
            over = font.render("GAME OVER (press R to reset, ESC to quit)", True, (255, 80, 80))
            screen.blit(over, (10, 25))

        pygame.display.flip()

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

    n_frames, n_actions = logger.save("task2_frames.npy", "task2_actions.csv")
    pygame.quit()
    print(f"[saved] task2_frames.npy shape={(n_frames, 18, 14)} | task2_actions.csv rows={n_actions}")

if __name__ == "__main__":
    main()
