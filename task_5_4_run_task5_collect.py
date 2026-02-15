# run_task5_collect.py
from __future__ import annotations
import time
import numpy as np
from task_5_1tetris_env import TetrisEnv, BOARD_W
from  task_6_13_nn_agent import NNAgent
agent = NNAgent("checkpoints/task6_xyrot_best.pt")

from task_5_3_dataset_logger import DatasetLogger
import task_5_1tetris_env
print("[env file]", task_5_1tetris_env.__file__)
import task_5_1tetris_env
print("[env file]", task_5_1tetris_env.__file__)

def encode_current_piece_label(pid: int, rot: int, x: int) -> int:
    x = int(np.clip(x, 0, BOARD_W - 1))
    rot = int(rot) % 4
    pid = int(pid)  # 0..6
    return pid * (4 * BOARD_W) + rot * BOARD_W + x  # 0..391

def encode_target_label(target_x: int, target_rot: int) -> int:
    target_x = int(np.clip(target_x, 0, BOARD_W - 1))
    target_rot = int(target_rot) % 4
    return target_rot * BOARD_W + target_x  # 0..55

def main():
    out_dir = "out_task5_detect"
    tag = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time()*1000)%1000:03d}"

    env = TetrisEnv(seed=123)
    agent = NNAgent()
    logger = DatasetLogger(out_dir=out_dir, tag=tag)

    max_steps = 5000  # 你想采多少帧就改这里
    steps = 0

    while steps < max_steps:
        if env.game_over:
            env.reset()
            agent.reset()

        # 让 agent 先根据当前局面更新 last_target（并拿到本步 action）
        action = agent.act(env)

        # 取当前块状态
        cur_label = encode_current_piece_label(env.cur_piece_id, env.cur_rot, env.cur_x)

        # 取PD目标（必须保证 agent.last_target 有值）
        if agent.last_target is None:
            # 极少情况（刚reset第一步）再act一次保证有目标
            _ = agent.act(env)
        tx, trot = agent.last_target
        tgt_label = encode_target_label(tx, trot)

        # 记录帧（棋盘+当前块叠加后的 18x14）
        frame = env.get_frame_matrix()
        logger.log(frame, action, cur_label, tgt_label)

        # 环境推进一步
        env.step(action)

        steps += 1

    logger.save()
    print(f"[OK] saved to {out_dir}/{tag}")
    print(f"frames: {len(logger.frames)} actions: {len(logger.actions)} (alignment OK)")

if __name__ == "__main__":
    main()
