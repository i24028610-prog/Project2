# dataset_logger.py
import csv
import numpy as np

class DatasetLogger:
    """
    任务2数据记录器：
    - 每帧记录一个18x14矩阵
    - 每帧记录一个动作字符串（没操作就写 NONE）
    - 最终保证 frames数 == actions数
    """
    def __init__(self):
        self.frames = []
        self.actions = []

    def add(self, board_18x14, action_str):
        board = np.asarray(board_18x14, dtype=np.int8)
        assert board.shape == (18, 14), f"board must be (18,14), got {board.shape}"
        self.frames.append(board.copy())
        self.actions.append(str(action_str))

    def save(self, frames_path="task2_frames.npy", actions_path="task2_actions.csv"):
        frames = np.stack(self.frames, axis=0).astype(np.int8) if self.frames else np.zeros((0, 18, 14), dtype=np.int8)

        # actions写csv：一行一个动作（和任务1一致的最常见写法）
        with open(actions_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["action"])
            for a in self.actions:
                w.writerow([a])

        np.save(frames_path, frames)
        return frames.shape[0], len(self.actions)
