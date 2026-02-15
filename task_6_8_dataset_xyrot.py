import numpy as np
import torch
from torch.utils.data import Dataset

def one_hot_board(board_18x14: np.ndarray, num_classes: int = 8) -> torch.Tensor:
    b = np.clip(board_18x14, 0, num_classes - 1).astype(np.int64)
    oh = np.eye(num_classes, dtype=np.float32)[b]   # (18,14,C)
    oh = np.transpose(oh, (2, 0, 1))                # (C,18,14)
    return torch.from_numpy(oh)

class Task6XYRotDataset(Dataset):
    """
    keys in npz:
      boards (N,18,14) int8
      piece_ids (N,) int64
      rots (N,) int64
      xs, ys, next_piece_ids (optional but present)
      target_xs (N,) int64 in [0,13]
      target_rots (N,) int64 in [0,3]
    """
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.boards = d["boards"].astype(np.int8)
        self.piece_ids = d["piece_ids"].astype(np.int64)
        self.rots = d["rots"].astype(np.int64)
        self.xs = d["xs"].astype(np.float32)
        self.ys = d["ys"].astype(np.float32)
        self.next_piece_ids = d["next_piece_ids"].astype(np.int64)
        self.target_xs = d["target_xs"].astype(np.int64)
        self.target_rots = d["target_rots"].astype(np.int64)

        # 保险：clip 到合法范围
        self.target_xs = np.clip(self.target_xs, 0, 13)
        self.target_rots = np.clip(self.target_rots, 0, 3)
        self.rots = np.clip(self.rots, 0, 3)

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        board = one_hot_board(self.boards[idx])  # (8,18,14)

        # 特征向量：piece_id(embedding更稳) + rot + x,y + next_piece_id
        # 这里用 “整数特征” 交给模型 Embedding/MLP 处理
        feat = {
            "piece_id": torch.tensor(int(self.piece_ids[idx]), dtype=torch.long),
            "rot": torch.tensor(int(self.rots[idx]), dtype=torch.long),
            "x": torch.tensor(float(self.xs[idx]), dtype=torch.float32),
            "y": torch.tensor(float(self.ys[idx]), dtype=torch.float32),
            "next_piece_id": torch.tensor(int(self.next_piece_ids[idx]), dtype=torch.long),
        }

        y_x = torch.tensor(int(self.target_xs[idx]), dtype=torch.long)
        y_rot = torch.tensor(int(self.target_rots[idx]), dtype=torch.long)
        return board, feat, y_x, y_rot
