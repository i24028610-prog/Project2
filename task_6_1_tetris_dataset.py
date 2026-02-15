import numpy as np
import torch
from torch.utils.data import Dataset

def one_hot_board(board_18x14: np.ndarray, num_classes: int = 8) -> torch.Tensor:
    """
    board: (18,14) int values in [0, num_classes-1]
    return: (C,18,14) float
    """
    b = np.clip(board_18x14, 0, num_classes - 1).astype(np.int64)
    oh = np.eye(num_classes, dtype=np.float32)[b]          # (18,14,C)
    oh = np.transpose(oh, (2, 0, 1))                       # (C,18,14)
    return torch.from_numpy(oh)

class TetrisImitationDataset(Dataset):
    """
    Expect an .npz with at least:
      boards (N,18,14)
      piece_ids (N,)
      rots (N,)
      target_xs (N,)
      target_rots (N,)

    Optional:
      xs (N,), ys (N,), next_piece_ids (N,)
    """
    def __init__(self, npz_path: str, num_board_classes: int = 8):
        data = np.load(npz_path, allow_pickle=True)

        self.boards = data["boards"]          # (N,18,14)
        self.piece_ids = data["piece_ids"]    # (N,)
        self.rots = data["rots"]              # (N,)
        self.target_xs = data["target_xs"]    # (N,)
        self.target_rots = data["target_rots"]# (N,)

        self.xs = data["xs"] if "xs" in data.files else None
        self.ys = data["ys"] if "ys" in data.files else None
        self.next_piece_ids = data["next_piece_ids"] if "next_piece_ids" in data.files else None

        self.num_board_classes = num_board_classes

    def __len__(self):
        return int(self.boards.shape[0])

    def __getitem__(self, idx: int):
        board = self.boards[idx]
        board_t = one_hot_board(board, self.num_board_classes)  # (C,18,14)

        # piece feature vector（缺什么就补0）
        piece_id = int(self.piece_ids[idx])
        rot = int(self.rots[idx])

        piece_oh = np.zeros(7, dtype=np.float32)
        if 1 <= piece_id <= 7:
            piece_oh[piece_id - 1] = 1.0

        rot_oh = np.zeros(4, dtype=np.float32)
        if 0 <= rot <= 3:
            rot_oh[rot] = 1.0

        x = float(self.xs[idx]) if self.xs is not None else 0.0
        y = float(self.ys[idx]) if self.ys is not None else 0.0

        next_oh = np.zeros(7, dtype=np.float32)
        if self.next_piece_ids is not None:
            nxt = int(self.next_piece_ids[idx])
            if 1 <= nxt <= 7:
                next_oh[nxt - 1] = 1.0

        feat = np.concatenate([piece_oh, rot_oh, np.array([x, y], dtype=np.float32), next_oh], axis=0)
        feat_t = torch.from_numpy(feat)  # (7+4+2+7=20)

        y_x = torch.tensor(int(self.target_xs[idx]), dtype=torch.long)
        y_rot = torch.tensor(int(self.target_rots[idx]), dtype=torch.long)

        return board_t, feat_t, y_x, y_rot
