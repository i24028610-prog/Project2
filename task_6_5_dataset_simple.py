import numpy as np
import torch
from torch.utils.data import Dataset

def one_hot_board(board_18x14: np.ndarray, num_classes: int = 8) -> torch.Tensor:
    b = np.clip(board_18x14, 0, num_classes - 1).astype(np.int64)
    oh = np.eye(num_classes, dtype=np.float32)[b]   # (18,14,C)
    oh = np.transpose(oh, (2, 0, 1))                # (C,18,14)
    return torch.from_numpy(oh)

class Task6Dataset(Dataset):
    """
    npz includes:
      boards: (N,18,14) int8
      piece : (N,) int32  (encoded)
      target: (N,) int32  (encoded placement_id)
    """
    def __init__(self, npz_path: str):
        d = np.load(npz_path, allow_pickle=True)
        self.boards = d["boards"].astype(np.int8)

        def pick_key(d, candidates):
            for k in candidates:
                if k in d.files:
                    return k
            raise KeyError(f"None of {candidates} found. Available keys: {d.files}")

        piece_key = pick_key(d, ["piece", "pieces", "piece_ids", "cur_piece", "cur_piece_label"])
        target_key = pick_key(d, ["target", "targets", "target_label", "placement", "placement_id"])

        self.piece = d[piece_key].astype(np.int64)
        self.target = d[target_key].astype(np.int64)

        print(f"[Dataset] using piece_key={piece_key}, target_key={target_key}")

    def __len__(self):
        return self.boards.shape[0]

    def __getitem__(self, idx):
        board = one_hot_board(self.boards[idx])  # (8,18,14)

        # piece 作为一个整数特征（Embedding更稳）
        piece_id = torch.tensor(self.piece[idx], dtype=torch.long)

        # target 作为分类标签
        y = torch.tensor(self.target[idx], dtype=torch.long)
        return board, piece_id, y
