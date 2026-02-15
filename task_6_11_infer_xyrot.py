import torch
import numpy as np
from task_6_9_model_xyrot import Task6XYRotNet

def one_hot_board(board_18x14: np.ndarray, num_classes=8):
    b = np.clip(board_18x14, 0, num_classes-1).astype(np.int64)
    oh = np.eye(num_classes, dtype=np.float32)[b]   # (18,14,C)
    oh = np.transpose(oh, (2,0,1))                  # (C,18,14)
    return torch.from_numpy(oh)

def load_policy(ckpt_path="checkpoints/task6_xyrot_best.pt", device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    model = Task6XYRotNet(piece_vocab=ckpt["piece_vocab"], next_piece_vocab=ckpt["next_piece_vocab"])
    model.load_state_dict(ckpt["state"])
    model.to(device).eval()
    return model

@torch.no_grad()
def predict(model, board18x14, piece_id, rot, x=0.0, y=0.0, next_piece_id=0, device="cpu"):
    board = one_hot_board(board18x14).unsqueeze(0).to(device)  # (1,8,18,14)

    feat = {
        "piece_id": torch.tensor([piece_id], dtype=torch.long, device=device),
        "rot": torch.tensor([rot], dtype=torch.long, device=device),
        "x": torch.tensor([x], dtype=torch.float32, device=device),
        "y": torch.tensor([y], dtype=torch.float32, device=device),
        "next_piece_id": torch.tensor([next_piece_id], dtype=torch.long, device=device),
    }

    logits_x, logits_rot = model(board, feat)
    tx = int(logits_x.argmax(1).item())
    trot = int(logits_rot.argmax(1).item())
    return tx, trot

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_policy(device=device)
    print("loaded on", device)
