import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

from task_6_1_tetris_dataset import TetrisImitationDataset
from task_6_2_tetris_model import TetrisPolicyNet

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct_x = 0
    correct_rot = 0
    loss_sum = 0.0

    for board, feat, y_x, y_rot in loader:
        board = board.to(device)
        feat = feat.to(device)
        y_x = y_x.to(device)
        y_rot = y_rot.to(device)

        logits_x, logits_rot = model(board, feat)
        loss = F.cross_entropy(logits_x, y_x) + F.cross_entropy(logits_rot, y_rot)

        pred_x = logits_x.argmax(dim=1)
        pred_rot = logits_rot.argmax(dim=1)

        total += y_x.size(0)
        correct_x += (pred_x == y_x).sum().item()
        correct_rot += (pred_rot == y_rot).sum().item()
        loss_sum += loss.item() * y_x.size(0)

    return {
        "loss": loss_sum / max(total, 1),
        "acc_x": correct_x / max(total, 1),
        "acc_rot": correct_rot / max(total, 1),
        "acc_both": ( (correct_x + correct_rot) / 2 ) / max(total, 1)  # 只是粗略指标
    }

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 改成你汇总后的npz路径（把任务3+任务5都合并进去）
    npz_path = "data/task6_imitation_dataset.npz"
    assert os.path.exists(npz_path), f"Dataset not found: {npz_path}"

    ds = TetrisImitationDataset(npz_path=npz_path)

    # split
    n = len(ds)
    n_train = int(n * 0.9)
    n_val = n - n_train
    train_ds, val_ds = random_split(ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = TetrisPolicyNet(board_channels=8, feat_dim=20, hidden=256).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best_val = 1e9
    save_path = "checkpoints/task6_policy_best.pt"
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    for epoch in range(1, 31):
        model.train()
        loss_sum = 0.0
        total = 0

        for board, feat, y_x, y_rot in train_loader:
            board = board.to(device)
            feat = feat.to(device)
            y_x = y_x.to(device)
            y_rot = y_rot.to(device)

            logits_x, logits_rot = model(board, feat)
            loss = F.cross_entropy(logits_x, y_x) + F.cross_entropy(logits_rot, y_rot)

            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            loss_sum += loss.item() * y_x.size(0)
            total += y_x.size(0)

        train_loss = loss_sum / max(total, 1)
        val_metrics = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch:02d}] train_loss={train_loss:.4f} "
              f"val_loss={val_metrics['loss']:.4f} "
              f"acc_x={val_metrics['acc_x']:.3f} acc_rot={val_metrics['acc_rot']:.3f}")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "feat_dim": 20,
                "board_channels": 8
            }, save_path)
            print(f"  saved -> {save_path}")

    print("Done.")

if __name__ == "__main__":
    main()
