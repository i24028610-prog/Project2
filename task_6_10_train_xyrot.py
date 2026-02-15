import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from task_6_8_dataset_xyrot import Task6XYRotDataset
from task_6_9_model_xyrot import Task6XYRotNet

def set_seed(seed=42):
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
        feat = {k: v.to(device) for k, v in feat.items()}
        y_x = y_x.to(device)
        y_rot = y_rot.to(device)

        logits_x, logits_rot = model(board, feat)
        loss = F.cross_entropy(logits_x, y_x) + F.cross_entropy(logits_rot, y_rot)

        pred_x = logits_x.argmax(1)
        pred_rot = logits_rot.argmax(1)

        total += y_x.size(0)
        correct_x += (pred_x == y_x).sum().item()
        correct_rot += (pred_rot == y_rot).sum().item()
        loss_sum += loss.item() * y_x.size(0)

    return (loss_sum / total, correct_x / total, correct_rot / total)

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = r"data\task6_imitation_dataset.npz"
    ds = Task6XYRotDataset(npz_path)

    # 自动推断 vocab（避免 embedding 越界）
    piece_vocab = int(ds.piece_ids.max()) + 1
    next_piece_vocab = int(ds.next_piece_ids.max()) + 1
    piece_vocab = max(piece_vocab, 8)
    next_piece_vocab = max(next_piece_vocab, 8)

    print("piece_vocab =", piece_vocab, "next_piece_vocab =", next_piece_vocab)

    n = len(ds)
    n_train = int(n * 0.9)
    train_ds, val_ds = random_split(ds, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = Task6XYRotNet(piece_vocab=piece_vocab, next_piece_vocab=next_piece_vocab).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    os.makedirs("checkpoints", exist_ok=True)
    best = 1e9
    save_path = "checkpoints/task6_xyrot_best.pt"

    for epoch in range(1, 21):
        model.train()
        loss_sum = 0.0
        total = 0

        for board, feat, y_x, y_rot in train_loader:
            board = board.to(device)
            feat = {k: v.to(device) for k, v in feat.items()}
            y_x = y_x.to(device)
            y_rot = y_rot.to(device)

            logits_x, logits_rot = model(board, feat)
            loss = F.cross_entropy(logits_x, y_x) + F.cross_entropy(logits_rot, y_rot)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item() * y_x.size(0)
            total += y_x.size(0)

        train_loss = loss_sum / total
        val_loss, acc_x, acc_rot = evaluate(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | acc_x={acc_x:.3f} acc_rot={acc_rot:.3f}")

        if val_loss < best:
            best = val_loss
            torch.save({
                "state": model.state_dict(),
                "piece_vocab": piece_vocab,
                "next_piece_vocab": next_piece_vocab
            }, save_path)
            print("  saved ->", save_path)

if __name__ == "__main__":
    main()
