import os, random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split

from task_6_5_dataset_simple import Task6Dataset
from task_6_6_model_simple import Task6Net

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def eval_one(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    loss_sum = 0.0
    for board, piece_id, y in loader:
        board = board.to(device)
        piece_id = piece_id.to(device)
        y = y.to(device)

        logits = model(board, piece_id)
        loss = F.cross_entropy(logits, y)

        pred = logits.argmax(1)
        total += y.size(0)
        correct += (pred == y).sum().item()
        loss_sum += loss.item() * y.size(0)

    return loss_sum / total, correct / total

def main():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    npz_path = r"data\task6_imitation_dataset.npz"
    ds = Task6Dataset(npz_path)

    # 自动推断类别数（很关键）
    num_classes = int(ds.target.max()) + 1
    piece_vocab = int(ds.piece.max()) + 1
    print("num_classes =", num_classes, "piece_vocab =", piece_vocab)

    # split
    n = len(ds)
    n_train = int(n * 0.9)
    train_ds, val_ds = random_split(ds, [n_train, n - n_train])

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = Task6Net(piece_vocab=piece_vocab, out_classes=num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)

    best = 1e9
    os.makedirs("checkpoints", exist_ok=True)
    save_path = "checkpoints/task6_best.pt"

    for epoch in range(1, 21):
        model.train()
        loss_sum = 0.0
        total = 0

        for board, piece_id, y in train_loader:
            board = board.to(device)
            piece_id = piece_id.to(device)
            y = y.to(device)

            logits = model(board, piece_id)
            loss = F.cross_entropy(logits, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            loss_sum += loss.item() * y.size(0)
            total += y.size(0)

        train_loss = loss_sum / total
        val_loss, val_acc = eval_one(model, val_loader, device)
        print(f"Epoch {epoch:02d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_acc={val_acc:.3f}")

        if val_loss < best:
            best = val_loss
            torch.save({"state": model.state_dict(), "num_classes": num_classes, "piece_vocab": piece_vocab}, save_path)
            print("  saved ->", save_path)

if __name__ == "__main__":
    main()
