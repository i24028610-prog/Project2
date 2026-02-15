import torch
import torch.nn as nn

class Task6Net(nn.Module):
    def __init__(self, board_channels=8, piece_vocab=64, emb_dim=16, hidden=256, out_classes=56):
        super().__init__()

        self.piece_emb = nn.Embedding(piece_vocab, emb_dim)

        self.conv = nn.Sequential(
            nn.Conv2d(board_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),          # (18,14)->(9,7)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(128 * 9 * 7 + emb_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Linear(hidden, out_classes)

    def forward(self, board, piece_id):
        x_board = self.conv(board)
        x_piece = self.piece_emb(piece_id)
        x = torch.cat([x_board, x_piece], dim=1)
        x = self.fc(x)
        return self.head(x)
