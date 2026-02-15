import torch
import torch.nn as nn

class Task6XYRotNet(nn.Module):
    def __init__(self, board_channels=8, piece_vocab=16, next_piece_vocab=16, emb_dim=16, hidden=256):
        super().__init__()

        self.piece_emb = nn.Embedding(piece_vocab, emb_dim)
        self.next_piece_emb = nn.Embedding(next_piece_vocab, emb_dim)
        self.rot_emb = nn.Embedding(4, 8)  # rot 0..3

        self.conv = nn.Sequential(
            nn.Conv2d(board_channels, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (18,14)->(9,7)
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )

        # board: 128*9*7=8064
        # feat: piece_emb(16)+next(16)+rot(8)+x+y(2) => 42
        feat_dim = emb_dim + emb_dim + 8 + 2

        self.fc = nn.Sequential(
            nn.Linear(128 * 9 * 7 + feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
        )

        self.head_x = nn.Linear(hidden, 14)  # 0..13
        self.head_rot = nn.Linear(hidden, 4) # 0..3

    def forward(self, board, feat):
        xb = self.conv(board)

        piece_id = feat["piece_id"]
        next_piece_id = feat["next_piece_id"]
        rot = feat["rot"]
        x = feat["x"].unsqueeze(1)  # (B,1)
        y = feat["y"].unsqueeze(1)

        ep = self.piece_emb(piece_id)
        en = self.next_piece_emb(next_piece_id)
        er = self.rot_emb(rot)

        xf = torch.cat([ep, en, er, x, y], dim=1)
        z = torch.cat([xb, xf], dim=1)
        z = self.fc(z)

        return self.head_x(z), self.head_rot(z)
