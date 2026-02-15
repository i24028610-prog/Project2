import torch
import torch.nn as nn
import torch.nn.functional as F

class TetrisPolicyNet(nn.Module):
    def __init__(self, board_channels: int = 8, feat_dim: int = 20, hidden: int = 256):
        super().__init__()

        # board encoder (C,18,14) -> embedding
        self.conv = nn.Sequential(
            nn.Conv2d(board_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),   # (18,14)->(9,7)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Flatten()
        )
        # 128 * 9 * 7 = 8064
        self.board_fc = nn.Sequential(
            nn.Linear(128 * 9 * 7, hidden),
            nn.ReLU(inplace=True),
        )

        # feature encoder
        self.feat_fc = nn.Sequential(
            nn.Linear(feat_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        # joint head
        self.joint = nn.Sequential(
            nn.Linear(hidden + 64, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
        )

        self.head_x = nn.Linear(hidden, 14)   # target_x: 0..13
        self.head_rot = nn.Linear(hidden, 4)  # target_rot: 0..3

    def forward(self, board, feat):
        """
        board: (B,C,18,14)
        feat: (B,feat_dim)
        """
        b = self.conv(board)
        b = self.board_fc(b)
        f = self.feat_fc(feat)
        z = self.joint(torch.cat([b, f], dim=1))
        logits_x = self.head_x(z)
        logits_rot = self.head_rot(z)
        return logits_x, logits_rot
