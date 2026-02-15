# dataset_logger.py
from __future__ import annotations
import os
import csv
import numpy as np

class DatasetLogger:
    def __init__(self, out_dir: str, tag: str):
        self.out_dir = out_dir
        self.tag = tag
        self.base = os.path.join(out_dir, tag)
        os.makedirs(self.base, exist_ok=True)

        self.frames = []
        self.actions = []
        self.cur_piece_label = []
        self.target_label = []

    def log(self, frame_18x14: np.ndarray, action: int, cur_piece_label: int, target_label: int):
        self.frames.append(frame_18x14.astype(np.int8))
        self.actions.append(int(action))
        self.cur_piece_label.append(int(cur_piece_label))
        self.target_label.append(int(target_label))

    def save(self):
        frames = np.stack(self.frames, axis=0) if self.frames else np.zeros((0,18,14), dtype=np.int8)

        np.save(os.path.join(self.base, "frames.npy"), frames)
        np.save(os.path.join(self.base, "cur_piece_label.npy"), np.array(self.cur_piece_label, dtype=np.int32))
        np.save(os.path.join(self.base, "target_label.npy"), np.array(self.target_label, dtype=np.int32))

        with open(os.path.join(self.base, "actions.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["action"])
            for a in self.actions:
                w.writerow([a])

        # quick meta
        with open(os.path.join(self.base, "meta.txt"), "w", encoding="utf-8") as f:
            f.write(f"frames={frames.shape}\n")
            f.write(f"actions={len(self.actions)}\n")
            f.write("alignment=actions==frames\n")
