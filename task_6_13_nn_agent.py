import torch
import numpy as np

from task_6_11_infer_xyrot import load_policy, predict
from task_6_12_plan_actions import plan_actions


def _get_attr(obj, names):
    """按候选名字依次尝试 getattr，取到第一个存在的。"""
    for n in names:
        if hasattr(obj, n):
            return getattr(obj, n)
    return None


def _call_if_callable(x):
    return x() if callable(x) else x


class NNAgent:
    """
    兼容两种调用方式：
      1) action = agent.act(env)          # 旧接口（你的 run_task5_collect 用的是这个）
      2) action = agent.act(board, piece_id, rot, cur_x, ...)  # 新接口

    重点：如果你传 env，它会自动从 env 里取 board/piece/rot/x，并返回动作。
    """

    def __init__(self, ckpt_path="checkpoints/task6_xyrot_best.pt", device=None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = load_policy(ckpt_path, device=device)
        self.pending = []

    def reset(self):
        self.pending = []
        self.last_target = None  # 兼容 run_task5_collect 的检查逻辑

    def _read_from_env(self, env):
        """
        尽量从 env 中读取：
          board18x14, piece_id, rot, cur_x, cur_y, next_piece_id
          动作常量 A_LEFT/A_RIGHT/A_ROTATE/A_HARD_DROP
        """
        # --- board ---
        # 常见命名：env.board / env.grid / env.state / env.get_board()
        board = _get_attr(env, ["board", "grid", "state", "board_mat", "board_matrix", "frame"])
        board = _call_if_callable(board)

        # 有些 env 用方法
        if board is None:
            fn = _get_attr(env, ["get_board", "get_grid", "get_state", "get_frame"])
            if fn is not None:
                board = fn()

        if board is None:
            raise RuntimeError("Cannot find board in env. Try adding env.board or env.get_board().")

        board = np.array(board, dtype=np.int64)
        if board.shape != (18, 14):
            # 有些是 (18,14,?) 或 (H,W) 但没对齐，尽量裁剪/reshape
            board = board.reshape(18, 14)

        # --- piece info ---
        piece_id = _get_attr(env, ["piece_id", "cur_piece_id", "current_piece_id"])
        piece_id = _call_if_callable(piece_id)

        rot = _get_attr(env, ["rot", "cur_rot", "rotation", "current_rot"])
        rot = _call_if_callable(rot)

        cur_x = _get_attr(env, ["x", "cur_x", "piece_x", "current_x"])
        cur_x = _call_if_callable(cur_x)

        cur_y = _get_attr(env, ["y", "cur_y", "piece_y", "current_y"])
        cur_y = _call_if_callable(cur_y)

        next_piece_id = _get_attr(env, ["next_piece_id", "next_id", "next_piece"])
        next_piece_id = _call_if_callable(next_piece_id)

        # 如果 env 把当前方块封装在 env.piece / env.cur_piece 里
        if piece_id is None or rot is None or cur_x is None:
            p = _get_attr(env, ["piece", "cur_piece", "current_piece"])
            p = _call_if_callable(p)
            if p is not None:
                if piece_id is None:
                    piece_id = _get_attr(p, ["id", "piece_id", "type", "kind"])
                    piece_id = _call_if_callable(piece_id)
                if rot is None:
                    rot = _get_attr(p, ["rot", "rotation"])
                    rot = _call_if_callable(rot)
                if cur_x is None:
                    cur_x = _get_attr(p, ["x", "cur_x"])
                    cur_x = _call_if_callable(cur_x)
                if cur_y is None:
                    cur_y = _get_attr(p, ["y", "cur_y"])
                    cur_y = _call_if_callable(cur_y)

        if piece_id is None or rot is None or cur_x is None:
            raise RuntimeError(
                "Cannot find piece_id/rot/cur_x from env. "
                "Please ensure env has attributes like piece_id, rot, x (or env.piece has them)."
            )

        # 默认值
        if cur_y is None:
            cur_y = 0.0
        if next_piece_id is None:
            next_piece_id = 0

        # --- action constants ---
        A_LEFT = _get_attr(env, ["A_LEFT"])
        A_RIGHT = _get_attr(env, ["A_RIGHT"])
        A_ROTATE = _get_attr(env, ["A_ROTATE"])
        A_HARD_DROP = _get_attr(env, ["A_HARD_DROP"])

        # 有的常量定义在 env 模块里，env 上没有；尝试从 env.__class__ 或 env.__dict__
        if A_LEFT is None:
            A_LEFT = _get_attr(env.__class__, ["A_LEFT"])
        if A_RIGHT is None:
            A_RIGHT = _get_attr(env.__class__, ["A_RIGHT"])
        if A_ROTATE is None:
            A_ROTATE = _get_attr(env.__class__, ["A_ROTATE"])
        if A_HARD_DROP is None:
            A_HARD_DROP = _get_attr(env.__class__, ["A_HARD_DROP"])

        if None in [A_LEFT, A_RIGHT, A_ROTATE, A_HARD_DROP]:
            raise RuntimeError("Cannot find action constants A_LEFT/A_RIGHT/A_ROTATE/A_HARD_DROP in env.")

        return board, int(piece_id), int(rot), int(cur_x), float(cur_y), int(next_piece_id), A_LEFT, A_RIGHT, A_ROTATE, A_HARD_DROP

    def act(self, *args, **kwargs):
        """
        支持：
          act(env)
          act(board18x14, piece_id, rot, cur_x, cur_y=0, next_piece_id=0, A_LEFT=..., ...)
        """
        # --- 旧接口：act(env) ---
        if len(args) == 1 and not kwargs:
            env = args[0]
            board, piece_id, rot, cur_x, cur_y, next_piece_id, A_LEFT, A_RIGHT, A_ROTATE, A_HARD_DROP = self._read_from_env(env)
        else:
            # --- 新接口 ---
            board = args[0]
            piece_id = args[1]
            rot = args[2]
            cur_x = args[3]
            cur_y = kwargs.get("cur_y", 0.0)
            next_piece_id = kwargs.get("next_piece_id", 0)
            A_LEFT = kwargs["A_LEFT"]
            A_RIGHT = kwargs["A_RIGHT"]
            A_ROTATE = kwargs["A_ROTATE"]
            A_HARD_DROP = kwargs["A_HARD_DROP"]

        # 如果没有待执行动作，就重新规划一次
        if not self.pending:
            tx, trot = predict(
                self.model,
            board18x14=np.array(board, dtype=np.int64),
                piece_id=int(piece_id),
                rot=int(rot),
                x=float(cur_x),
                y=float(cur_y),
                next_piece_id=int(next_piece_id),
                device=self.device
            )
            self.last_target = (tx, trot)
            self.pending = plan_actions(int(cur_x), int(rot), tx, trot, A_LEFT, A_RIGHT, A_ROTATE, A_HARD_DROP)

        return self.pending.pop(0)
