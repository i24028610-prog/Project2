# tetris_env.py
import random
import numpy as np

BOARD_H, BOARD_W = 18, 14

# 动作标签：全项目统一（任务1/2/3/7都用这一套）
A_NONE = "NONE"
A_LEFT = "LEFT"
A_RIGHT = "RIGHT"
A_ROTATE = "ROTATE"
A_SOFT_DROP = "SOFT_DROP"
A_HARD_DROP = "HARD_DROP"

ACTIONS = [A_NONE, A_LEFT, A_RIGHT, A_ROTATE, A_SOFT_DROP, A_HARD_DROP]

# 7种方块编号：1..7（语义固定）
# 1=I, 2=O, 3=T, 4=S, 5=Z, 6=J, 7=L
PIECE_IDS = {
    "I": 1, "O": 2, "T": 3, "S": 4, "Z": 5, "J": 6, "L": 7
}

# 形状定义：使用方块坐标(相对原点)，原点在piece的(0,0)
SHAPES = {
    "I": [(0, 1), (1, 1), (2, 1), (3, 1)],
    "O": [(1, 0), (2, 0), (1, 1), (2, 1)],
    "T": [(1, 0), (0, 1), (1, 1), (2, 1)],
    "S": [(1, 0), (2, 0), (0, 1), (1, 1)],
    "Z": [(0, 0), (1, 0), (1, 1), (2, 1)],
    "J": [(0, 0), (0, 1), (1, 1), (2, 1)],
    "L": [(2, 0), (0, 1), (1, 1), (2, 1)],
}


def rotate_cells_cw(cells):
    """顺时针旋转： (x,y) -> (y, -x)，然后平移到非负坐标"""
    rotated = [(y, -x) for (x, y) in cells]
    min_x = min(x for x, y in rotated)
    min_y = min(y for x, y in rotated)
    rotated = [(x - min_x, y - min_y) for x, y in rotated]
    return rotated


class TetrisEnv:
    """
    任务2：pygame仿真用的环境
    - board坐标方向：row=0顶部, col=0左侧
    - 输出矩阵：18x14，0空，1..7方块编号（包含正在下落的方块）
    """
    def __init__(self, seed=None):
        self.rng = random.Random(seed)
        self.locked = np.zeros((BOARD_H, BOARD_W), dtype=np.int8)  # 已锁定
        self.cur_name = None
        self.cur_id = 0
        self.cur_cells = None  # 相对坐标列表
        self.cur_row = 0
        self.cur_col = 0
        self.game_over = False
        self.score = 0
        self.lines = 0

    def reset(self):
        self.locked[:] = 0
        self.game_over = False
        self.score = 0
        self.lines = 0
        self._spawn_new_piece()
        return self.get_board()

    def _spawn_new_piece(self):
        name = self.rng.choice(list(SHAPES.keys()))
        self.cur_name = name
        self.cur_id = PIECE_IDS[name]
        self.cur_cells = list(SHAPES[name])

        # 出生位置：靠上居中
        # 注意：row=0为顶部，所以从第0行开始落下
        self.cur_row = 0
        self.cur_col = (BOARD_W // 2) - 2

        # 如果一出生就碰撞，游戏结束
        if not self._valid(self.cur_row, self.cur_col, self.cur_cells):
            self.game_over = True

    def _valid(self, row, col, cells):
        for (x, y) in cells:
            r = row + y
            c = col + x
            if r < 0 or r >= BOARD_H or c < 0 or c >= BOARD_W:
                return False
            if self.locked[r, c] != 0:
                return False
        return True

    def _lock_piece(self):
        for (x, y) in self.cur_cells:
            r = self.cur_row + y
            c = self.cur_col + x
            if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                self.locked[r, c] = self.cur_id

        cleared = self._clear_lines()
        self._spawn_new_piece()
        return cleared

    def _clear_lines(self):
        # 满行：全非0
        full = [r for r in range(BOARD_H) if np.all(self.locked[r, :] != 0)]
        if not full:
            return 0

        # 删除这些行，在顶部补0行
        remain = [r for r in range(BOARD_H) if r not in full]
        new_locked = np.zeros_like(self.locked)
        new_rows = BOARD_H - len(full)
        new_locked[len(full):, :] = self.locked[remain, :]
        self.locked = new_locked

        self.lines += len(full)
        # 简单计分（你后续可在任务3/7改更细）
        self.score += [0, 100, 300, 500, 800][len(full)]
        return len(full)

    def _try_move(self, drow, dcol):
        nr = self.cur_row + drow
        nc = self.cur_col + dcol
        if self._valid(nr, nc, self.cur_cells):
            self.cur_row = nr
            self.cur_col = nc
            return True
        return False

    def _try_rotate(self):
        if self.cur_name == "O":
            return True  # O旋转不变

        rotated = rotate_cells_cw(self.cur_cells)

        # 简单“踢墙”偏移尝试（够用来做任务2/3）
        kicks = [(0, 0), (0, -1), (0, 1), (0, -2), (0, 2), (-1, 0), (1, 0)]
        for drow, dcol in kicks:
            nr = self.cur_row + drow
            nc = self.cur_col + dcol
            if self._valid(nr, nc, rotated):
                self.cur_row = nr
                self.cur_col = nc
                self.cur_cells = rotated
                return True
        return False

    def _hard_drop(self):
        # 一直下落到不能下
        while self._try_move(1, 0):
            pass
        cleared = self._lock_piece()
        return cleared

    def step(self, action):
        """
        每一步对应“1帧”：
        - 输入 action 字符串（必须是 ACTIONS 之一）
        - 返回：obs(18x14), reward, done, info
        """
        if self.game_over:
            return self.get_board(), 0.0, True, {"score": self.score, "lines": self.lines}

        if action not in ACTIONS:
            action = A_NONE

        reward = 0.0
        cleared = 0

        if action == A_LEFT:
            self._try_move(0, -1)
        elif action == A_RIGHT:
            self._try_move(0, 1)
        elif action == A_ROTATE:
            self._try_rotate()
        elif action == A_SOFT_DROP:
            # 软降：能下就下1格；不能下就锁定
            if not self._try_move(1, 0):
                cleared = self._lock_piece()
        elif action == A_HARD_DROP:
            cleared = self._hard_drop()
        else:
            # NONE：重力自然下落1格；不能下就锁定
            if not self._try_move(1, 0):
                cleared = self._lock_piece()

        if cleared > 0:
            # 你也可以把reward策略放到任务3/7再调
            reward += float(cleared) * 1.0

        done = self.game_over
        info = {"score": self.score, "lines": self.lines}
        return self.get_board(), reward, done, info

    def get_board(self):
        """
        输出观测矩阵（18x14）：
        locked + 当前下落方块（同一语义：0空，1..7方块编号）
        row=0顶部, col=0左侧
        """
        board = self.locked.copy()
        if not self.game_over and self.cur_cells is not None:
            for (x, y) in self.cur_cells:
                r = self.cur_row + y
                c = self.cur_col + x
                if 0 <= r < BOARD_H and 0 <= c < BOARD_W:
                    board[r, c] = self.cur_id
        return board.astype(np.int8)
