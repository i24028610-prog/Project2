import copy
import pygame
import numpy as np
from collections import defaultdict, Counter

from task_2_1 import TetrisEnv, A_NONE
from task_3_1 import HeuristicAgent


def norm_shape(cells):
    pts = [(int(r), int(c)) for r, c in cells]
    minr = min(r for r, _ in pts)
    minc = min(c for _, c in pts)
    return tuple(sorted((r - minr, c - minc) for r, c in pts))


def snap(env):
    return {
        "pid": int(getattr(env, "cur_id")),
        "col": int(getattr(env, "cur_col")),
        "row": int(getattr(env, "cur_row")),
        "shape": norm_shape(getattr(env, "cur_cells")),
        "board01": (np.array(env.get_board()) != 0).astype(np.uint8) if hasattr(env, "get_board") else None
    }


def board_changed(a, b):
    if a["board01"] is None or b["board01"] is None:
        return False
    return not np.array_equal(a["board01"], b["board01"])


def classify(before_none, after_none, before_act, after_act):
    """
    Compare action effect vs NONE baseline.
    """
    # baseline deltas (gravity etc.)
    base_dc = after_none["col"] - before_none["col"]
    base_dr = after_none["row"] - before_none["row"]
    base_shape = (after_none["shape"] != before_none["shape"])
    base_board = board_changed(before_none, after_none)

    # action deltas
    act_dc = after_act["col"] - before_act["col"]
    act_dr = after_act["row"] - before_act["row"]
    act_shape = (after_act["shape"] != before_act["shape"])
    act_board = board_changed(before_act, after_act)
    pid_changed = (after_act["pid"] != before_act["pid"])  # often indicates lock/new piece

    # remove baseline
    dc = act_dc - base_dc
    dr = act_dr - base_dr
    shape = act_shape and (not base_shape)
    lock = (act_board and not base_board) or pid_changed

    if shape:
        return "ROTATE"
    if dc < 0:
        return "LEFT"
    if dc > 0:
        return "RIGHT"
    # hard drop: lock/new piece OR row jump beyond baseline
    if lock or dr >= 2:
        return "HARD_DROP"
    # soft drop: row faster than baseline by 1
    if dr == 1:
        return "SOFT_DROP"
    return "NONE"


def main():
    pygame.init()
    pygame.display.set_mode((200, 200))  # just to keep pygame happy
    clock = pygame.time.Clock()

    env = TetrisEnv()
    env.reset()
    agent = HeuristicAgent()

    id_votes = defaultdict(Counter)

    print("Running fixed inference... Press ESC to stop.")
    for t in range(600):
        for ev in pygame.event.get():
            if ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                pygame.quit()
                return

        # choose a candidate action from agent (take the first in its planned seq)
        pid = int(getattr(env, "cur_id"))
        px  = int(getattr(env, "cur_col"))
        py  = int(getattr(env, "cur_row"))
        prot_guess = 0
        board01 = (np.array(env.get_board()) != 0).astype(np.uint8)

        seq = agent.plan_action_sequence(board01, pid, px, py, prot_guess)
        if not seq:
            act = A_NONE
        else:
            act = seq[0]

        # clone env for baseline and action branch
        env_none = copy.deepcopy(env)
        env_act  = copy.deepcopy(env)

        b_none = snap(env_none)
        b_act  = snap(env_act)

        env_none.step(A_NONE)
        env_act.step(act)

        a_none = snap(env_none)
        a_act  = snap(env_act)

        kind = classify(b_none, a_none, b_act, a_act)

        if isinstance(act, (int, np.integer)):
            id_votes[int(act)][kind] += 1

        # advance the real env with the action (so we explore)
        env.step(act)

        if hasattr(env, "game_over") and env.game_over:
            env.reset()

        clock.tick(120)

    pygame.quit()

    print("\n=== Inferred mapping (fixed, baseline-diff) ===")
    for aid in sorted(id_votes.keys()):
        votes = id_votes[aid]
        best = votes.most_common(1)[0][0]
        print(f"ID_{aid} -> {best}   votes={dict(votes)}")


if __name__ == "__main__":
    main()
