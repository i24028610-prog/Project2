import inspect
from task_3_1 import HeuristicAgent
from task_2_1 import TetrisEnv

agent = HeuristicAgent()
env = TetrisEnv()
try:
    env.reset()
except TypeError:
    env.reset()

print("=== Agent class ===")
print(type(agent))

print("\n=== Agent candidate methods ===")
cands = []
for n in dir(agent):
    if any(k in n.lower() for k in ["act", "choose", "action", "policy", "step", "move", "best"]):
        attr = getattr(agent, n)
        if callable(attr):
            cands.append(n)
print(cands)

print("\n=== Signatures ===")
for n in cands[:20]:
    fn = getattr(agent, n)
    try:
        print(n, "->", inspect.signature(fn))
    except Exception:
        print(n, "-> (no signature)")

# 尝试用不同输入调用，看看返回值长什么样
def try_call(name, x):
    fn = getattr(agent, name)
    try:
        out = fn(x)
        print(f"[CALL {name}(x)] ok  type={type(out)}  value={out}")
    except Exception as e:
        print(f"[CALL {name}(x)] FAIL  {repr(e)}")

print("\n=== Try call with env first ===")
for n in cands[:10]:
    try_call(n, env)

print("\n=== Try call with board/occ if available ===")
# 尝试从 env 拿 board（不同人写法不同）
board = None
for k in ["board", "grid", "field", "state", "matrix", "playfield", "board_mat"]:
    if hasattr(env, k):
        board = getattr(env, k)
        if board is not None:
            break
if board is not None:
    for n in cands[:10]:
        try_call(n, board)
else:
    print("env has no obvious board attribute to test.")
