import numpy as np
frames = np.load("task2_frames.npy")
for i in [0, 1, 2]:
    print("==== frame", i, "====")
    print("top rows:")
    print(frames[i][0])
    print(frames[i][1])
    print("bottom rows:")
    print(frames[i][-2])
    print(frames[i][-1])
