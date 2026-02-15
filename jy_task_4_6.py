# clean_task4_bad_sets.py
import os, sys, glob, shutil, csv
import numpy as np

VALID_ACTIONS = {"NONE","LEFT","RIGHT","ROTATE","SOFT_DROP","HARD_DROP"}

def read_actions(path):
    actions=[]
    with open(path,"r",encoding="utf-8") as f:
        r=csv.reader(f)
        rows=list(r)
    if not rows:
        return actions
    start = 1 if rows[0] and rows[0][0].strip().lower() in ("action","actions","action_id") else 0
    for row in rows[start:]:
        if row:
            actions.append(row[0].strip())
    return actions

def tag_from(path, prefix, suffix):
    name=os.path.basename(path)
    if not name.startswith(prefix) or not name.endswith(suffix): return None
    return name[len(prefix):-len(suffix)]

def main(base):
    frames_files=glob.glob(os.path.join(base,"frames_*.npy"))
    actions_files=glob.glob(os.path.join(base,"actions_*.csv"))
    fm={tag_from(p,"frames_",".npy"):p for p in frames_files}
    am={tag_from(p,"actions_",".csv"):p for p in actions_files}
    tags=sorted(set(fm.keys())|set(am.keys()))
    bad_dir=os.path.join(base,"_bad_sets")
    os.makedirs(bad_dir,exist_ok=True)

    moved=0
    for t in tags:
        fp=fm.get(t); ap=am.get(t)
        if fp is None or ap is None:
            # 缺文件 => bad
            for p in [fp,ap]:
                if p and os.path.exists(p):
                    shutil.move(p, os.path.join(bad_dir, os.path.basename(p)))
                    moved+=1
            continue

        try:
            frames=np.load(fp)
            actions=read_actions(ap)
        except Exception:
            # 读不了 => bad
            shutil.move(fp, os.path.join(bad_dir, os.path.basename(fp))); moved+=1
            shutil.move(ap, os.path.join(bad_dir, os.path.basename(ap))); moved+=1
            continue

        n=frames.shape[0] if frames.ndim==3 else -1
        m=len(actions)

        # 规则：必须 n==m；非空；动作必须都在 VALID_ACTIONS
        non_empty = 0.0
        if frames.ndim==3 and n>0:
            non_empty = float(((frames.reshape(n,-1)!=0).any(axis=1)).mean())

        actions_ok = (m>0 and all(a in VALID_ACTIONS for a in actions))

        is_bad = (n!=m) or (non_empty<0.5) or (not actions_ok)
        if is_bad:
            shutil.move(fp, os.path.join(bad_dir, os.path.basename(fp))); moved+=1
            shutil.move(ap, os.path.join(bad_dir, os.path.basename(ap))); moved+=1

    print(f"Done. Moved {moved} files into {bad_dir}")

if __name__=="__main__":
    base = sys.argv[1] if len(sys.argv)>1 else "out_task4_surface"
    if not os.path.isdir(base):
        print("dir not found:", base); sys.exit(1)
    main(base)
