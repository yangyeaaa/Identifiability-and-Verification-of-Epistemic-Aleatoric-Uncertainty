import os, json, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.environ.get("UQ_OUT", os.path.join(HERE, "..")))
EPOCHS = int(os.environ.get("UQ_EPOCHS", "50"))
RES = os.path.join(OUT, "results"); FIG = os.path.join(OUT, "figures"); CKPT = os.path.join(OUT, "checkpoints")
for d in (RES, FIG, CKPT):
    os.makedirs(d, exist_ok=True)


class Logger:
    def __init__(self, name):
        self.path = os.path.join(RES, name + ".txt")
        open(self.path, "w").close()
        self.t0 = time.time()
    def __call__(self, s=""):
        print(s, flush=True)
        with open(self.path, "a") as f:
            f.write(s + "\n")
    def elapsed(self):
        return f"[{time.time() - self.t0:7.1f}s]"


def save_json(name, obj):
    def conv(o):
        if isinstance(o, (np.floating, np.integer)): return o.item()
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, dict): return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [conv(v) for v in o]
        return o
    with open(os.path.join(RES, name + ".json"), "w") as f:
        json.dump(conv(obj), f, indent=1)


def load_json(name):
    with open(os.path.join(RES, name + ".json")) as f:
        return json.load(f)


def ck(name):
    return os.path.join(CKPT, name + ".pth")


def fmt(d, keys=("MAE", "MAPE", "CRPS", "ECE", "PICP90")):
    parts = []
    for k in keys:
        if k in d:
            parts.append(f"{k}={d[k]:.2f}" if k in ("MAE", "CRPS") else f"{k}={d[k]:.3f}")
    return " ".join(parts)
