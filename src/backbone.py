"""
Shared backbone, data handling and training loop.

Forecasting set-up (both datasets): each sample holds the same clock hour on the previous seven
days (load, temperature, calendar) and the target is the load at that hour on the next day, i.e. a
24-hour-ahead forecast.  With FUTURE=True (default) an eighth step is appended that carries the
target day's calendar variables and observed temperature (ex-post forecasting, a perfect-weather-
forecast stand-in); its load entry is a placeholder equal to the previous day's load, so no target
information enters the inputs.  The last 24 samples of each split, whose target-day row is not
available inside the split, are dropped.
"""
import os, random, time
import numpy as np
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

device = torch.device("cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))

ROOT = os.environ.get("UQ_DATA_ROOT", os.path.join(os.path.dirname(__file__), "..", "data"))
DATASETS = {"gef": "GEFCom2014", "covid": "postCOVID"}
# column layout of the raw arrays (N, 7, 6)
COLS = {"gef":   dict(load=1, temp=2, names=["hour", "load", "temperature", "month", "day", "weekday"]),
        "covid": dict(load=0, temp=1, names=["load", "temperature", "month", "day", "hour", "weekday"])}
FUTURE = os.environ.get("UQ_FUTURE", "1") == "1"


# ------------------------------------------------------------------ utils
def setup_seed(seed: int):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


# ------------------------------------------------------------------ data
class TimeSeriesDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data; self.labels = labels
    def __getitem__(self, i):
        return torch.tensor(self.data[i], dtype=torch.float32), torch.tensor(self.labels[i], dtype=torch.float32)
    def __len__(self):
        return len(self.data)


def add_target_day(X, y, load_col):
    """Append the target-day row (calendar + observed temperature) as an eighth step.
    Row 6 of sample i+24 is the target day of sample i; its load (= y[i]) is replaced by the
    previous-day load X[i, 6, load_col]."""
    assert np.allclose(X[24:, 6, load_col], y[:-24]), "unexpected sample layout"
    fut = X[24:, 6, :].copy(); fut[:, load_col] = X[:-24, 6, load_col]
    return np.concatenate([X[:-24], fut[:, None, :]], axis=1), y[:-24]


def load_raw(name, future=None):
    future = FUTURE if future is None else future
    d = os.path.join(ROOT, DATASETS[name])
    out = {}
    for s in ("train", "val", "test"):
        X, y = np.load(f"{d}/{s}_data.npy"), np.load(f"{d}/{s}_label.npy")
        if future:
            X, y = add_target_day(X, y, COLS[name]["load"])
        out[s] = (X, y)
    return out


class Scaled:
    """Standardize inputs and labels with statistics of the (possibly sub-sampled) training set."""
    def __init__(self, tr, trl, vd, vdl, te, tel):
        self.sl, self.nv = tr.shape[1], tr.shape[2]
        self.sdd = StandardScaler().fit(tr.reshape(-1, self.nv))
        self.sdl = StandardScaler().fit(trl.reshape(-1, 1))
        f = lambda a: self.sdd.transform(a.reshape(-1, self.nv)).reshape(-1, self.sl, self.nv).astype(np.float32)
        g = lambda a: self.sdl.transform(a.reshape(-1, 1)).flatten().astype(np.float32)
        self.trn, self.trln = f(tr), g(trl)
        self.vdn, self.vdln = f(vd), g(vdl)
        self.ten, self.teln = f(te), g(tel)
        self.scale = float(self.sdl.scale_[0]); self.mean = float(self.sdl.mean_[0])
    def loaders(self, bs=256):
        tl = DataLoader(TimeSeriesDataset(self.trn, self.trln), batch_size=bs, shuffle=True)
        vl = DataLoader(TimeSeriesDataset(self.vdn, self.vdln), batch_size=bs, shuffle=False)
        return tl, vl
    def X(self, split):
        return torch.tensor({"train": self.trn, "val": self.vdn, "test": self.ten}[split])
    def Y(self, split):   # labels in original units
        return {"train": self.trln, "val": self.vdln, "test": self.teln}[split] * self.scale + self.mean


def subset_index(n_total, frac=1.0, perm_seed=123, recent=False, n=None):
    n = int(n_total * frac) if n is None else n
    if n >= n_total:
        return np.arange(n_total)
    if recent:
        return np.arange(n_total - n, n_total)
    return np.sort(np.random.RandomState(perm_seed).permutation(n_total)[:n])


def prepare(name, frac=1.0, perm_seed=123, recent=False, n=None, labels=None, index=None):
    """Return a Scaled object. frac<1 keeps a nested uniform random subset of the training set
    (one fixed permutation, so smaller subsets are contained in larger ones); recent=True keeps the
    most recent part instead. labels (dict split->array) replaces the observed labels (semi-synthetic)."""
    raw = load_raw(name)
    if labels is not None:
        raw = {k: (raw[k][0], labels[k]) for k in raw}
    tr, trl = raw["train"]; vd, vdl = raw["val"]; te, tel = raw["test"]
    idx = subset_index(len(tr), frac, perm_seed, recent, n) if index is None else np.asarray(index)
    sc = Scaled(tr[idx], trl[idx], vd, vdl, te, tel)
    sc.raw_train, sc.raw_test, sc.name, sc.idx, sc.n_total = tr[idx], te, name, idx, len(tr)
    return sc


# ------------------------------------------------------------------ encoder
class iTransformerEncoder(nn.Module):
    def __init__(self, seq_len, n_variates, d_model=64, n_heads=4, n_layers=2, dropout=0.1):
        super().__init__()
        self.variate_embedding = nn.Linear(seq_len, d_model)
        self.variate_pos = nn.Parameter(torch.zeros(1, n_variates, d_model))
        nn.init.trunc_normal_(self.variate_pos, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                                           dropout=dropout, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.temporal_ffn = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                                          nn.Linear(d_model * 2, d_model))

    def forward(self, x):
        x = x.transpose(1, 2)
        tokens = self.variate_embedding(x) + self.variate_pos
        tokens = self.norm(self.encoder(tokens))
        tokens = tokens + self.temporal_ffn(tokens)
        return tokens.mean(1)          # pool over variate tokens (dataset-independent readout)


# ------------------------------------------------------------------ generic training loop
def fit(model, loss_fn, tl, vl, epochs=50, patience=12, lr=5e-4, wd=1e-5, save="best.pth", val_fn=None):
    """Early stopping on validation loss (val_fn defaults to loss_fn). Resume-safe: if `save` and its
    metadata exist the weights are loaded instead of retrained."""
    import json
    meta = save.replace(".pth", ".json")
    if os.path.exists(save) and os.path.exists(meta):
        model.load_state_dict(torch.load(save, weights_only=True)); model.to(device)
        return json.load(open(meta))
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=5, factor=0.5, min_lr=1e-6)
    val_fn = val_fn or loss_fn
    best, pc, t0 = float("inf"), 0, time.time()
    for ep in range(epochs):
        model.train()
        for x, y in tl:
            loss = loss_fn(model, x.to(device), y.to(device))
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        model.eval(); vs = []
        with torch.no_grad():
            for x, y in vl:
                vs.append(val_fn(model, x.to(device), y.to(device)).item())
        v = float(np.mean(vs)); sch.step(v)
        if v < best:
            best, pc = v, 0; torch.save(model.state_dict(), save)
        else:
            pc += 1
            if pc >= patience: break
    model.load_state_dict(torch.load(save, weights_only=True))
    info = dict(best_val=best, epochs=ep + 1, train_time=time.time() - t0)
    json.dump(info, open(meta, "w"))
    return info
