"""Decompositions under study, trained or loaded resume-safely, and their evaluation.

  MVE   : heteroscedastic network evaluated deterministically (aleatoric only)
  MC    : MC dropout, all dropout active, M = 40 passes
  LA-1  : last-layer Laplace, evidence-fitted prior precision
  LA-2  : two-layer Laplace (hidden + mean layer), evidence-fitted prior precision
  ENS   : deep ensemble of K = 5 heteroscedastic networks
  FREE  : likelihood-based construction with free epistemic/aleatoric outputs
  CMC   : classical MC dropout (point network + homoscedastic noise)
  QR    : quantile regression
  PERS  : day-ahead persistence (same hour, previous day) with residual quantiles from validation
"""
import time, numpy as np, torch
from backbone import setup_seed, count_params
from models import HeteroMCDropout, FreeHead, ClassicMCDropout, QuantileNet, ensemble_predict, LaplaceDecomposition, QGRID
from metrics import gaussian_metrics, LEVELS
from common import ck

M_PASSES = 40


def tag(sc, extra=""):
    n = len(sc.idx)
    if n == sc.n_total: kind = "full"
    elif sc.idx[-1] == sc.n_total - 1 and np.all(np.diff(sc.idx) == 1): kind = f"recent{n}"
    else: kind = f"unif{n}"
    return f"{sc.name}_{kind}{extra}"


def base_net(sc, seed, p, tagx=""):
    setup_seed(seed); m = HeteroMCDropout(sc.sl, sc.nv, dropout=p)
    info = m.train_model(*sc.loaders(), save=ck(f"base_{tag(sc, tagx)}_p{int(round(p*100)):02d}_s{seed}"))
    return m, info


def ensemble(sc, members_seeds, p, tagx=""):
    ms, tt = [], 0.0
    for s in members_seeds:
        m, info = base_net(sc, s, p, tagx); ms.append(m); tt += info["train_time"]
    return ms, tt


def free_head(sc, seed, init_ale=0.0, init_epi=-1.0, tagx=""):
    setup_seed(seed); m = FreeHead(sc.sl, sc.nv, init_ale=init_ale, init_epi=init_epi)
    info = m.train_model(*sc.loaders(), save=ck(f"free_{tag(sc, tagx)}_a{init_ale:+.0f}e{init_epi:+.0f}_s{seed}"))
    return m, info


def decompose(m, sc, which=("MVE", "MC", "LA-1", "LA-2"), splits=("test",), timing=False):
    """Return {method: {split: predict-dict (normalized units)}} for one trained base network."""
    out = {k: {} for k in which}; t = {}
    las = {}
    for k in which:
        if k.startswith("LA-"):
            t0 = time.time(); las[k] = LaplaceDecomposition(int(k[-1])).fit(m, sc.X("train")); t[k + "_fit"] = time.time() - t0
    for sp in splits:
        X = sc.X(sp)
        for k in which:
            t0 = time.time()
            if k == "MVE": r = m.predict(X, mc_dropout=False)
            elif k == "MC": r = m.predict(X, M=M_PASSES, seed=123)
            else: r = las[k].predict(m, X)
            out[k][sp] = r
            if sp == "test": t[k + "_infer"] = time.time() - t0
    meta = {k: dict(alpha=v.alpha, gamma=v.gamma, n_params=v.n_params) for k, v in las.items()}
    return (out, meta, t) if timing else (out, meta)


def units(r, sc):
    return dict(P=r["mu"] * sc.scale + sc.mean, S=r["tot"] * sc.scale, ale=r["ale"] * sc.scale,
                epi=r["epi"] * sc.scale, R=float(r["R_epi"]))


def qr_metrics(Q, Y, sc):
    Q = np.sort(Q, 1) * sc.scale + sc.mean
    med = Q[:, np.argmin(np.abs(QGRID - 0.5))]
    e = Y[:, None] - Q; pb = np.maximum(QGRID[None] * e, (QGRID[None] - 1) * e)
    ip = lambda tau: np.array([np.interp(tau, QGRID, Q[i]) for i in range(len(Q))])
    obs = np.array([np.mean((Y >= ip((1 - c) / 2)) & (Y <= ip((1 + c) / 2))) for c in LEVELS])
    return dict(MAE=float(np.mean(np.abs(Y - med))), MAPE=float(np.mean(np.abs((Y - med) / Y)) * 100),
                CRPS=float(2 * pb.mean()), ECE=float(np.mean(np.abs(LEVELS - obs))), PICP90=float(obs[-2]))


def persistence(sc, load_col):
    """Same hour on the previous day (row 6 of the raw inputs), Gaussian with validation residual scale."""
    from backbone import load_raw
    raw = load_raw(sc.name)
    Pv = raw["val"][0][:, 6, load_col]; Pt = raw["test"][0][:, 6, load_col]
    Yv, Yt = sc.Y("val"), sc.Y("test")
    s = np.std(Yv - Pv)
    return gaussian_metrics(Yt, Pt, np.full_like(Pt, s))
