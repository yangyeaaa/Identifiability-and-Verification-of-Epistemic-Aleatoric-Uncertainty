"""Evaluation metrics and the three post-hoc calibration procedures."""
import numpy as np
from scipy.stats import norm

LEVELS = np.linspace(0.5, 0.95, 10)


def crps_gaussian(y, mu, sig):
    sig = np.maximum(sig, 1e-9); z = (y - mu) / sig
    return sig * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))


def coverage_curve(Y, P, S, levels=LEVELS):
    return np.array([np.mean(np.abs(Y - P) / S <= norm.ppf((1 + c) / 2)) for c in levels])


def ece_from_curve(obs, levels=LEVELS):
    return float(np.mean(np.abs(levels - obs)))


def gaussian_metrics(Y, P, S):
    obs = coverage_curve(Y, P, S)
    return dict(MAE=float(np.mean(np.abs(Y - P))),
                MAPE=float(np.mean(np.abs((Y - P) / (np.abs(Y) + 1e-8))) * 100),
                CRPS=float(np.mean(crps_gaussian(Y, P, S))),
                ECE=ece_from_curve(obs), PICP90=float(obs[-2]))   # LEVELS[-2] == 0.90


def to_units(r, sc):
    """Convert a predict() dict (normalized) to original units."""
    return dict(P=r["mu"] * sc.scale + sc.mean, S=r["tot"] * sc.scale,
                ale=r["ale"] * sc.scale, epi=r["epi"] * sc.scale, R_epi=r["R_epi"], r_epi=r["r_epi"])


# ---------------------------------------------------------------- calibration
def temperature(Yv, Pv, Sv):
    return float(np.sqrt(np.mean(((Yv - Pv) / Sv) ** 2)))


def conformal_q(scores, c):
    n = len(scores); k = min(n, int(np.ceil((n + 1) * c)))
    return np.sort(scores)[k - 1]


def split_conformal(Yv, Pv, Sv, Yt, Pt, St, levels=LEVELS):
    sv = np.abs(Yv - Pv) / Sv; st = np.abs(Yt - Pt) / St
    qs = np.array([conformal_q(sv, c) for c in levels])
    obs = np.array([np.mean(st <= q) for q in qs])
    return dict(curve=obs, ECE=ece_from_curve(obs, levels), PICP90=float(obs[list(levels).index(0.9)] if 0.9 in levels else obs[-2]), q=qs)


def rolling_conformal(Yv, Pv, Sv, Yt, Pt, St, W=24 * 90, lag=24, levels=LEVELS):
    """Causal rolling-window conformal: the multiplier for test hour t is the conformal quantile
    of the normalized residuals observed in the W hours ending `lag` hours before t.  The
    validation residuals seed the window, so the procedure is defined from the first test hour."""
    s_all = np.concatenate([np.abs(Yv - Pv) / Sv, np.abs(Yt - Pt) / St])
    n0 = len(Yv); st = np.abs(Yt - Pt) / St
    hit = np.zeros((len(levels), len(Yt)), dtype=bool)
    for t in range(len(Yt)):
        end = n0 + t - lag; start = max(0, end - W)
        win = s_all[start:end]
        for j, c in enumerate(levels):
            hit[j, t] = st[t] <= conformal_q(win, c)
    obs = hit.mean(1)
    return dict(curve=obs, ECE=ece_from_curve(obs, levels), PICP90=float(obs[-2]), hit90=hit[-2])


def rolling_scale_90(Yv, Pv, Sv, Yt, Pt, St, W=24 * 90, lag=24, c=0.90):
    """Per-hour multiplier of the 90% interval under the rolling procedure (for plots)."""
    s_all = np.concatenate([np.abs(Yv - Pv) / Sv, np.abs(Yt - Pt) / St]); n0 = len(Yv)
    return np.array([conformal_q(s_all[max(0, n0 + t - lag - W):n0 + t - lag], c) for t in range(len(Yt))])


# ---------------------------------------------------------------- reserve policies (Section 6.1)
def reserve_costs(Yv, Pv, Sv, Yt, Pt, St, ratios=(2, 5, 10, 20)):
    """Mean operating cost (c_o = 1) of three scheduling policies; see Section 6.1.
    deterministic: S = P ; fixed: S = P + global tau*-quantile of validation residuals ;
    adaptive: S = P + q_tau* * sigma_total with q_tau* the tau*-quantile of validation normalized residuals."""
    out = {}
    rv = Yv - Pv; zv = rv / Sv
    for cu in ratios:
        tau = cu / (cu + 1.0)
        cost = lambda S: np.mean(cu * np.maximum(Yt - S, 0) + np.maximum(S - Yt, 0))
        out[cu] = dict(deterministic=cost(Pt),
                       fixed=cost(Pt + np.quantile(rv, tau)),
                       adaptive=cost(Pt + np.quantile(zv, tau) * St))
    return out
