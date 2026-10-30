"""E5 (Section 4.3): support probe on the full-data models of E3.
Distance of each test input to its ten nearest training inputs, measured on the continuous inputs only
(load and temperature over all eight steps, standardized with training statistics) - calendar codes are
excluded. Statistics per decomposition and seed: Spearman rho(sigma_epi, d), rho(sigma_epi, |x|),
within-load-decile rho (averaged), partial rho given |x|, far/near ratio (top vs bottom distance decile).
95 % block-bootstrap intervals (blocks of 168 consecutive test hours, i.e. one week) for rho and partial rho."""
import sys, numpy as np
from scipy.stats import spearmanr, rankdata
from sklearn.neighbors import NearestNeighbors
from backbone import prepare, COLS
from common import Logger, save_json, ck

log = Logger("e5_support")
METHODS = ("MC", "LA-1", "LA-2", "ENS", "FREE")
rng = np.random.RandomState(0)


def resid(u, v):
    return u - np.polyval(np.polyfit(v, u, 1), v)


def stats(epi, d, xn, dec):
    ra, rb, rc = rankdata(epi), rankdata(d), rankdata(xn)
    dq = np.quantile(d, [0.1, 0.9])
    return dict(rho=spearmanr(epi, d)[0], rho_x=spearmanr(epi, xn)[0],
                within=float(np.mean([spearmanr(epi[dec == b], d[dec == b])[0] for b in range(10)])),
                partial=float(np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1]),
                far_near=float(epi[d >= dq[1]].mean() / epi[d <= dq[0]].mean()))


def boot(epi, d, xn, B=300, L=168):
    nb = len(epi) // L; blocks = np.arange(nb * L).reshape(nb, L); out = []
    for _ in range(B):
        idx = blocks[rng.randint(0, nb, nb)].ravel(); ra, rb, rc = rankdata(epi[idx]), rankdata(d[idx]), rankdata(xn[idx])
        out.append((spearmanr(epi[idx], d[idx])[0], np.corrcoef(resid(ra, rc), resid(rb, rc))[0, 1]))
    out = np.array(out); return np.percentile(out, [2.5, 97.5], axis=0).T.tolist()


OUT = {}
for name in ("gef", "covid"):
    sc = prepare(name); c = COLS[name]; cont = [c["load"], c["temp"]]
    f = lambda Xn: Xn[:, :, cont].reshape(len(Xn), -1)
    Xtr, Xte = f(sc.trn), f(sc.ten)
    d = NearestNeighbors(n_neighbors=10).fit(Xtr).kneighbors(Xte)[0].mean(1); xn = np.linalg.norm(Xte, axis=1)
    Y = sc.Y("test"); q = np.quantile(Y, np.linspace(0, 1, 11)); dec = np.clip(np.searchsorted(q, Y, side="right") - 1, 0, 9)
    pr = np.load(ck(f"e3_preds_{name}").replace(".pth", ".npz"))
    res = {}
    for k in METHODS:
        seeds = sorted({key.split("__")[0] for key in pr.files if key.startswith(k + "_s")})
        per = []
        for sk in seeds:
            epi = pr[f"{sk}__test__epi"]; st = stats(epi, d, xn, dec); st["ci_rho"], st["ci_partial"] = boot(epi, d, xn); per.append(st)
        agg = {x: (float(np.mean([p[x] for p in per])), float(np.std([p[x] for p in per], ddof=1)) if len(per) > 1 else 0.0)
               for x in ("rho", "rho_x", "within", "partial", "far_near")}
        res[k] = dict(per_seed=per, mean=agg)
        log(f"{name:>5} {k:>5}: rho(epi,d)={agg['rho'][0]:+.2f}±{agg['rho'][1]:.2f} rho(epi,|x|)={agg['rho_x'][0]:+.2f} "
            f"within={agg['within'][0]:+.2f}±{agg['within'][1]:.2f} partial={agg['partial'][0]:+.2f}±{agg['partial'][1]:.2f} "
            f"far/near={agg['far_near'][0]:.2f} | seed-0 CI rho [{per[0]['ci_rho'][0]:+.2f},{per[0]['ci_rho'][1]:+.2f}] "
            f"partial [{per[0]['ci_partial'][0]:+.2f},{per[0]['ci_partial'][1]:+.2f}]")
    # decile curves for figures (seed 0 / first ensemble)
    dd = np.clip(np.searchsorted(np.quantile(d, np.linspace(0, 1, 11)), d, side="right") - 1, 0, 9)
    res["curves"] = {k: dict(load_decile=[float(pr[f"{k}_s0__test__epi"][dec == b].mean()) for b in range(10)],
                             dist_decile=[float(pr[f"{k}_s0__test__epi"][dd == b].mean()) for b in range(10)],
                             dist_decile_ale=[float(pr[f"{k}_s0__test__ale"][dd == b].mean()) for b in range(10)])
                     for k in METHODS}
    res["load_centers"] = [float(0.5 * (q[b] + q[b + 1])) for b in range(10)]
    OUT[name] = res
save_json("e5_support", OUT)
log("DONE")
