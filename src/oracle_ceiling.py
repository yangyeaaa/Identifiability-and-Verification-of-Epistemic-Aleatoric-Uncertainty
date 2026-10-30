"""Ceiling of the dominant-source decision for an ORACLE that knows the expected error and noise levels
(Proposition 6; supplementary Table S13).

At hour t the estimation error of the deployed network is e_t = s_t z_t, s_t^2 = E[e_t^2 | x_t], noise variance
sigma_t^2, r_t = s_t^2 / sigma_t^2. The hour label is 1{e_t^2 > sigma_t^2}; its conditional probability is
p_t = 2 (1 - Phi(r_t^{-1/2})). Proposition 6(i): the best balanced accuracy of any rule is
BA* = 1/2 + E|p - pi| / (4 pi (1 - pi)), pi = E[p]; the split's own rule r_t > 1 (sigma_epi^2 > sigma_ale^2) is shown
for comparison. Block labels over H hours (Proposition 6(ii)) are simulated with r constant within a block and
z_t an AR(1) Gaussian process with coefficient PHI, so that the lag-1 autocorrelation of z_t^2 is PHI^2 = 0.85,
within the range measured on the squared estimation errors (E14: 0.83-0.96); the block BA* is the best
balanced accuracy over thresholds on r (Bayes-optimal because the block label probability is monotone in r).
Usage: python oracle_ceiling.py OUTDIR   (writes OUTDIR/tables/oracle.tex and results/oracle_ceiling.json)"""
import os, sys, json
import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_curve
from common import RES

OUTDIR = sys.argv[1] if len(sys.argv) > 1 else ".."
TAB = os.path.join(OUTDIR, "tables"); os.makedirs(TAB, exist_ok=True)
rng = np.random.default_rng(2026)
PHI = float(np.sqrt(0.85)); N_HOURS, N_BLOCKS = 1_000_000, 20_000


def ba_rule(t, pred):
    return float(0.5 * (pred[t].mean() + (~pred[~t]).mean()))


def ba_best(t, score):
    fpr, tpr, _ = roc_curve(t, score)
    return float(np.max(0.5 * (tpr + 1 - fpr)))


def hourly(r_med, lsd):
    r = r_med * np.exp(lsd * rng.standard_normal(N_HOURS)); z2 = rng.standard_normal(N_HOURS) ** 2; t = r * z2 > 1
    p = 2 * (1 - norm.cdf(1 / np.sqrt(r))); pi = p.mean()
    return dict(share=float(r.mean() / (r.mean() + 1)), epi_dom=float(t.mean()), ba_rule=ba_rule(t, r > 1),
                ba_star=float(0.5 + np.mean(np.abs(p - pi)) / (4 * pi * (1 - pi))), ba_star_sim=ba_best(t, r))


def block(r_med, lsd, H):
    r = r_med * np.exp(lsd * rng.standard_normal(N_BLOCKS))
    z = np.empty((N_BLOCKS, H)); z[:, 0] = rng.standard_normal(N_BLOCKS)
    for h in range(1, H):
        z[:, h] = PHI * z[:, h - 1] + np.sqrt(1 - PHI ** 2) * rng.standard_normal(N_BLOCKS)
    t = r * (z ** 2).mean(1) > 1
    return ba_best(t, r) if 0 < t.mean() < 1 else float("nan")


def heff(H):
    k = np.arange(1, H)
    return float(H / (1 + 2 * np.sum((1 - k / H) * PHI ** (2 * k)))) if H > 1 else 1.0


rows = []
for lsd in (1.0, 2.0):
    for r_med in (0.1, 1.0, 10.0):
        rows.append(dict(r_med=r_med, lsd=lsd, **hourly(r_med, lsd), ba_day=block(r_med, lsd, 24),
                         ba_week=block(r_med, lsd, 168), ba_month=block(r_med, lsd, 720)))
meta = dict(phi=PHI, heff_day=heff(24), heff_week=heff(168), heff_month=heff(720))
json.dump(dict(rows=rows, meta=meta), open(os.path.join(RES, "oracle_ceiling.json"), "w"), indent=1)
lines = [f"{r['r_med']:g} & {r['lsd']:g} & {r['share']:.2f} & {100*r['epi_dom']:.0f} & {r['ba_rule']:.2f} & {r['ba_star']:.2f} & "
         f"{r['ba_day']:.2f} & {r['ba_week']:.2f} & {r['ba_month']:.2f} \\\\" for r in rows]
open(os.path.join(TAB, "oracle.tex"), "w").write("\n".join(lines))
print(json.dumps(meta))
for r in rows:
    print({k: round(v, 3) if isinstance(v, float) else v for k, v in r.items()})
