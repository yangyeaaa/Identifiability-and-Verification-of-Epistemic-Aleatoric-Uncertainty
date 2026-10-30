"""E2: dropout rate of the base network. p is a training hyperparameter of the heteroscedastic network;
it is selected by the validation CRPS of the deterministic network (decomposition-neutral). The sweep
also records how each decomposition's epistemic share responds to p and whether validation data can
tell the p values apart (scale identifiability from residuals)."""
import numpy as np
from backbone import prepare
from decomp import base_net, decompose, units
from metrics import gaussian_metrics, crps_gaussian
from common import Logger, save_json

log = Logger("e2_dropout_sweep")
sc = prepare("gef"); Yv, Yt = sc.Y("val"), sc.Y("test")
PS = (0.02, 0.05, 0.10, 0.20, 0.30); rows = []
import os
SEEDS = (0, 1) if os.environ.get("UQ_QUICK") else (0, 1, 2)
if os.environ.get("UQ_QUICK"): PS = (0.05, 0.1)
for p in PS:
    for s in SEEDS:
        m, _ = base_net(sc, s, p)
        dec, meta = decompose(m, sc, ("MVE", "MC", "LA-1", "LA-2"), ("val", "test"))
        row = dict(p=p, seed=s)
        for k, d in dec.items():
            uv, ut = units(d["val"], sc), units(d["test"], sc)
            row[k] = dict(val_CRPS=float(np.mean(crps_gaussian(Yv, uv["P"], uv["S"]))),
                          val_NLL=float(np.mean(0.5 * np.log(2 * np.pi * uv["S"] ** 2) + 0.5 * ((Yv - uv["P"]) / uv["S"]) ** 2)),
                          R_epi=ut["R"], epi=float(ut["epi"].mean()), **gaussian_metrics(Yt, ut["P"], ut["S"]))
        rows.append(row)
        log(f"p={p:.2f} s={s}: val CRPS MVE={row['MVE']['val_CRPS']:.2f} MC={row['MC']['val_CRPS']:.2f} | "
            f"test R_epi MC={row['MC']['R_epi']:.3f} LA-1={row['LA-1']['R_epi']:.4f} LA-2={row['LA-2']['R_epi']:.3f}")
summ = {}
for p in PS:
    rr = [r for r in rows if r["p"] == p]; g = lambda k, f: (float(np.mean([r[k][f] for r in rr])), float(np.std([r[k][f] for r in rr], ddof=1)))
    summ[p] = {k: {f: g(k, f) for f in ("val_CRPS", "val_NLL", "R_epi", "epi", "CRPS", "MAE")} for k in ("MVE", "MC", "LA-1", "LA-2")}
    log(f"p={p:.2f}: val CRPS(MVE)={summ[p]['MVE']['val_CRPS'][0]:.2f}±{summ[p]['MVE']['val_CRPS'][1]:.2f} | R_epi MC={summ[p]['MC']['R_epi'][0]:.3f} "
        f"LA-1={summ[p]['LA-1']['R_epi'][0]:.4f} LA-2={summ[p]['LA-2']['R_epi'][0]:.3f}")
best = min(PS, key=lambda p: summ[p]["MVE"]["val_CRPS"][0])
log(f"selected p (lowest mean validation CRPS of the deterministic network) = {best}")
save_json("e2_dropout_sweep", dict(rows=rows, summary={str(k): v for k, v in summ.items()}, selected_p=best))
log("DONE")
