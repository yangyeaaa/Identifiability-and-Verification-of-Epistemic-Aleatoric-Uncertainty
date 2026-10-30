"""E3 (Sections 4.1 and 5): all decompositions and baselines on one dataset, five seeds each.
Usage: python e3_main.py gef|covid
Saves per-seed test/validation predictions (MW) for the probes in E5."""
import sys, time, numpy as np
from backbone import prepare, COLS, count_params
from decomp import base_net, ensemble, free_head, decompose, units, qr_metrics, persistence
from models import ClassicMCDropout, QuantileNet, ensemble_predict
from backbone import setup_seed
from metrics import gaussian_metrics, split_conformal, rolling_conformal
from common import Logger, save_json, load_json, ck, fmt

name = sys.argv[1]
log = Logger(f"e3_main_{name}")
P = load_json("e2_dropout_sweep")["selected_p"]
sc = prepare(name); Yv, Yt = sc.Y("val"), sc.Y("test")
SEEDS = (0, 1, 2, 3, 4); ENS_SEEDS = ((10, 11, 12, 13, 14), (20, 21, 22, 23, 24), (30, 31, 32, 33, 34))
import os
if os.environ.get("UQ_QUICK"): SEEDS, ENS_SEEDS = (0, 1), ((10, 11), (20, 21))
R = {k: [] for k in ("MVE", "MC", "LA-1", "LA-2", "ENS", "FREE", "CMC", "QR")}
preds = {}


def record(key, seed, rv, rt, extra=None):
    uv, ut = units(rv, sc), units(rt, sc)
    met = gaussian_metrics(Yt, ut["P"], ut["S"])
    sp = split_conformal(Yv, uv["P"], uv["S"], Yt, ut["P"], ut["S"])
    ro = rolling_conformal(Yv, uv["P"], uv["S"], Yt, ut["P"], ut["S"], W=24 * 90, lag=24)
    row = dict(seed=seed, R_epi=ut["R"], epi=float(ut["epi"].mean()), ale=float(ut["ale"].mean()), **met,
               ECE_split=sp["ECE"], PICP90_split=sp["PICP90"], ECE_roll=ro["ECE"], PICP90_roll=ro["PICP90"], **(extra or {}))
    R[key].append(row); preds[f"{key}_s{seed}"] = dict(test=ut, val=uv)
    log(f"{log.elapsed()} {key:>5} s={seed}: R_epi={ut['R']:.3f} epi={ut['epi'].mean():8.1f} | {fmt(met)} | "
        f"split ECE={sp['ECE']:.3f} roll ECE={ro['ECE']:.3f} PICP90 roll={ro['PICP90']:.3f}")


for s in SEEDS:
    m, info = base_net(sc, s, P)
    dec, meta, t = decompose(m, sc, ("MVE", "MC", "LA-1", "LA-2"), ("val", "test"), timing=True)
    for k in dec:
        record(k, s, dec[k]["val"], dec[k]["test"], dict(train_time=info["train_time"], infer_time=t[k + "_infer"],
               fit_time=t.get(k + "_fit", 0.0), params=count_params(m), **meta.get(k, {})))
for j, seeds in enumerate(ENS_SEEDS):
    ms, tt = ensemble(sc, seeds, P)
    t0 = time.time(); rt = ensemble_predict(ms, sc.X("test")); ti = time.time() - t0
    record("ENS", j, ensemble_predict(ms, sc.X("val")), rt, dict(train_time=tt, infer_time=ti, params=5 * count_params(ms[0])))
for s in SEEDS:
    f, info = free_head(sc, s)
    record("FREE", s, f.predict(sc.X("val")), f.predict(sc.X("test")), dict(train_time=info["train_time"]))
    setup_seed(s); c = ClassicMCDropout(sc.sl, sc.nv, dropout=P)
    info = c.train_model(*sc.loaders(), save=ck(f"cmc_{name}_p{int(round(P*100)):02d}_s{s}")); c.fit_noise(sc.X("val"), sc.vdln)
    record("CMC", s, c.predict(sc.X("val"), seed=123), c.predict(sc.X("test"), seed=123), dict(train_time=info["train_time"]))
    setup_seed(s); q = QuantileNet(sc.sl, sc.nv, dropout=P)
    info = q.train_model(*sc.loaders(), save=ck(f"qr_{name}_p{int(round(P*100)):02d}_s{s}"))
    row = dict(seed=s, train_time=info["train_time"], **qr_metrics(q.predict(sc.X("test")), Yt, sc)); R["QR"].append(row)
    log(f"{log.elapsed()}    QR s={s}: {fmt(row)}")
R["PERS"] = [persistence(sc, COLS[name]["load"])]
log(f"PERS: {fmt(R['PERS'][0])}")

log("\nmean ± sd over seeds (ensembles: over 3 independent ensembles)")
summary = {}
for k, rows in R.items():
    keys = [x for x in rows[0] if isinstance(rows[0][x], float) and x != "seed"]
    summary[k] = {x: (float(np.mean([r[x] for r in rows])), float(np.std([r[x] for r in rows], ddof=1)) if len(rows) > 1 else 0.0) for x in keys}
    s_ = summary[k]
    log(f"  {k:>5}: " + " ".join(f"{x}={s_[x][0]:.3f}±{s_[x][1]:.3f}" for x in ("MAE", "CRPS", "ECE", "PICP90", "R_epi", "ECE_roll", "infer_time") if x in s_))
save_json(f"e3_main_{name}", dict(rows=R, summary=summary, p=P))
np.savez_compressed(ck(f"e3_preds_{name}").replace(".pth", ".npz"),
                    **{f"{k}__{sp}__{q}": v[sp][q] for k, v in preds.items() for sp in ("test", "val") for q in ("P", "S", "ale", "epi")})
log("DONE")
