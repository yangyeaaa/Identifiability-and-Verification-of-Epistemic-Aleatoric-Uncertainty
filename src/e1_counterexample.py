"""E1 (Section 2.3, Table 1): the likelihood-based construction on GEFCom2014; only the initial values
of the two dispersion scales are changed (seed, data, architecture, optimizer, schedule fixed)."""
import numpy as np
from backbone import prepare
from decomp import free_head, units
from metrics import gaussian_metrics
from common import Logger, save_json, fmt

log = Logger("e1_counterexample")
sc = prepare("gef"); Yt = sc.Y("test"); rows = []
for name, ia, ie in (("default", 0.0, -1.0), ("swapped", -1.0, 0.0), ("epistemic-heavy", -2.0, 0.5)):
    m, info = free_head(sc, 1, ia, ie)
    r = m.predict(sc.X("test")); u = units(r, sc); met = gaussian_metrics(Yt, u["P"], u["S"])
    rows.append(dict(name=name, init_ale=ia, init_epi=ie, R_epi=u["R"], sigma_tot_norm=float(r["tot"].mean()), **met))
    log(f"{name:>16}: R_epi={u['R']:.3f} sigma_tot(norm)={r['tot'].mean():.4f} | {fmt(met)}")
save_json("e1_counterexample", rows)
log("DONE")
