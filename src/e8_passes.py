"""E8 (Section 5.5): number of MC-dropout passes M, canonical GEFCom2014 network (seed 1)."""
import numpy as np
from backbone import prepare
from decomp import base_net, units
from metrics import gaussian_metrics
from common import Logger, save_json, load_json

log = Logger("e8_passes")
P = load_json("e2_dropout_sweep")["selected_p"]
sc = prepare("gef"); m, _ = base_net(sc, 1, P); Yt = sc.Y("test"); rows = []
for M in (2, 5, 10, 20, 40, 80):
    u = units(m.predict(sc.X("test"), M=M, seed=123), sc); met = gaussian_metrics(Yt, u["P"], u["S"])
    rows.append(dict(M=M, R=u["R"], epi=float(u["epi"].mean()), **met))
    log(f"M={M:3d}: R_epi={u['R']:.4f} epi={u['epi'].mean():6.2f} MAE={met['MAE']:.2f} CRPS={met['CRPS']:.2f}")
save_json("e8_passes", rows)
log("DONE")
