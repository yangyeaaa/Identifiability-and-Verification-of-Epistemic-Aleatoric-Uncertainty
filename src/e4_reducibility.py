"""E4 (Section 4.2): reducibility under nested, uniformly sampled training subsets of GEFCom2014.
Base networks (MC, LA-1, LA-2) and the free-head construction with three seeds; one five-member
ensemble per fraction. The full-data models are shared with E3."""
import numpy as np
from backbone import prepare
from decomp import base_net, ensemble, free_head, decompose, units
from models import ensemble_predict
from common import Logger, save_json, load_json

log = Logger("e4_reducibility")
P = load_json("e2_dropout_sweep")["selected_p"]
FRACS = (0.05, 0.10, 0.25, 0.50, 1.0); rows = []
for frac in FRACS:
    sc = prepare("gef", frac=frac); Xt = sc.X("test"); n = len(sc.trn)
    for s in (0, 1, 2):
        m, _ = base_net(sc, s, P)
        dec, meta = decompose(m, sc, ("MC", "LA-1", "LA-2"), ("test",))
        for k, d in dec.items():
            u = units(d["test"], sc); rows.append(dict(method=k, frac=frac, n=n, seed=s, epi=float(u["epi"].mean()), ale=float(u["ale"].mean()), R=u["R"]))
        f, _ = free_head(sc, s); u = units(f.predict(Xt), sc)
        rows.append(dict(method="FREE", frac=frac, n=n, seed=s, epi=float(u["epi"].mean()), ale=float(u["ale"].mean()), R=u["R"]))
    ms, _ = ensemble(sc, (10, 11, 12, 13, 14), P); u = units(ensemble_predict(ms, Xt), sc)
    rows.append(dict(method="ENS", frac=frac, n=n, seed=0, epi=float(u["epi"].mean()), ale=float(u["ale"].mean()), R=u["R"]))
    save_json("e4_reducibility", rows)
    for k in ("MC", "LA-1", "LA-2", "ENS", "FREE"):
        rr = [r for r in rows if r["method"] == k and r["frac"] == frac]
        g = lambda x: (np.mean([r[x] for r in rr]), np.std([r[x] for r in rr], ddof=1) if len(rr) > 1 else 0.0)
        e, a, R = g("epi"), g("ale"), g("R")
        log(f"{log.elapsed()} frac={frac:.2f} {k:>5}: epi={e[0]:7.1f}±{e[1]:5.1f} ale={a[0]:7.1f}±{a[1]:5.1f} R_epi={R[0]:.4f}±{R[1]:.4f}")
log("DONE")
