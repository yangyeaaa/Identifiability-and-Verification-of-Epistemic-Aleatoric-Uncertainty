"""E6 (Section 4.3, real data): genuine out-of-support tests on GEFCom2014.
  recent 1 / 2 / 3 months : training restricted to the most recent months (seasonal extrapolation)
  temperature hold-out     : training samples with any input temperature outside the central 90 % of
                             the training temperatures are removed; test hours with extreme
                             temperatures are then outside the training support
For every condition: base networks with three seeds (MC, LA-1, LA-2) and one five-member ensemble.
Test inputs are 'out of support' when any input temperature lies outside the training range.
Reports R_epi, epistemic out/in ratio, Spearman(sigma_epi, temperature excess), and 90 % coverage of
the total versus the aleatoric-only interval (raw)."""
import numpy as np
from scipy.stats import spearmanr
from backbone import prepare, load_raw, COLS
from decomp import base_net, ensemble, decompose, units
from models import ensemble_predict
from metrics import gaussian_metrics
from common import Logger, save_json, load_json

log = Logger("e6_ood")
P = load_json("e2_dropout_sweep")["selected_p"]
name = "gef"; TC = COLS[name]["temp"]
raw = load_raw(name); tr_all = raw["train"][0]; N = len(tr_all)
Tall = tr_all[:, :, TC]; lo, hi = np.percentile(Tall, [5, 95]); keep = np.all((Tall >= lo) & (Tall <= hi), axis=1)
CONDS = [("recent 1 month", dict(n=720, recent=True)), ("recent 2 months", dict(n=1440, recent=True)),
         ("recent 3 months", dict(n=2160, recent=True)), ("full year", dict()),
         ("temperature hold-out", dict(index=np.where(keep)[0]))]
rows = []
for cname, kw in CONDS:
    sc = prepare(name, **kw); Xt, Yt = sc.X("test"), sc.Y("test")
    T = sc.raw_train[:, :, TC]; Tt = sc.raw_test[:, :, TC]
    excess = np.maximum(0, np.maximum(Tt - T.max(), T.min() - Tt)).max(1); out = excess > 0
    tagx = "_thold" if cname.startswith("temperature") else ""
    cands = []
    for s in (0, 1, 2):
        m, _ = base_net(sc, s, P, tagx)
        dec, _ = decompose(m, sc, ("MC", "LA-1", "LA-2"), ("test",))
        cands += [(k, s, d["test"]) for k, d in dec.items()]
    ms, _ = ensemble(sc, (10, 11, 12, 13, 14), P, tagx); cands.append(("ENS", 0, ensemble_predict(ms, Xt)))
    for k, s, r in cands:
        u = units(r, sc); tot = gaussian_metrics(Yt, u["P"], u["S"]); ale = gaussian_metrics(Yt, u["P"], u["ale"])
        row = dict(cond=cname, method=k, seed=s, n=len(sc.trn), frac_out=float(out.mean()), R=u["R"], epi=float(u["epi"].mean()),
                   ale=float(u["ale"].mean()), MAE=tot["MAE"], PICP90_tot=tot["PICP90"], PICP90_ale=ale["PICP90"],
                   CRPS_tot=tot["CRPS"], CRPS_ale=ale["CRPS"])
        if out.any() and (~out).any():
            row.update(out_in=float(u["epi"][out].mean() / u["epi"][~out].mean()), rho_T=float(spearmanr(u["epi"][out], excess[out])[0]),
                       ale_out_in=float(u["ale"][out].mean() / u["ale"][~out].mean()),
                       PICP90_tot_out=float(np.mean(np.abs(Yt - u["P"])[out] <= 1.645 * u["S"][out])),
                       PICP90_ale_out=float(np.mean(np.abs(Yt - u["P"])[out] <= 1.645 * u["ale"][out])))
        rows.append(row)
    save_json("e6_ood", rows)
    log(f"\n=== {cname} (n_train={len(sc.trn)}, test inputs outside training temperature range: {100*out.mean():.1f}%) ===")
    for k in ("MC", "LA-1", "LA-2", "ENS"):
        rr = [r for r in rows if r["cond"] == cname and r["method"] == k]; g = lambda x: float(np.nanmean([r.get(x, np.nan) for r in rr]))
        log(f"  {k:>5}: R_epi={g('R'):.3f} epi={g('epi'):7.1f} ale={g('ale'):7.1f} MAE={g('MAE'):6.1f} | PICP90 tot={g('PICP90_tot'):.3f} "
            f"ale-only={g('PICP90_ale'):.3f} | out-of-support: epi out/in={g('out_in'):.2f} ale out/in={g('ale_out_in'):.2f} "
            f"rho(epi,excess)={g('rho_T'):+.2f} PICP90 tot/ale={g('PICP90_tot_out'):.3f}/{g('PICP90_ale_out'):.3f}")
log("DONE")
