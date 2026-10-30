"""E13: reserve-relevant evaluation of the total predictive distributions (real data, E3 predictions).

For each dataset, decomposition and seed:
  * upper one-sided coverage P(y <= mu + q_tau * sigma) at tau = 0.99 and 0.995, overall and on peak-load
    hours (top 10 % of test load), raw (Gaussian quantile) and after causal rolling calibration of the
    signed normalized residual (window 90 days, lag 24 h, seeded with validation residuals);
  * expected cost of an upward reserve R_t = mu_t + r_t with asymmetric cost c_u * shortfall + c_o * surplus
    (c_o = 1, c_u in {10, 20}): a constant reserve (rolling quantile of raw residuals), an adaptive reserve
    scaled by sigma_tot, and an adaptive reserve scaled by sigma_ale only (does the epistemic term help?)."""
import numpy as np
from backbone import prepare
from common import Logger, save_json, ck

log = Logger("e13_reserve")
W, LAG = 24 * 90, 24
METHODS = ("MVE", "MC", "ENS", "LA-2")


def rolling_q(seed_scores, test_scores, tau):
    s_all = np.concatenate([seed_scores, test_scores]); n0 = len(seed_scores); out = np.empty(len(test_scores))
    for t in range(len(test_scores)):
        end = n0 + t - LAG; win = s_all[max(0, end - W):end]
        out[t] = np.quantile(win, tau, method="higher")
    return out


OUT = []
for name in ("gef", "covid"):
    pr = np.load(ck(f"e3_preds_{name}").replace(".pth", ".npz")); sc = prepare(name)
    Yv, Yt = sc.Y("val"), sc.Y("test"); peak = Yt >= np.quantile(Yt, 0.9)
    for k in METHODS:
        seeds = sorted({key.split("__")[0] for key in pr.files if key.startswith(k + "_s")})
        for sk in seeds:
            g = lambda sp, q: pr[f"{sk}__{sp}__{q}"]
            Pv, Sv, Pt, St = g("val", "P"), g("val", "S"), g("test", "P"), g("test", "S")
            Av, At = (g("val", "ale"), g("test", "ale")) if f"{sk}__test__ale" in pr.files else (Sv, St)
            zv, zt = (Yv - Pv) / Sv, (Yt - Pt) / St
            row = dict(dataset=name, method=k, seed=sk)
            for tau in (0.99, 0.995):
                from scipy.stats import norm
                raw = Yt <= Pt + norm.ppf(tau) * St
                qr = rolling_q(zv, zt, tau); cal = Yt <= Pt + qr * St
                row[f"up{tau}_raw"] = float(raw.mean()); row[f"up{tau}_raw_peak"] = float(raw[peak].mean())
                row[f"up{tau}_cal"] = float(cal.mean()); row[f"up{tau}_cal_peak"] = float(cal[peak].mean())
            for cu in (10, 20):
                tau = cu / (cu + 1.0)
                cost = lambda R: float(np.mean(cu * np.maximum(Yt - R, 0) + np.maximum(R - Yt, 0)))
                rv, rt = Yv - Pv, Yt - Pt
                fixed = Pt + rolling_q(rv, rt, tau)
                adapt = Pt + rolling_q(zv, zt, tau) * St
                adapt_ale = Pt + rolling_q(rv / Av, rt / At, tau) * At
                row[f"cost{cu}_fixed"] = cost(fixed); row[f"cost{cu}_tot"] = cost(adapt); row[f"cost{cu}_ale"] = cost(adapt_ale)
            OUT.append(row)
        rr = [r for r in OUT if r["dataset"] == name and r["method"] == k]; m = lambda x: float(np.mean([r[x] for r in rr]))
        log(f"{name:5} {k:5}: up99 raw={m('up0.99_raw'):.3f} cal={m('up0.99_cal'):.3f} (peak raw={m('up0.99_raw_peak'):.3f} cal={m('up0.99_cal_peak'):.3f}) | "
            f"up99.5 cal={m('up0.995_cal'):.3f} peak={m('up0.995_cal_peak'):.3f} | cost(cu=20): fixed={m('cost20_fixed'):.1f} "
            f"tot={m('cost20_tot'):.1f} ale-only={m('cost20_ale'):.1f} ({100*(1-m('cost20_tot')/m('cost20_fixed')):+.1f}% vs fixed)")
    save_json("e13_reserve", OUT)
log("DONE")
