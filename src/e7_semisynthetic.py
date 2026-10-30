"""E7 (Section 4.4): semi-synthetic reference test with known ground truth.
Usage: python e7_semisynthetic.py gef|covid

A teacher network (same model class, seed 99, trained on the real data) defines the true conditional
mean f*(x) and the true noise level sigma*(x); labels y = f* + sigma* eps are generated on the real
inputs of every split. Since f* lies in the model class, (mu_hat - f*)^2 is pure estimation error.
Conditions: uniform 5 / 25 / 100 % of the training set and the most recent month.
Per decomposition: true vs reported epistemic share; calibration ratio E[(mu_hat-f*)^2]/E[sigma_epi^2];
coverage of f* by mu_hat +/- 1.645 sigma_epi; Spearman(sigma_epi, |mu_hat - f*|); aleatoric ratio
E[sigma_ale^2]/E[sigma*^2] and Spearman(sigma_ale, sigma*); per-hour agreement of the dominant source
(operator action) with the truth, and the share of truly epistemic-dominant hours labelled aleatoric."""
import sys, numpy as np, torch
from scipy.stats import spearmanr
from backbone import prepare, COLS, device
from decomp import base_net, ensemble, free_head, decompose, units
from models import ensemble_predict
from common import Logger, save_json, load_json

name = sys.argv[1]
log = Logger(f"e7_semisynthetic_{name}")
P = load_json("e2_dropout_sweep")["selected_p"]; TC = COLS[name]["temp"]; Z90 = 1.6448536

real = prepare(name); teacher, _ = base_net(real, 99, P)


@torch.no_grad()
def truth(Xn):
    teacher.eval(); mu, lv = teacher(torch.tensor(Xn).to(device))
    return mu.numpy() * real.scale + real.mean, np.sqrt(np.exp(lv.numpy())) * real.scale


fs, ss = {}, {}
for sp, Xn in (("train", real.trn), ("val", real.vdn), ("test", real.ten)):
    fs[sp], ss[sp] = truth(Xn)
rng = np.random.RandomState(2026); ys = {sp: fs[sp] + ss[sp] * rng.randn(len(fs[sp])) for sp in fs}
log(f"truth: RMS sigma* (test) = {np.sqrt(np.mean(ss['test']**2)):.1f}; sd f* (test) = {fs['test'].std():.1f} (data units)")


def score(u, f_true, s_true, mask=None):
    mu, se, sa = u["P"], u["epi"], u["ale"]
    if mask is not None:
        mu, se, sa, f_true, s_true = mu[mask], se[mask], sa[mask], f_true[mask], s_true[mask]
    e2 = (mu - f_true) ** 2
    true_epi_dom = e2 > s_true ** 2; rep_epi_dom = se ** 2 > sa ** 2
    return dict(rmse_mean=float(np.sqrt(e2.mean())), rms_epi=float(np.sqrt(np.mean(se ** 2))),
                true_R=float(e2.mean() / (e2.mean() + np.mean(s_true ** 2))), R=float(np.mean(se ** 2 / (se ** 2 + sa ** 2))),
                epi_ratio=float(e2.mean() / max(np.mean(se ** 2), 1e-12)),
                epi_cov90=float(np.mean(np.sqrt(e2) <= Z90 * np.maximum(se, 1e-12))),
                epi_rho=float(spearmanr(se, np.sqrt(e2))[0]) if se.std() > 0 else float("nan"),
                ale_ratio=float(np.mean(sa ** 2) / np.mean(s_true ** 2)), ale_rho=float(spearmanr(sa, s_true)[0]),
                true_epi_dom=float(true_epi_dom.mean()), action_agree=float(np.mean(true_epi_dom == rep_epi_dom)),
                miss_epi=float(np.mean(~rep_epi_dom[true_epi_dom])) if true_epi_dom.any() else float("nan"))


N = len(real.trn)
CONDS = [("uniform 5%", dict(frac=0.05)), ("uniform 25%", dict(frac=0.25)), ("uniform 100%", dict()), ("recent 1 month", dict(n=720, recent=True))]
OUT = {}
for cname, kw in CONDS:
    sc = prepare(name, labels=ys, **kw); Xt = sc.X("test")
    T = sc.raw_train[:, :, TC]; Tt = sc.raw_test[:, :, TC]
    out = np.maximum(0, np.maximum(Tt - T.max(), T.min() - Tt)).max(1) > 0
    rows = []

    def add(k, s, r):
        u = units(r, sc); d = score(u, fs["test"], ss["test"]); d.update(method=k, seed=s)
        if out.any() and (~out).any():
            d["in"] = score(u, fs["test"], ss["test"], ~out); d["out"] = score(u, fs["test"], ss["test"], out)
        rows.append(d)

    for s in (0, 1, 2):
        m, _ = base_net(sc, s, P, "_semi")
        dec, meta = decompose(m, sc, ("MC", "LA-1", "LA-2"), ("test",))
        for k, dd in dec.items():
            add(k, s, dd["test"])
        f, _ = free_head(sc, s, tagx="_semi"); add("FREE", s, f.predict(Xt))
    ms, _ = ensemble(sc, (10, 11, 12, 13, 14), P, "_semi"); add("ENS", 0, ensemble_predict(ms, Xt))
    OUT[cname] = dict(rows=rows, frac_out=float(out.mean()), n=len(sc.trn))
    save_json(f"e7_semisynthetic_{name}", OUT)
    log(f"\n=== {cname} (n_train={len(sc.trn)}; test inputs outside training temperature range {100*out.mean():.0f}%) ===")
    for k in ("LA-1", "LA-2", "MC", "ENS", "FREE"):
        rr = [r for r in rows if r["method"] == k]; g = lambda x: float(np.nanmean([r[x] for r in rr]))
        log(f"  {k:>5}: true R={g('true_R'):.2f} reported R={g('R'):.3f} | ratio E[e2]/E[s2]={g('epi_ratio'):7.2f} cov90(f*)={g('epi_cov90'):.2f} "
            f"rho={g('epi_rho'):+.2f} | ale ratio={g('ale_ratio'):.2f} rho={g('ale_rho'):+.2f} | action agree={g('action_agree'):.2f} "
            f"epi-dominant hours labelled aleatoric={g('miss_epi'):.2f} (truly epi-dominant {100*g('true_epi_dom'):.0f}%)")
        if "out" in rr[0]:
            o = lambda x: float(np.nanmean([r["out"][x] for r in rr]))
            log(f"         out-of-support: ratio={o('epi_ratio'):7.2f} cov90={o('epi_cov90'):.2f} ale ratio={o('ale_ratio'):.2f}")
log("DONE")
