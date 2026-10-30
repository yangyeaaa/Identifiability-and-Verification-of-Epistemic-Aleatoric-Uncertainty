"""E14: time scale of the retraining decision (Proposition 6) on the semi-synthetic reference of E7 / E10.
Usage: python e14_timescale.py gef|covid

The operator's question is whether, over a window of H hours, the deployed network's error is dominated by
estimation error, Y_B = 1{sum_B (mu_hat - f*)^2 > sum_B sigma*^2}. Within one training condition of T4 this label
is (almost) constant over the test year, so the question is posed on two deployment states: the network trained
on the full year (inside its experience) and the network trained on the most recent month (outside it; early
stopping on the validation year as in E7, or on the window as in E10). For every decomposition, seed and window
length H in {1, 24, 168, 720} hours the test-year blocks of the two states are pooled and the script reports
  * AUROC of the block ratio sum sigma_epi^2 / sum sigma_ale^2 for the label (ranking, threshold-free);
  * balanced accuracy and flagged share of the split's own rule sum sigma_epi^2 > sum sigma_ale^2 (= R_epi > 1/2);
  * the flagged share predicted by Proposition 6(iii), i.e. the share of one-month blocks with r_B > kappa * a,
    kappa = E[(mu_hat - f*)^2] / E[sigma_epi^2] and a = E[sigma_ale^2] / E[sigma*^2] of the one-month network.
Per state it also reports the hour-level AUROC and the autocorrelation and effective block size H_eff of the
squared estimation error (an empirical proxy: Eq. (4) with equal weights and the autocorrelation of the
squared error along the test year in place of rho_k^2). Models are shared with E7 / E10 (same checkpoint names)."""
import sys, numpy as np, torch
from sklearn.metrics import roc_auc_score
from backbone import prepare, load_raw, Scaled, device
from decomp import base_net, ensemble, free_head, decompose, units
from models import ensemble_predict
from common import Logger, save_json, load_json

name = sys.argv[1]
log = Logger(f"e14_timescale_{name}")
P = load_json("e2_dropout_sweep")["selected_p"]
SCALES = ((1, "hour"), (24, "day"), (168, "week"), (720, "30 days"))

real = prepare(name); teacher, _ = base_net(real, 99, P)
with torch.no_grad():
    fs, ss = {}, {}
    for sp, Xn in (("train", real.trn), ("val", real.vdn), ("test", real.ten)):
        teacher.eval(); mu, lv = teacher(torch.tensor(Xn).to(device))
        fs[sp], ss[sp] = mu.numpy() * real.scale + real.mean, np.sqrt(np.exp(lv.numpy())) * real.scale
rng = np.random.RandomState(2026); ys = {sp: fs[sp] + ss[sp] * rng.randn(len(fs[sp])) for sp in fs}


def window_sc(n=720, val_frac=0.2):
    raw = load_raw(name); raw = {k: (raw[k][0], ys[k]) for k in raw}
    tr, trl = raw["train"]; te, tel = raw["test"]; N = len(tr)
    idx = np.arange(N - n, N); nv = int(round(val_frac * n)); tri, vai = idx[:-nv], idx[-nv:]
    sc = Scaled(tr[tri], trl[tri], tr[vai], trl[vai], te, tel)
    sc.raw_train, sc.raw_test, sc.name, sc.idx, sc.n_total = tr[tri], te, name, tri, N
    return sc


STATES = {"full": (lambda: prepare(name, labels=ys), "_semi"),
          "1mo val": (lambda: prepare(name, n=720, recent=True, labels=ys), "_semi"),
          "1mo win": (window_sc, "_semi_iv")}
ARR = {}
for st, (mk, tagx) in STATES.items():
    sc = mk(); Xt = sc.X("test")

    def keep(k, s, r):
        u = units(r, sc)
        ARR[(st, k, s)] = ((u["P"] - fs["test"]) ** 2, ss["test"] ** 2, u["epi"] ** 2, u["ale"] ** 2)

    for s in (0, 1, 2):
        m, _ = base_net(sc, s, P, tagx)
        dec, _ = decompose(m, sc, ("MC", "LA-1", "LA-2"), ("test",))
        for k, dd in dec.items():
            keep(k, s, dd["test"])
        f, _ = free_head(sc, s, tagx=tagx); keep("FREE", s, f.predict(Xt))
    ms, _ = ensemble(sc, (10, 11, 12, 13, 14), P, tagx); keep("ENS", 0, ensemble_predict(ms, Xt))
    log(f"{log.elapsed()} state {st}: n_train={len(sc.trn)}")


def blk(a, H):
    n = len(a) // H * H
    return a[:n].reshape(-1, H).sum(1)


def heff(x, H):
    x = x - x.mean(); v = np.mean(x * x)
    rho = np.array([np.mean(x[:-k] * x[k:]) / v for k in range(1, H)])
    return float(H / (1 + 2 * np.sum((1 - np.arange(1, H) / H) * rho)))


OUT = dict(pairs=[], states=[])
for (st, k, s), (e2, s2, se2, sa2) in ARR.items():
    t = e2 > s2
    OUT["states"].append(dict(state=st, method=k, seed=s, base=float(t.mean()),
                              auc_hour=float(roc_auc_score(t, se2 / sa2)) if 0 < t.mean() < 1 else float("nan"),
                              acf1=float(np.corrcoef(e2[:-1], e2[1:])[0, 1]), heff_day=heff(e2, 24), heff_week=heff(e2, 168),
                              kappa=float(e2.mean() / se2.mean()), a=float(sa2.mean() / s2.mean())))
for out_state in ("1mo val", "1mo win"):
    for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
        for s in ((0,) if k == "ENS" else (0, 1, 2)):
            A, B = ARR[("full", k, s)], ARR[(out_state, k, s)]
            kappa_a = (B[0].mean() / B[2].mean()) * (B[3].mean() / B[1].mean())
            for H, lab in SCALES:
                (e2a, s2a, sea, saa), (e2b, s2b, seb, sab) = ([blk(x, H) for x in A], [blk(x, H) for x in B])
                t = np.concatenate([e2a > s2a, e2b > s2b]); q = np.concatenate([sea / saa, seb / sab]); rep = q > 1
                ba = 0.5 * (rep[t].mean() + (~rep[~t]).mean()) if 0 < t.mean() < 1 else float("nan")
                OUT["pairs"].append(dict(out_state=out_state, method=k, seed=s, scale=lab, H=H,
                                         auc=float(roc_auc_score(t, q)) if 0 < t.mean() < 1 else float("nan"), ba_rule=float(ba),
                                         true_full=float((e2a > s2a).mean()), true_out=float((e2b > s2b).mean()),
                                         flag_full=float((sea > saa).mean()), flag_out=float((seb > sab).mean()),
                                         flag_out_pred=float((e2b / s2b > kappa_a).mean()), kappa_a=float(kappa_a)))
save_json(f"e14_timescale_{name}", OUT)
for out_state in ("1mo val", "1mo win"):
    log(f"\n=== {name}: full year vs one month ({out_state}) ===")
    for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
        for H, lab in SCALES:
            rr = [r for r in OUT["pairs"] if r["out_state"] == out_state and r["method"] == k and r["H"] == H]
            g = lambda x: float(np.nanmean([r[x] for r in rr]))
            log(f"  {k:>5} {lab:>7}: AUROC={g('auc'):.2f} BA(rule)={g('ba_rule'):.2f} | flagged out={g('flag_out'):.2f} "
                f"(predicted {g('flag_out_pred'):.2f}, kappa*a={g('kappa_a'):.1f}; truth {g('true_out'):.2f}) full={g('flag_full'):.2f} (truth {g('true_full'):.2f})")
log("DONE")
