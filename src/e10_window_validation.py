"""E10 (review round): robustness of the extrapolation results to the early-stopping rule.

In E6 and E7 the short-window models (most recent one to three months) are early-stopped on the
validation YEAR, which lies outside the training window; that rule selects networks after one to three
epochs. Here the same conditions are re-run with an operator-style rule: the last 20 % of the training
window (chronological) is held out for early stopping, so the network is trained to convergence on
what it can see.
  (A) real GEFCom2014 data, windows of 1 / 2 / 3 months: MC, LA-1, LA-2 (3 seeds) and one ensemble,
      statistics as in E6;
  (B) semi-synthetic reference (both datasets), window of 1 month: MC, LA-1, LA-2, FREE (3 seeds),
      one ensemble, and the beta-NLL remedy (MC, LA-2), statistics as in E7 plus balanced accuracy."""
import numpy as np, torch
from scipy.stats import spearmanr
from backbone import load_raw, prepare, Scaled, COLS, setup_seed, device
from models import HeteroMCDropout, ensemble_predict
from decomp import base_net, ensemble, free_head, decompose, units, tag
from metrics import gaussian_metrics
from common import Logger, save_json, load_json, ck

log = Logger("e10_window_validation")
P = load_json("e2_dropout_sweep")["selected_p"]; Z90 = 1.6448536
OUT = {"real": [], "semi": []}


def window_sc(name, n, labels=None, val_frac=0.2):
    raw = load_raw(name)
    if labels is not None:
        raw = {k: (raw[k][0], labels[k]) for k in raw}
    tr, trl = raw["train"]; te, tel = raw["test"]; N = len(tr)
    idx = np.arange(N - n, N); nv = int(round(val_frac * n)); tri, vai = idx[:-nv], idx[-nv:]
    sc = Scaled(tr[tri], trl[tri], tr[vai], trl[vai], te, tel)
    sc.raw_train, sc.raw_test, sc.name, sc.idx, sc.n_total = tr[tri], te, name, tri, N
    return sc


# ------------------------------------------------------------------ (A) real data, seasonal windows
name = "gef"; TC = COLS[name]["temp"]
for cname, n in (("recent 1 month", 720), ("recent 2 months", 1440), ("recent 3 months", 2160)):
    sc = window_sc(name, n); Xt, Yt = sc.X("test"), sc.Y("test")
    T = sc.raw_train[:, :, TC]; Tt = sc.raw_test[:, :, TC]
    excess = np.maximum(0, np.maximum(Tt - T.max(), T.min() - Tt)).max(1); out = excess > 0
    cands, ep = [], []
    for s in (0, 1, 2):
        m, info = base_net(sc, s, P, "_iv"); ep.append(info["epochs"])
        dec, _ = decompose(m, sc, ("MC", "LA-1", "LA-2"), ("test",))
        cands += [(k, s, d["test"]) for k, d in dec.items()]
    ms, _ = ensemble(sc, (10, 11, 12, 13, 14), P, "_iv"); cands.append(("ENS", 0, ensemble_predict(ms, Xt)))
    for k, s, r in cands:
        u = units(r, sc); tot = gaussian_metrics(Yt, u["P"], u["S"]); ale = gaussian_metrics(Yt, u["P"], u["ale"])
        row = dict(cond=cname, method=k, seed=s, n=len(sc.trn), frac_out=float(out.mean()), R=u["R"], epi=float(u["epi"].mean()),
                   ale=float(u["ale"].mean()), MAE=tot["MAE"], PICP90_tot=tot["PICP90"], PICP90_ale=ale["PICP90"], epochs=ep)
        if out.any() and (~out).any():
            row.update(out_in=float(u["epi"][out].mean() / u["epi"][~out].mean()),
                       PICP90_tot_out=float(np.mean(np.abs(Yt - u["P"])[out] <= Z90 * u["S"][out])),
                       PICP90_ale_out=float(np.mean(np.abs(Yt - u["P"])[out] <= Z90 * u["ale"][out])))
        OUT["real"].append(row)
    save_json("e10_window_validation", OUT)
    log(f"\n=== real {cname} (n_train={len(sc.trn)}, epochs of base nets {ep}, out-of-support {100*out.mean():.0f}%) ===")
    for k in ("MC", "LA-1", "LA-2", "ENS"):
        rr = [r for r in OUT["real"] if r["cond"] == cname and r["method"] == k]; g = lambda x: float(np.nanmean([r.get(x, np.nan) for r in rr]))
        log(f"  {k:>5}: R_epi={g('R'):.3f} epi={g('epi'):7.1f} ale={g('ale'):7.1f} MAE={g('MAE'):6.1f} out/in={g('out_in'):.2f} "
            f"dcov_out={100*(g('PICP90_tot_out')-g('PICP90_ale_out')):+.1f}")


# ------------------------------------------------------------------ (B) semi-synthetic, one month
def score(u, f, s):
    e2 = (u["P"] - f) ** 2; se2 = u["epi"] ** 2; sa2 = u["ale"] ** 2
    dom = e2 > s ** 2; rep = se2 > sa2
    rec = float(rep[dom].mean()) if dom.any() else float("nan"); spec = float((~rep[~dom]).mean()) if (~dom).any() else float("nan")
    return dict(true_R=float(e2.mean() / (e2.mean() + np.mean(s ** 2))), R_means=float(se2.mean() / (se2.mean() + sa2.mean())),
                epi_ratio=float(e2.mean() / max(se2.mean(), 1e-12)), epi_cov90=float(np.mean(np.sqrt(e2) <= Z90 * u["epi"])),
                epi_rho=float(spearmanr(u["epi"], np.sqrt(e2))[0]), ale_ratio=float(sa2.mean() / np.mean(s ** 2)),
                ale_rho=float(spearmanr(u["ale"], s)[0]), base=float(dom.mean()), recall=rec, ba=0.5 * (rec + spec),
                rmse=float(np.sqrt(e2.mean())))


for name in ("gef", "covid"):
    # the teacher is the full-data network of E7 (seed 99), loaded from its checkpoint
    full = prepare(name); teacher, _ = base_net(full, 99, P)

    @torch.no_grad()
    def tr_(Xn):
        teacher.eval(); mu, lv = teacher(torch.tensor(Xn).to(device))
        return mu.numpy() * full.scale + full.mean, np.sqrt(np.exp(lv.numpy())) * full.scale
    fs, ss = {}, {}
    for sp, Xn in (("train", full.trn), ("val", full.vdn), ("test", full.ten)):
        fs[sp], ss[sp] = tr_(Xn)
    rng = np.random.RandomState(2026); ys = {sp: fs[sp] + ss[sp] * rng.randn(len(fs[sp])) for sp in fs}
    sc = window_sc(name, 720, labels=ys); Xt = sc.X("test")

    def add(k, s, r, variant):
        d = score(units(r, sc), fs["test"], ss["test"]); d.update(dataset=name, variant=variant, method=k, seed=s); OUT["semi"].append(d)

    ep = []
    for s in (0, 1, 2):
        m, info = base_net(sc, s, P, "_semi_iv"); ep.append(info["epochs"])
        dec, _ = decompose(m, sc, ("MC", "LA-1", "LA-2"), ("test",))
        for k, dd in dec.items():
            add(k, s, dd["test"], "NLL")
        f, _ = free_head(sc, s, tagx="_semi_iv"); add("FREE", s, f.predict(Xt), "NLL")
        setup_seed(s); mb = HeteroMCDropout(sc.sl, sc.nv, dropout=P); mb.beta = 0.5
        ib = mb.train_model(*sc.loaders(), save=ck(f"base_{tag(sc, '_semi_iv_bnll')}_p{int(round(P*100)):02d}_s{s}")); ep.append(ib["epochs"])
        dec, _ = decompose(mb, sc, ("MC", "LA-2"), ("test",))
        for k, dd in dec.items():
            add(k, s, dd["test"], "beta-NLL")
    ms, _ = ensemble(sc, (10, 11, 12, 13, 14), P, "_semi_iv"); add("ENS", 0, ensemble_predict(ms, Xt), "NLL")
    save_json("e10_window_validation", OUT)
    log(f"\n=== semi-synthetic {name}, recent 1 month, in-window validation (epochs {ep}) ===")
    for variant, k in (("NLL", "FREE"), ("NLL", "MC"), ("NLL", "ENS"), ("NLL", "LA-1"), ("NLL", "LA-2"), ("beta-NLL", "MC"), ("beta-NLL", "LA-2")):
        rr = [r for r in OUT["semi"] if r["dataset"] == name and r["variant"] == variant and r["method"] == k]
        g = lambda x: float(np.nanmean([r[x] for r in rr]))
        log(f"  {variant:>8} {k:>4}: true R={g('true_R'):.2f} reported={g('R_means'):.3f} ratio={g('epi_ratio'):7.2f} cov90={g('epi_cov90'):.2f} "
            f"rho={g('epi_rho'):+.2f} ale ratio={g('ale_ratio'):6.2f} ale rho={g('ale_rho'):+.2f} base={g('base'):.2f} recall={g('recall'):.2f} "
            f"BA={g('ba'):.2f} RMSE={g('rmse'):.1f}")
log("DONE")
