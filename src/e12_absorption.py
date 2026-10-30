"""E12: what the heteroscedastic head estimates (Proposition 5).
On the semi-synthetic reference of E7, compare the variance output v(x) of the deployed networks (three
seeds per condition) with the true noise variance sigma*^2(x) and with sigma*^2(x) + (mu(x) - f*(x))^2,
the quantity the Gaussian likelihood trains the head to estimate. Reports the ratio of mean levels and the
Spearman correlation with each target on the test inputs."""
import numpy as np, torch
from scipy.stats import spearmanr
from backbone import prepare, load_raw, Scaled, device
from decomp import base_net
from common import Logger, save_json, load_json

log = Logger("e12_absorption")
P = load_json("e2_dropout_sweep")["selected_p"]
OUT = []


def window_sc(name, labels, n=720, val_frac=0.2):
    raw = load_raw(name); raw = {k: (raw[k][0], labels[k]) for k in raw}
    tr, trl = raw["train"]; te, tel = raw["test"]; N = len(tr)
    idx = np.arange(N - n, N); nv = int(round(val_frac * n)); tri, vai = idx[:-nv], idx[-nv:]
    sc = Scaled(tr[tri], trl[tri], tr[vai], trl[vai], te, tel)
    sc.raw_train, sc.raw_test, sc.name, sc.idx, sc.n_total = tr[tri], te, name, tri, N
    return sc


for name in ("gef", "covid"):
    full = prepare(name); teacher, _ = base_net(full, 99, P); out = {}
    with torch.no_grad():
        for sp, Xn in (("train", full.trn), ("val", full.vdn), ("test", full.ten)):
            teacher.eval(); mu, lv = teacher(torch.tensor(Xn).to(device))
            out[sp] = (mu.numpy() * full.scale + full.mean, np.sqrt(np.exp(lv.numpy())) * full.scale)
    rng = np.random.RandomState(2026); ys = {sp: out[sp][0] + out[sp][1] * rng.randn(len(out[sp][0])) for sp in out}
    f, s = out["test"]
    for cname, kw in (("uniform 5%", dict(frac=0.05)), ("uniform 25%", dict(frac=0.25)), ("uniform 100%", dict()),
                      ("recent 1 month", dict(n=720, recent=True)), ("recent 1 month (window)", None)):
        sc = prepare(name, labels=ys, **kw) if kw is not None else window_sc(name, ys)
        for seed in (0, 1, 2):
            m, _ = base_net(sc, seed, P, "_semi" if kw is not None else "_semi_iv"); m.eval()
            with torch.no_grad():
                mu, lv = m(sc.X("test").to(device))
            mu = mu.numpy() * sc.scale + sc.mean; v = np.exp(lv.numpy()) * sc.scale ** 2; e2 = (mu - f) ** 2
            OUT.append(dict(dataset=name, cond=cname, seed=seed, lvl_noise=float(v.mean() / (s ** 2).mean()),
                            lvl_total=float(v.mean() / (s ** 2 + e2).mean()), rho_noise=float(spearmanr(v, s ** 2)[0]),
                            rho_total=float(spearmanr(v, s ** 2 + e2)[0])))
        rr = [r for r in OUT if r["dataset"] == name and r["cond"] == cname]; g = lambda x: float(np.mean([r[x] for r in rr]))
        log(f"{name:5} {cname:15} E[v]/E[s*^2]={g('lvl_noise'):6.2f}  E[v]/E[s*^2+e^2]={g('lvl_total'):5.2f}  "
            f"rho(v,s*^2)={g('rho_noise'):+.2f}  rho(v,s*^2+e^2)={g('rho_total'):+.2f}")
save_json("e12_absorption", OUT)
log("DONE")
