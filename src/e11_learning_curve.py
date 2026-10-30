"""E11: cross-fitted learning-curve decomposition (LCD) on the semi-synthetic reference.
Usage: python e11_learning_curve.py gef|covid

Definition. For a model trained on n samples, the expected held-out squared error at x is
    R_n(x) = E[(y - mu_n(x))^2 | x] = sigma^2(x) + e_n(x),
with sigma^2 the irreducible (aleatoric) part and e_n the reducible (epistemic) part. Under a power law
e_n(x) = c(x) (n/N)^(-alpha) both parts are identified from held-out errors at several sample sizes.

Estimation. The training set D (size N) is split into K folds (random for uniform subsets, contiguous
blocks for the one-month window). For each fold k and each fraction of the remaining pool (1/8, 1/4,
1/2, 1; nested), a network is trained with the same procedure as the deployed model and its residuals
on fold k are recorded. A second network g(x) = (a(x), b(x)) and a global alpha are fitted to all
held-out residuals by the Gaussian likelihood with variance exp(a(x)) + exp(b(x)) (n/N)^(-alpha).
For the deployed network (trained on all of D): sigma_ale^2 = exp(a), sigma_epi^2 = exp(b).

Evaluation: the T4 metrics of E7/E10 on the same deployed networks (seeds 0-2), and an ablation that
fits a single variance to the held-out residuals of the largest pool (no learning curve)."""
import sys, numpy as np, torch, torch.nn as nn
from scipy.stats import spearmanr
from backbone import load_raw, prepare, Scaled, setup_seed, subset_index, device
from models import HeteroMCDropout
from decomp import base_net
from common import Logger, save_json, load_json, ck

name = sys.argv[1] if len(sys.argv) > 1 else "gef"
log = Logger(f"e11_lcd_{name}")
P = load_json("e2_dropout_sweep")["selected_p"]; Z90 = 1.6448536
FRACS = (1 / 8, 1 / 4, 1 / 2, 1.0)

# ------------------------------------------------------------------ truth (same as E7)
full = prepare(name); teacher, _ = base_net(full, 99, P)


@torch.no_grad()
def tr_(Xn):
    teacher.eval(); mu, lv = teacher(torch.tensor(Xn).to(device))
    return mu.numpy() * full.scale + full.mean, np.sqrt(np.exp(lv.numpy())) * full.scale


fs, ss = {}, {}
for sp, Xn in (("train", full.trn), ("val", full.vdn), ("test", full.ten)):
    fs[sp], ss[sp] = tr_(Xn)
rng = np.random.RandomState(2026); ys = {sp: fs[sp] + ss[sp] * rng.randn(len(fs[sp])) for sp in fs}
raw = load_raw(name); tr, te, vd = raw["train"][0], raw["test"][0], raw["val"][0]
trl, tel, vdl = ys["train"], ys["test"], ys["val"]; N_ALL = len(tr)


def sc_from(idx, val):
    V, vy = (vd, vdl) if val is None else (tr[val], trl[val])
    sc = Scaled(tr[idx], trl[idx], V, vy, te, tel)
    sc.raw_train, sc.raw_test, sc.name, sc.idx, sc.n_total = tr[idx], te, name, np.asarray(idx), N_ALL
    return sc


def mu_raw(m, sc, Xraw):
    Xn = sc.sdd.transform(Xraw.reshape(-1, sc.nv)).reshape(-1, sc.sl, sc.nv).astype(np.float32)
    return m.predict(torch.tensor(Xn), mc_dropout=False)["mu"] * sc.scale + sc.mean


# ------------------------------------------------------------------ conditions
N720 = np.arange(N_ALL - 720, N_ALL)
CONDS = [
    ("uniform 5%", "u5", subset_index(N_ALL, 0.05), None, 5, "random", lambda: (prepare(name, frac=0.05, labels=ys), "_semi")),
    ("recent 1 month (window stop)", "m1w", N720[:576], N720[576:], 4, "block", lambda: (sc_from(N720[:576], N720[576:]), "_semi_iv")),
    ("recent 1 month (val. year stop)", "m1v", N720, None, 4, "block", lambda: (prepare(name, n=720, recent=True, labels=ys), "_semi")),
    ("uniform 25%", "u25", subset_index(N_ALL, 0.25), None, 5, "random", lambda: (prepare(name, frac=0.25, labels=ys), "_semi")),
    ("uniform 100%", "u100", np.arange(N_ALL), None, 5, "random", lambda: (prepare(name, labels=ys), "_semi")),
]


class G(nn.Module):
    def __init__(self, d, a0, b0, curve=True):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 2))
        with torch.no_grad():
            self.net[-1].weight.mul_(0.1); self.net[-1].bias.copy_(torch.tensor([a0, b0]))
        self.th = nn.Parameter(torch.tensor(0.0)); self.curve = curve

    def forward(self, x, ln):
        o = self.net(x); a, b = o[:, 0], o[:, 1]
        alpha = 0.2 + 1.8 * torch.sigmoid(self.th)
        v = torch.exp(a) + (torch.exp(b - alpha * ln) if self.curve else 0.0)
        return v, a, b, alpha


def fit_g(Xf, ln, r, groups, curve=True, seed=0):
    torch.manual_seed(seed); rs = np.random.RandomState(seed)
    ug = np.unique(groups); hold = np.isin(groups, rs.choice(ug, int(0.15 * len(ug)), replace=False))
    m2 = float(np.mean(r ** 2)); g = G(Xf.shape[1], np.log(0.7 * m2), np.log(0.3 * m2), curve)
    opt = torch.optim.Adam(g.parameters(), lr=1e-3, weight_decay=1e-4)
    T = lambda a: torch.tensor(a, dtype=torch.float32)
    Xt, Lt, Rt = T(Xf[~hold]), T(ln[~hold]), T(r[~hold]); Xv, Lv, Rv = T(Xf[hold]), T(ln[hold]), T(r[hold])
    best, state, pc = np.inf, None, 0
    for ep in range(400):
        g.train(); perm = torch.randperm(len(Xt))
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]; v, *_ = g(Xt[b], Lt[b]); loss = 0.5 * (torch.log(v) + Rt[b] ** 2 / v).mean()
            opt.zero_grad(); loss.backward(); opt.step()
        g.eval()
        with torch.no_grad():
            v, *_ = g(Xv, Lv); vl = float(0.5 * (torch.log(v) + Rv ** 2 / v).mean())
        if vl < best - 1e-5: best, state, pc = vl, {k: t.clone() for k, t in g.state_dict().items()}, 0
        else:
            pc += 1
            if pc >= 25: break
    g.load_state_dict(state); return g


def score(P_, epi, ale, f, s):
    e2 = (P_ - f) ** 2; se2 = epi ** 2; sa2 = ale ** 2; dom = e2 > s ** 2; rep = se2 > sa2
    rec = float(rep[dom].mean()) if dom.any() else float("nan"); spec = float((~rep[~dom]).mean()) if (~dom).any() else float("nan")
    return dict(true_R=float(e2.mean() / (e2.mean() + np.mean(s ** 2))), R_means=float(se2.mean() / (se2.mean() + sa2.mean())),
                epi_ratio=float(e2.mean() / max(se2.mean(), 1e-12)), epi_cov90=float(np.mean(np.sqrt(e2) <= Z90 * epi)),
                epi_rho=float(spearmanr(epi, np.sqrt(e2))[0]), ale_ratio=float(sa2.mean() / np.mean(s ** 2)),
                ale_rho=float(spearmanr(ale, s)[0]), base=float(dom.mean()), recall=rec, ba=0.5 * (rec + spec),
                rmse=float(np.sqrt(e2.mean())))


OUT = []
for cname, ctag, D, val, K, kind, dep in CONDS:
    N = len(D); rs = np.random.RandomState(7)
    folds = np.array_split(rs.permutation(N) if kind == "random" else np.arange(N), K)
    Xr, Lr, Rr, Gr = [], [], [], []
    for k, fk in enumerate(folds):
        pool = np.setdiff1d(np.arange(N), fk); pool = pool[np.random.RandomState(100 + k).permutation(len(pool))]
        for j, fr in enumerate(FRACS):
            sub = np.sort(pool[: int(round(fr * len(pool)))])
            sc = sc_from(D[sub], val); setup_seed(10 * k + j)
            m = HeteroMCDropout(sc.sl, sc.nv, dropout=P)
            m.train_model(*sc.loaders(), save=ck(f"lcd_{name}_{ctag}_k{k}_f{j}"))
            r = trl[D[fk]] - mu_raw(m, sc, tr[D[fk]])
            Xr.append(D[fk]); Lr.append(np.full(len(fk), np.log(len(sub) / N))); Rr.append(r); Gr.append(D[fk])
    Xr, Lr, Rr, Gr = map(np.concatenate, (Xr, Lr, Rr, Gr))
    scd, tagx = dep()
    Xf = scd.sdd.transform(tr[Xr].reshape(-1, scd.nv)).reshape(len(Xr), -1).astype(np.float32)
    Xte = scd.ten.reshape(len(scd.ten), -1)
    g = fit_g(Xf, Lr, Rr / scd.scale, Gr, curve=True)
    sel = Lr == Lr.max(); g1 = fit_g(Xf[sel], Lr[sel] * 0, Rr[sel] / scd.scale, Gr[sel], curve=False)
    with torch.no_grad():
        T = torch.tensor(Xte, dtype=torch.float32); z = torch.zeros(len(T))
        _, a, b, alpha = g(T, z); ale = np.sqrt(np.exp(a.numpy())) * scd.scale; epi = np.sqrt(np.exp(b.numpy())) * scd.scale
        v1, *_ = g1(T, z); ho = np.sqrt(v1.numpy()) * scd.scale
    mse_n = {round(float(np.exp(l)), 3): float(np.mean(Rr[Lr == l] ** 2)) for l in np.unique(Lr)}
    rows = []
    for s in (0, 1, 2):
        m, _ = base_net(scd, s, P, tagx); Pd = m.predict(scd.X("test"), mc_dropout=False)["mu"] * scd.scale + scd.mean
        d = score(Pd, epi, ale, fs["test"], ss["test"]); d.update(cond=cname, method="LCD", seed=s, alpha=float(alpha))
        d["cov_y90"] = float(np.mean(np.abs(tel - Pd) <= Z90 * np.sqrt(ale ** 2 + epi ** 2))); rows.append(d)
        h = dict(cond=cname, method="HO-var", seed=s, ale_ratio=float(np.mean(ho ** 2) / np.mean(ss["test"] ** 2)),
                 cov_y90=float(np.mean(np.abs(tel - Pd) <= Z90 * ho))); rows.append(h)
    OUT += rows; save_json(f"e11_lcd_{name}", OUT)
    q = [r for r in rows if r["method"] == "LCD"]; gm = lambda x: float(np.nanmean([r[x] for r in q]))
    qh = [r for r in rows if r["method"] == "HO-var"]
    log(f"\n=== {cname} (N={N}, K={K}, held-out MSE by n/N: " + ", ".join(f"{k}:{v:.0f}" for k, v in mse_n.items()) + f"; alpha={float(alpha):.2f}) ===")
    log(f"  LCD   : true R={gm('true_R'):.2f} reported={gm('R_means'):.3f} ratio={gm('epi_ratio'):7.2f} cov90(f*)={gm('epi_cov90'):.2f} "
        f"rho={gm('epi_rho'):+.2f} | ale ratio={gm('ale_ratio'):5.2f} ale rho={gm('ale_rho'):+.2f} | base={gm('base'):.2f} "
        f"recall={gm('recall'):.2f} BA={gm('ba'):.2f} | cov90(y)={gm('cov_y90'):.3f} RMSE={gm('rmse'):.1f}")
    log(f"  HO-var: ale ratio (held-out variance, no curve)={np.mean([r['ale_ratio'] for r in qh]):5.2f} cov90(y)={np.mean([r['cov_y90'] for r in qh]):.3f}")
log("DONE")
