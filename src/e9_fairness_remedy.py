"""E9 (review round): (a) MC dropout at its own validation-optimal rate p = 0.1 - post-COVID accuracy
(5 seeds) and the T4 reference test (full year and recent month, both datasets, 3 seeds);
(b) beta-NLL (beta = 0.5) as a remedy for aleatoric absorption - T4 for MC and LA-2 (p = 0.2)."""
import numpy as np, torch
from scipy.stats import spearmanr
from backbone import prepare, setup_seed, device
from models import HeteroMCDropout
from decomp import base_net, decompose, units, tag
from metrics import gaussian_metrics, rolling_conformal
from common import Logger, save_json, load_json, ck

log = Logger("e9_fairness_remedy"); Z90 = 1.6448536
P_MC = 0.10; P = load_json("e2_dropout_sweep")["selected_p"]; OUT = {"covid_mc_p10": [], "semi": []}

# ---------------- (a1) post-COVID accuracy of MC dropout at p = 0.1
sc = prepare("covid"); Yv, Yt = sc.Y("val"), sc.Y("test")
for s in (0, 1, 2, 3, 4):
    m, _ = base_net(sc, s, P_MC)
    d, _ = decompose(m, sc, ("MC",), ("val", "test"))
    uv, ut = units(d["MC"]["val"], sc), units(d["MC"]["test"], sc)
    met = gaussian_metrics(Yt, ut["P"], ut["S"]); ro = rolling_conformal(Yv, uv["P"], uv["S"], Yt, ut["P"], ut["S"])
    OUT["covid_mc_p10"].append(dict(seed=s, R=ut["R"], **met, ECE_roll=ro["ECE"]))
    log(f"{log.elapsed()} covid MC p=0.1 s={s}: R_epi={ut['R']:.3f} MAE={met['MAE']/1e3:.2f} CRPS={met['CRPS']/1e3:.2f} ECE={met['ECE']:.3f} roll={ro['ECE']:.3f}")
save_json("e9_fairness_remedy", OUT)

# ---------------- (a2, b) semi-synthetic students
def truth_and_labels(name):
    real = prepare(name); teacher, _ = base_net(real, 99, P)
    @torch.no_grad()
    def tr(Xn):
        teacher.eval(); mu, lv = teacher(torch.tensor(Xn).to(device))
        return mu.numpy() * real.scale + real.mean, np.sqrt(np.exp(lv.numpy())) * real.scale
    fs, ss = {}, {}
    for sp, Xn in (("train", real.trn), ("val", real.vdn), ("test", real.ten)):
        fs[sp], ss[sp] = tr(Xn)
    rng = np.random.RandomState(2026); ys = {sp: fs[sp] + ss[sp] * rng.randn(len(fs[sp])) for sp in fs}
    return fs, ss, ys

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
    fs, ss, ys = truth_and_labels(name)
    for cname, kw in (("uniform 100%", dict()), ("recent 1 month", dict(n=720, recent=True))):
        sc = prepare(name, labels=ys, **kw)
        for variant, p, beta, methods in (("MC p=0.1", P_MC, 0.0, ("MC",)), ("beta-NLL", P, 0.5, ("MC", "LA-2"))):
            for s in (0, 1, 2):
                setup_seed(s); m = HeteroMCDropout(sc.sl, sc.nv, dropout=p); m.beta = beta
                m.train_model(*sc.loaders(), save=ck(f"base_{tag(sc, '_semi' + ('_bnll' if beta else ''))}_p{int(round(p*100)):02d}_s{s}"))
                d, _ = decompose(m, sc, methods, ("test",))
                for k in methods:
                    r = score(units(d[k]["test"], sc), fs["test"], ss["test"]); r.update(dataset=name, cond=cname, variant=variant, method=k, seed=s)
                    OUT["semi"].append(r)
            for k in methods:
                rr = [r for r in OUT["semi"] if r["dataset"] == name and r["cond"] == cname and r["variant"] == variant and r["method"] == k]
                g = lambda x: float(np.nanmean([r[x] for r in rr]))
                log(f"{log.elapsed()} {name} [{cname}] {variant:>9} {k:>4}: true R={g('true_R'):.2f} reported R(means)={g('R_means'):.3f} ratio={g('epi_ratio'):6.2f} "
                    f"cov90={g('epi_cov90'):.2f} ale ratio={g('ale_ratio'):5.2f} recall={g('recall'):.2f} BA={g('ba'):.2f} RMSE(mean)={g('rmse'):.1f}")
            save_json("e9_fairness_remedy", OUT)
log("DONE")
