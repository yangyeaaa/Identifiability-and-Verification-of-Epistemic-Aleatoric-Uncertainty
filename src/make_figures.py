"""Regenerate all result figures from the JSON results of exp01-exp11."""
import os, sys, numpy as np, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import norm
from common import load_json, FIG, ck
from backbone import prepare
from metrics import conformal_q, rolling_scale_90

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
C_EPI, C_ALE, C_FREE, C_ENS, C_OURS = "#2ca02c", "#d2691e", "#8c564b", "#9467bd", "#1f77b4"
which = sys.argv[1:] or ["all"]
do = lambda k: "all" in which or k in which


def savefig(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=200, bbox_inches="tight"); plt.close()
    print("saved", name)


# ------------------------------------------------------------------ Fig: non-identifiability (Table 1 visual)
if do("nonident"):
    rows = load_json("01_counterexample")
    fig, ax = plt.subplots(figsize=(6.2, 4.2)); ax2 = ax.twinx(); ax2.spines["right"].set_visible(True)
    R = [r["R_epi"] for r in rows]; mae = [r["MAE"] for r in rows]; crps = [r["CRPS"] for r in rows]; ece = [r["ECE"] for r in rows]
    for r in rows:
        ax.axvline(r["R_epi"], color="0.8", ls=":", lw=1)
    ax.scatter(R, mae, s=110, color=C_OURS, label="MAE (left axis)", zorder=3)
    ax.scatter(R, crps, s=110, marker="D", color="#17becf", label="CRPS (left axis)", zorder=3)
    ax2.scatter(R, ece, s=110, marker="s", color=C_ALE, label="ECE (right axis)", zorder=3)
    ax.axhline(np.mean(mae), color=C_OURS, alpha=0.35); ax.axhline(np.mean(crps), color="#17becf", alpha=0.35)
    ax2.axhline(np.mean(ece), color=C_ALE, alpha=0.35, ls="--")
    for r in rows:
        ax.annotate(f"$R_{{\\rm epi}}$={r['R_epi']:.2f}", (r["R_epi"], r["MAE"]), textcoords="offset points", xytext=(0, -16), ha="center", color=C_OURS, fontweight="bold", fontsize=9)
    ax.set_xlim(-0.02, 1.02); ax.set_xlabel(r"Reported epistemic ratio $R_{\rm epi}$"); ax.set_ylabel("MAE / CRPS (MW)", color=C_OURS)
    lo = min(min(mae), min(crps)) - 5; hi = max(max(mae), max(crps)) + 14; ax.set_ylim(lo, hi)
    ax2.set_ylabel("ECE", color=C_ALE); ax2.set_ylim(0, 0.08)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", fontsize=9, ncol=3); ax.grid(alpha=0.25)
    savefig("nonidentifiability.png")

# ------------------------------------------------------------------ Fig: verification (reducibility, decile, k-NN)
if do("validation"):
    red = load_json("05_reducibility"); ood = load_json("06_ood")
    fracs = red["fracs"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    # (a) reducibility
    def agg(key, comp):
        m, s = [], []
        for f in fracs:
            v = np.array([r[comp] for r in red[key] if abs(r["frac"] - f) < 1e-9]); m.append(v.mean()); s.append(v.std(ddof=1) if len(v) > 1 else 0)
        return np.array(m), np.array(s)
    pct = np.array(fracs) * 100
    for key, col, lab in (("ours", C_OURS, "Proposed (MC dropout)"), ("ens", C_ENS, "Deep ensemble (K=5)"), ("free", C_FREE, "Free-head construction")):
        me, se = agg(key, "epi"); ma, sa = agg(key, "ale")
        ax[0].errorbar(pct, me, yerr=se, fmt="o-", color=col, lw=2, capsize=3, label=f"{lab}: epistemic")
        ax[0].errorbar(pct, ma, yerr=sa, fmt="s--", color=col, lw=1.2, alpha=0.6, capsize=3, label=f"{lab}: aleatoric")
    me, _ = agg("ours", "epi"); ref = me[-1] * np.sqrt(fracs[-1] / np.array(fracs))
    ax[0].plot(pct, ref, ":", color="0.4", lw=1.5, label=r"$1/\sqrt{n}$ reference (anchored at 100%)")
    ax[0].set_xscale("log"); ax[0].set_xticks(pct); ax[0].set_xticklabels([f"{p:g}" for p in pct]); ax[0].minorticks_off()
    ax[0].set_xlabel("Training data used (%)"); ax[0].set_ylabel("Mean uncertainty (MW)"); ax[0].set_title("(a) Reducibility")
    ax[0].legend(fontsize=7.5, frameon=False); ax[0].grid(alpha=0.3)
    # (b) decile smile
    get = lambda tag, ds="gef": [o for o in ood if o["tag"] == tag and o["dataset"] == ds][0]
    for tag, col, lab in (("ours", C_OURS, "Proposed"), ("ens", C_ENS, "Deep ensemble"), ("free", C_FREE, "Free-head")):
        o = get(tag); c = [s["center"] for s in o["smile"]]; e = [s["epi"] for s in o["smile"]]
        ax[1].plot(c, np.array(e) / np.mean(e), "o-", color=col, lw=2, label=lab)
    o = get("ours"); ax[1].axvspan(ax[1].get_xlim()[0], o["train_p5"], color="0.9"); ax[1].axvspan(o["train_p95"], ax[1].get_xlim()[1], color="0.9")
    ax[1].set_xlabel("True load level (MW)"); ax[1].set_ylabel("Mean epistemic / its average"); ax[1].set_title("(b) Epistemic by load decile")
    ax[1].legend(fontsize=8.5, frameon=False); ax[1].grid(alpha=0.3)
    # (c) kNN distance
    for tag, col, lab in (("ours", C_OURS, "Proposed"), ("ens", C_ENS, "Deep ensemble"), ("free", C_FREE, "Free-head")):
        o = get(tag); b = o["dist_bins"]; d = [x["dist"] for x in b]; e = np.array([x["epi"] for x in b]); a = np.array([x["ale"] for x in b])
        ax[2].plot(range(1, 11), e / e[0], "o-", color=col, lw=2, label=f"{lab}: epistemic ($\\rho$={o['rho_epi_dist']:.2f})")
        ax[2].plot(range(1, 11), a / a[0], "s--", color=col, lw=1.2, alpha=0.6, label=f"{lab}: aleatoric ($\\rho$={o['rho_ale_dist']:.2f})")
    ax[2].set_xlabel("Decile of k-NN distance to training inputs (k=10)"); ax[2].set_ylabel("Mean uncertainty / value in 1st decile")
    ax[2].set_title("(c) Distance to training support"); ax[2].legend(fontsize=7.5, frameon=False); ax[2].grid(alpha=0.3); ax[2].set_xticks(range(1, 11))
    savefig("decomposition_validation.png")

# ------------------------------------------------------------------ Fig: calibration curves
if do("calib"):
    cal = load_json("07_calibration")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
    for ax, (name, title) in zip(axs, (("gef", "(a) GEFCom2014"), ("covid", "(b) post-COVID demand"))):
        R = cal[name]; L = np.array(R["levels"])
        ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=1.2, label="Ideal")
        ax.fill_between([0.5, 0.95], [0.45, 0.90], [0.55, 1.0], color="0.88", label=r"$\pm$5% band")
        ax.plot(L, R["raw"], "o-", color="0.5", lw=2, label=f"Raw (ECE={R['ECE_raw']:.3f})")
        ax.plot(L, R["temp"], "^--", color=C_ALE, lw=2, label=f"Temperature (ECE={R['ECE_temp']:.3f})")
        ax.plot(L, R["conf"], "s-", color=C_EPI, lw=2, label=f"Split conformal (ECE={R['ECE_conf']:.3f})")
        ax.plot(L, R["roll"], "D-", color=C_OURS, lw=2, label=f"Rolling conformal (ECE={R['ECE_roll']:.3f})")
        ax.set_xlabel("Nominal confidence level"); ax.set_ylabel("Empirical coverage"); ax.set_title(title)
        ax.set_xlim(0.48, 0.97); ax.set_ylim(0.45, 1.0); ax.grid(alpha=0.25); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    savefig("calibration_curves.png")

# ------------------------------------------------------------------ Fig: sensitivity
if do("sens"):
    M = load_json("09_sensitivity_M"); P = load_json("02_select_p")["rows"]; cs = load_json("07_calibration")["gef"]["calset"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    ax[0].plot([r["M"] for r in M], [r["epi_MW"] for r in M], "o-", color=C_EPI, lw=2); ax[0].set_xscale("log")
    ax[0].set_xticks([r["M"] for r in M]); ax[0].set_xticklabels([str(r["M"]) for r in M]); ax[0].minorticks_off()
    ax[0].set_xlabel("Number of MC forward passes $M$"); ax[0].set_ylabel("Mean epistemic (MW)"); ax[0].set_title("(a) Forward passes"); ax[0].grid(alpha=0.3)
    a0 = ax[0].twinx(); a0.spines["right"].set_visible(True); a0.plot([r["M"] for r in M], [r["CRPS"] for r in M], "s--", color=C_OURS, lw=1.5); a0.set_ylabel("CRPS (MW)", color=C_OURS)
    ps = [r["p"] for r in P]
    ax[1].plot(ps, [r["epi_MW"] for r in P], "o-", color=C_EPI, lw=2, label="Epistemic (test)")
    ax[1].plot(ps, [r["ale_MW"] for r in P], "s--", color=C_ALE, lw=2, label="Aleatoric (test)")
    ax[1].set_xlabel("Dropout rate $p$"); ax[1].set_ylabel("Mean uncertainty (MW)"); ax[1].set_title("(b) Dropout rate"); ax[1].grid(alpha=0.3); ax[1].legend(frameon=False, fontsize=9, loc="upper left")
    a1 = ax[1].twinx(); a1.spines["right"].set_visible(True); a1.plot(ps, [r["val_CRPS"] for r in P], "D-", color=C_OURS, lw=1.5, label="Validation CRPS"); a1.set_ylabel("Validation CRPS (MW)", color=C_OURS)
    sel = min(P, key=lambda r: r["val_CRPS"]); a1.annotate("selected", (sel["p"], sel["val_CRPS"]), textcoords="offset points", xytext=(10, 6), color=C_OURS, fontsize=9)
    ax[2].plot([c["n"] for c in cs], [c["ECE"] for c in cs], "o-", color=C_EPI, lw=2); ax[2].set_xscale("log")
    ax[2].set_xticks([c["n"] for c in cs]); ax[2].set_xticklabels([str(c["n"]) for c in cs]); ax[2].minorticks_off()
    ax[2].set_xlabel("Calibration-set size $n$"); ax[2].set_ylabel("ECE after split conformal"); ax[2].set_title("(c) Calibration-set size"); ax[2].grid(alpha=0.3); ax[2].set_ylim(0, max(0.05, 1.3 * max(c["ECE"] for c in cs)))
    savefig("sensitivity.png")

# ------------------------------------------------------------------ Fig: reserve cost
if do("reserve"):
    rc = load_json("10_reserve")
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for ax, (name, title) in zip(axs, (("gef", "(a) GEFCom2014"), ("covid", "(b) post-COVID demand"))):
        ks = sorted(rc[name], key=lambda k: int(k)); x = [int(k) for k in ks]
        for pol, col, mk, lab in (("deterministic", "0.4", "o", "Deterministic (no reserve)"), ("fixed", C_ALE, "s", "Fixed reserve"), ("adaptive", C_OURS, "D", "Adaptive (calibrated predictive quantile)")):
            ax.plot(x, [rc[name][k][pol] for k in ks], marker=mk, color=col, lw=2, label=lab)
        ax.set_xscale("log"); ax.set_xticks(x); ax.set_xticklabels([f"{k}:1" for k in x]); ax.set_xlabel(r"Shortfall-to-surplus cost ratio $c_u:c_o$")
        ax.set_ylabel("Mean operating cost (MW, $c_o=1$)"); ax.set_title(title); ax.grid(alpha=0.3); ax.legend(frameon=False, fontsize=8.5)
    savefig("reserve_cost.png")

# ------------------------------------------------------------------ Fig: post-COVID band
if do("band"):
    sc = prepare("covid"); Yv, Yt = sc.Y("val") / 1e3, sc.Y("test") / 1e3
    uv = np.load(ck("ours_covid_seed1_valpred").replace(".pth", ".npz")); ut = np.load(ck("ours_covid_seed1_testpred").replace(".pth", ".npz"))
    Pv, Sv, Pt, St = uv["P"] / 1e3, uv["S"] / 1e3, ut["P"] / 1e3, ut["S"] / 1e3
    q = rolling_scale_90(Yv, Pv, Sv, Yt, Pt, St)
    a, b = 0, 300; t = np.arange(a, b)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.fill_between(t, (Pt - q * St)[a:b], (Pt + q * St)[a:b], color=C_OURS, alpha=0.22, label="90% PI (rolling conformal)")
    ax.plot(t, Yt[a:b], "k-", lw=1.8, label="Ground truth"); ax.plot(t, Pt[a:b], "--", color=C_OURS, lw=1.8, label="Prediction")
    ax.set_xlabel("Time step (hour)"); ax.set_ylabel("Load (MW)"); ax.set_xlim(a, b - 1); ax.grid(alpha=0.25); ax.legend(frameon=False, ncol=3, loc="upper right", fontsize=9)
    savefig("covid_pi_band.png")

# ------------------------------------------------------------------ Fig: data-scarce regime
if do("scarce"):
    rows = load_json("11_scarce"); months = sorted(set(r["months"] for r in rows))
    g = lambda k, m: np.mean([r[k] for r in rows if r["months"] == m]); e = lambda k, m: np.std([r[k] for r in rows if r["months"] == m], ddof=1)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.0))
    x = np.arange(len(months)); lab = [f"{m} mo" if m < 12 else "full year" for m in months]
    ax[0].bar(x - 0.18, [g("PICP90_tot", m) for m in months], 0.36, yerr=[e("PICP90_tot", m) for m in months], color=C_OURS, capsize=3, label="Total (aleatoric + epistemic)")
    ax[0].bar(x + 0.18, [g("PICP90_ale", m) for m in months], 0.36, yerr=[e("PICP90_ale", m) for m in months], color=C_ALE, capsize=3, label="Aleatoric only")
    ax[0].axhline(0.9, color="k", ls="--", lw=1); ax[0].set_xticks(x); ax[0].set_xticklabels(lab); ax[0].set_ylabel("Empirical 90% coverage (raw)"); ax[0].set_title("(a) Coverage of the raw intervals")
    ax[0].set_ylim(0.5, 1.0); ax[0].legend(frameon=False, fontsize=9, loc="lower right"); ax[0].grid(alpha=0.3, axis="y")
    ax[1].errorbar(x, [g("R_epi", m) for m in months], yerr=[e("R_epi", m) for m in months], fmt="o-", color=C_EPI, lw=2, capsize=3)
    ax[1].set_xticks(x); ax[1].set_xticklabels(lab); ax[1].set_ylabel(r"Mean epistemic ratio $R_{\rm epi}$"); ax[1].set_title("(b) Epistemic share vs. training length"); ax[1].grid(alpha=0.3)
    savefig("scarce_regime.png")
