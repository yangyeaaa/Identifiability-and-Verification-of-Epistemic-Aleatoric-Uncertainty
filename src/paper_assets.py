"""Build every table (LaTeX snippets) and figure of the manuscript from results/*.json.
Usage: python paper_assets.py OUTDIR   (writes OUTDIR/tables/*.tex and OUTDIR/figures/*.png)"""
import os, sys, json, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import RES

OUTDIR = sys.argv[1]; TAB = os.path.join(OUTDIR, "tables"); FIG = os.path.join(OUTDIR, "figures")
os.makedirs(TAB, exist_ok=True); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
COL = {"MC": "#1f77b4", "LA-1": "#9ecae1", "LA-2": "#2ca02c", "ENS": "#9467bd", "FREE": "#8c564b", "MVE": "#7f7f7f",
       "CMC": "#ff7f0e", "QR": "#d62728", "PERS": "#bcbd22"}
LAB = {"MC": "MC dropout", "LA-1": "Laplace (last layer)", "LA-2": "Laplace (two layers)", "ENS": "Deep ensemble",
       "FREE": "Free outputs", "MVE": "MVE", "CMC": "Classical MC dropout", "QR": "Quantile regression", "PERS": "Persistence"}
DEC = ("FREE", "MC", "ENS", "LA-1", "LA-2")
UNIT = {"gef": 1.0, "covid": 1e-3}


def J(name):
    p = os.path.join(RES, name + ".json")
    return json.load(open(p)) if os.path.exists(p) else None


def w(name, txt):
    open(os.path.join(TAB, name + ".tex"), "w").write(txt); print("table", name)


def pm(m, s, d=3):
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


def save(name):
    plt.tight_layout(); plt.savefig(os.path.join(FIG, name), dpi=220, bbox_inches="tight"); plt.close(); print("figure", name)


# =============================================================== E3: stability + accuracy/cost
E3 = {n: J(f"e3_main_{n}") for n in ("gef", "covid")}
if all(E3.values()):
    rows = []
    for k in DEC:
        cells = []
        for n in ("gef", "covid"):
            S = E3[n]["summary"][k]; m, s = S["R_epi"]
            cells.append(f"{pm(m, s)} & {s / m:.2f}" if m > 0 else "-- & --")
        rows.append(f"{LAB[k]} & " + " & ".join(cells) + r" \\")
    w("stability", "\n".join(rows))
    body = []
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID demand")):
        u = UNIT[n]; body.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for k in ("PERS", "QR", "CMC", "MVE", "FREE", "MC", "ENS", "LA-1", "LA-2"):
            S = E3[n]["summary"][k]; g = lambda x, d=2, f=1.0: pm(S[x][0] * f, S[x][1] * f, d) if x in S else "--"
            tt = f"{S['train_time'][0]/60:.1f}" if "train_time" in S else "--"
            it = f"{S['infer_time'][0]:.2f}" if "infer_time" in S else "--"
            body.append(f"{LAB[k]} & {g('MAE', 1, u)} & {g('MAPE', 2)} & {g('CRPS', 1, u)} & {g('ECE', 3)} & {g('PICP90', 3)} & "
                        f"{g('ECE_roll', 3)} & {g('PICP90_roll', 3)} & {tt} & {it} \\\\")
        body.append(r"\midrule")
    w("accuracy", "\n".join(body[:-1]))

# =============================================================== E4: reducibility
E4 = J("e4_reducibility")
if E4:
    fr = sorted({r["frac"] for r in E4}); lines = []
    agg = lambda k, f, x: (np.mean([r[x] for r in E4 if r["method"] == k and r["frac"] == f]),
                           np.std([r[x] for r in E4 if r["method"] == k and r["frac"] == f], ddof=1) if k != "ENS" else 0.0)
    for f in fr:
        n = [r["n"] for r in E4 if r["frac"] == f][0]; cells = []
        for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
            e = agg(k, f, "epi"); cells.append(f"${e[0]:.1f}$" if k == "ENS" else f"${e[0]:.1f}\\pm{e[1]:.1f}$")
        a = agg("LA-2", f, "ale")
        lines.append(f"{int(round(f*100))}\\,\\% & {n} & " + " & ".join(cells) + f" & ${a[0]:.1f}\\pm{a[1]:.1f}$ \\\\")
    ratio = {k: agg(k, fr[0], "epi")[0] / agg(k, fr[-1], "epi")[0] for k in ("FREE", "MC", "ENS", "LA-1", "LA-2")}
    lines.append(r"\midrule")
    lines.append(f"\\multicolumn{{2}}{{l}}{{Ratio 5\\,\\%/100\\,\\%}} & " + " & ".join(f"{ratio[k]:.2f}" for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"))
                 + f" & {agg('LA-2', fr[0], 'ale')[0]/agg('LA-2', fr[-1], 'ale')[0]:.2f} \\\\")
    Rr = lambda k, f: np.mean([r["R"] for r in E4 if r["method"] == k and r["frac"] == f])
    lines.append(f"\\multicolumn{{2}}{{l}}{{$R_{{\\mathrm{{epi}}}}$, 25\\,\\% $\\to$ 100\\,\\%}} & "
                 + " & ".join(f"{Rr(k, 0.25):.3f}$\\to${Rr(k, 1.0):.3f}" for k in ("FREE", "MC", "ENS", "LA-1", "LA-2")) + " & -- \\\\")
    w("reducibility", "\n".join(lines))
    json.dump(ratio, open(os.path.join(TAB, "reducibility_ratio.json"), "w"))

# =============================================================== E5: support probe
E5 = J("e5_support")
if E5:
    lines = []
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID demand")):
        lines.append(f"\\multicolumn{{6}}{{l}}{{\\textit{{{lab}}}}} \\\\")
        for k in DEC:
            m = E5[n][k]["mean"]; ci = E5[n][k]["per_seed"][0]
            lines.append(f"{LAB[k]} & {pm(*m['rho'], 2)} & {pm(*m['rho_x'], 2)} & {pm(*m['within'], 2)} & {pm(*m['partial'], 2)} & ${m['far_near'][0]:.2f}$ \\\\")
        lines.append(r"\midrule")
    w("support", "\n".join(lines[:-1]))

# =============================================================== E6 + E10: out-of-support (both early-stopping rules)
E6 = J("e6_ood"); E10 = J("e10_window_validation")
WIN = ["recent 1 month", "recent 2 months", "recent 3 months"]


def ood_rows(rows, c, k):
    return [r for r in rows if r["cond"] == c and r["method"] == k]


if E6:
    lines = []
    plan = [(c, "val", E6) for c in WIN]
    if E10 and E10.get("real"):
        plan = [x for c in WIN for x in ((c, "val", E6), (c, "win", E10["real"]))]
    plan += [("temperature hold-out", "val", E6), ("full year", "val", E6)]
    for c, rule, src in plan:
        rr = [r for r in src if r["cond"] == c]
        if not rr: continue
        fo = rr[0]["frac_out"]; cells = []
        for k in ("MC", "ENS", "LA-1", "LA-2"):
            q = ood_rows(src, c, k); g = lambda x: np.nanmean([r.get(x, np.nan) for r in q])
            dc = 100 * (g("PICP90_tot_out") - g("PICP90_ale_out"))
            cells.append(f"{g('R'):.3f} & {g('epi'):.0f} & " + (f"{dc:+.1f}" if np.isfinite(dc) else "--"))
        ale = np.mean([r["ale"] for r in rr if r["method"] == "LA-2"])
        stop = {"val": "val.\\ year", "win": "window"}[rule]
        lines.append(f"{c[0].upper()+c[1:]} & {stop} & {rr[0]['n']} & {100*fo:.0f} & {ale:.0f} & " + " & ".join(cells) + r" \\")
    w("ood", "\n".join(lines))


# =============================================================== E7: semi-synthetic
def t4(r):
    """Per-seed T4 statistics; shares are ratios of means, recall/balanced accuracy of epistemic-dominant hours."""
    e2 = r["rmse_mean"] ** 2; s2 = e2 * (1 - r["true_R"]) / r["true_R"]; se2 = r["rms_epi"] ** 2; sa2 = r["ale_ratio"] * s2
    b = r["true_epi_dom"]; rec = 1 - r["miss_epi"] if b > 0 else np.nan
    spec = (r["action_agree"] - b * (rec if b > 0 else 0)) / (1 - b) if b < 1 else np.nan
    return dict(true=r["true_R"], rep=se2 / (se2 + sa2), ratio=r["epi_ratio"], cov=r["epi_cov90"], rho=r["epi_rho"],
                ale=r["ale_ratio"], recall=rec, ba=0.5 * (rec + spec) if b > 0 else np.nan, base=b)


def t4agg(rows, k):
    q = [t4(r) for r in rows if r["method"] == k]
    return {x: float(np.nanmean([d[x] for d in q])) for x in q[0]}


E7 = {n: J(f"e7_semisynthetic_{n}") for n in ("gef", "covid")}
E7 = {n: v for n, v in E7.items() if v and len(v) == 4}
def t4agg10(n, k, variant="NLL"):
    q = [r for r in E10["semi"] if r["dataset"] == n and r["method"] == k and r["variant"] == variant]
    if not q: return None
    g = lambda x: float(np.nanmean([r[x] for r in q]))
    return dict(true=g("true_R"), rep=g("R_means"), ratio=g("epi_ratio"), cov=g("epi_cov90"), rho=g("epi_rho"), ale=g("ale_ratio"),
                recall=g("recall"), ba=g("ba"), base=g("base"), rmse=g("rmse"))


HAS10 = {n: bool(E10 and E10.get("semi") and any(r["dataset"] == n for r in E10["semi"])) for n in ("gef", "covid")}


def t4conds(n):
    """(label, aggregator) per training condition; the one-month condition under both early-stopping rules."""
    out = []
    for c, D in E7[n].items():
        lab = (c[0].upper() + c[1:]).replace("%", "\\,\\%")
        if c == "recent 1 month":
            lab = "\\makecell[l]{Recent 1 month\\\\(val.\\ year stop)}"
        out.append((c, lab, (lambda k, D=D: t4agg(D["rows"], k))))
        if c == "recent 1 month" and HAS10[n]:
            out.append(("recent 1 month (window)", "\\makecell[l]{Recent 1 month\\\\(window stop)}", (lambda k, n=n: t4agg10(n, k))))
    return out


if E7:
    for n in E7:
        lines = []
        for c, cl, agg in t4conds(n):
            first = True
            for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
                a = agg(k)
                cname = f"\\multirow{{5}}{{*}}{{{cl}}}" if first else ""
                first = False
                rec = f"{a['recall']:.2f}" if np.isfinite(a["recall"]) else "--"; ba = f"{a['ba']:.2f}" if np.isfinite(a["ba"]) else "--"
                lines.append(f"{cname} & {LAB[k]} & {a['true']:.2f} & {a['rep']:.3f} & {a['ratio']:.2f} & {a['cov']:.2f} & {a['rho']:+.2f} & "
                             f"{a['ale']:.2f} & {100*a['base']:.0f} & {rec} & {ba} \\\\")
            lines.append(r"\midrule")
        w(f"semisynthetic_{n}", "\n".join(lines[:-1]))

    # figure: reported vs true R_epi, calibration ratio
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.0))
    mk = {"gef": "o", "covid": "s"}
    for n in E7:
        for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
            xs, ys = [], []
            for c, cl, agg in t4conds(n):
                a = agg(k); xs.append(a["true"]); ys.append(a["rep"])
            ax[0].scatter(xs, ys, color=COL[k], marker=mk[n], s=40, label=LAB[k] if n == "gef" else None, zorder=3)
    ax[0].plot([0, 1], [0, 1], "k--", lw=1); ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)
    ax[0].set_xlabel("True epistemic share")
    ax[0].set_ylabel(r"Reported $R_{\rm epi}$"); ax[0].set_title("(a) Reported vs. true epistemic share")
    ax[0].legend(fontsize=8, loc="upper left", bbox_to_anchor=(0.42, 0.98), frameon=True, framealpha=0.95, edgecolor="none"); ax[0].text(0.58, 0.42, "circles: GEFCom2014\nsquares: post-COVID", ha="left", va="top", fontsize=8, transform=ax[0].transAxes)
    cg = t4conds("gef"); x = np.arange(len(cg)); wd = 0.16
    tick = {"uniform 5%": "5%", "uniform 25%": "25%", "uniform 100%": "100%", "recent 1 month": "1 mo\n(val. year)",
            "recent 1 month (window)": "1 mo\n(window)"}
    for j, k in enumerate(("FREE", "MC", "ENS", "LA-1", "LA-2")):
        vals = [agg(k)["ratio"] for c, cl, agg in cg]
        ax[1].bar(x + (j - 2) * wd, vals, wd, color=COL[k], label=LAB[k])
    ax[1].axhline(1, color="k", ls="--", lw=1); ax[1].set_yscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels([tick[c] for c, cl, agg in cg], fontsize=8.5)
    ax[1].set_ylabel(r"$\mathbb{E}[(\hat\mu-f^\star)^2]\,/\,\mathbb{E}[\hat\sigma^2_{\rm epi}]$"); ax[1].set_title("(b) Epistemic calibration ratio, GEFCom2014 (1 = calibrated)"); ax[1].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    save("semisynthetic.png")

# =============================================================== scorecard (thresholds stated in the caption of the table)
SYM = {2: r"$\checkmark$", 1: r"$\sim$", 0: r"$\times$", None: "--"}
if all(E3.values()) and E4 and E5 and E6 and len(E7) == 2 and all(HAS10.values()):
    def band(x, lo, hi):
        return lo <= x <= hi
    score, why = {}, {}
    for k in DEC:
        sc_ = {}
        cv = max(E3[n]["summary"][k]["R_epi"][1] / E3[n]["summary"][k]["R_epi"][0] for n in ("gef", "covid"))
        sc_["T1"] = 2 if cv <= 0.2 else (0 if cv > 0.5 else 1)
        fr = sorted({r["frac"] for r in E4}); q = lambda f, x: [r[x] for r in E4 if r["method"] == k and r["frac"] == f]
        m = [np.mean(q(f, "epi")) for f in fr]; sd = max(np.std(q(f, "epi"), ddof=1) if len(q(f, "epi")) > 1 else 0 for f in fr)
        mono = all(m[i + 1] < m[i] for i in range(len(m) - 1)); dist = (m[0] - m[-1]) > 2 * sd
        i25 = fr.index(0.25); Rf = [np.mean(q(f, "R")) for f in fr]
        faster = Rf[-1] <= 0.9 * Rf[i25]          # epistemic share falls by at least 10 % from 25 % to 100 % of the data
        sc_["T2"] = 0 if not (mono and dist) else (2 if faster else 1)
        lv = []
        for n in ("gef", "covid"):
            per = E5[n][k]["per_seed"]
            if all(p_["ci_partial"][0] > 0 for p_ in per): lv.append(2)
            elif any(p_["ci_partial"][1] < 0 for p_ in per): lv.append(0)
            else: lv.append(1)
        sc_["T3d"] = min(lv)
        if k == "FREE":
            sc_["T3o"] = None
        else:
            # every shorter window must exceed the full year in both sigma_epi and R_epi; fail if sigma_epi with one
            # month does not exceed the full-year value; evaluated under each early-stopping rule, the lower level counts
            full = ood_rows(E6, "full year", k); ef, Rf = np.mean([r["epi"] for r in full]), np.mean([r["R"] for r in full])
            lv = []
            for src in [E6] + ([E10["real"]] if E10 and E10.get("real") else []):
                e = [np.mean([r["epi"] for r in ood_rows(src, c, k)]) for c in WIN]
                R = [np.mean([r["R"] for r in ood_rows(src, c, k)]) for c in WIN]
                lv.append(2 if (min(e) > ef and min(R) > Rf) else (0 if e[0] <= ef else 1))
            sc_["T3o"] = min(lv)
        for key, cond in (("T4i", "uniform 100%"), ("T4x", "recent 1 month")):
            lv = []
            for n in ("gef", "covid"):
                rs = [t4agg(E7[n][cond]["rows"], k)["ratio"]]
                if key == "T4x" and HAS10[n]:
                    rs.append(t4agg10(n, k)["ratio"])
                for r_ in rs:
                    lv.append(2 if band(r_, 0.4, 2.5) else (1 if band(r_, 0.2, 5.0) else 0))
            sc_[key] = min(lv)
        E14 = [J(f"e14_timescale_{n}") for n in ("gef", "covid")]
        if all(E14):
            # decision time scale (Proposition 6): pooled one-week AUROC of the window ratio, both datasets, both stopping rules
            aucs = [float(np.nanmean([r["auc"] for r in E["pairs"] if r["method"] == k and r["H"] == 168 and r["out_state"] == st]))
                    for E in E14 for st in ("1mo val", "1mo win")]
            sc_["T4d"] = 2 if min(aucs) >= 0.9 else (0 if min(aucs) <= 0.6 else 1)
        bas = {n: [t4agg(D["rows"], k)["ba"] for D in E7[n].values()] + [t4agg10(n, k)["ba"]] for n in E7}
        allba = [b for v in bas.values() for b in v if np.isfinite(b)]
        sc_["T4s"] = 2 if min(allba) >= 0.75 else (1 if all(max(v) >= 0.6 for v in bas.values()) else 0)
        score[k] = sc_
    lines = []
    for k in DEC:
        lab = LAB[k] + ("$^{a}$" if k == "FREE" else "")
        # T4s (hour-level dominant source) is kept in scorecard.json but not shown in Table IV: even an oracle split
        # cannot pass it (Remark 2; oracle_ceiling.py)
        cols = ("T1", "T2", "T3d", "T3o", "T4i", "T4x") + (("T4d",) if "T4d" in score[k] else ())
        lines.append(f"{lab} & " + " & ".join(SYM[score[k][c]] for c in cols) + r" \\")
    w("scorecard", "\n".join(lines))
    json.dump(score, open(os.path.join(TAB, "scorecard.json"), "w"), indent=1)

# =============================================================== E9 + E10: fairness (MC at p = 0.1) and beta-NLL remedy
E9 = J("e9_fairness_remedy")
if E9 and E7 and E9.get("semi") and "ba" in E9["semi"][0] and all(HAS10.values()):
    lines = []
    VL = {"base": "NLL, $p=0.2$", "MC p=0.1": "NLL, $p=0.1$", "beta-NLL": "$\\beta$-NLL, $p=0.2$"}
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID")):
        for c, cl in (("uniform 100%", "full year"), ("recent 1 month", "1 month, val.\\ year stop"), ("recent 1 month (window)", "1 month, window stop")):
            rows = []
            for k, var in (("MC", "base"), ("MC", "MC p=0.1"), ("MC", "beta-NLL"), ("LA-2", "base"), ("LA-2", "beta-NLL")):
                if c.endswith("(window)"):
                    a = t4agg10(n, k, "NLL" if var == "base" else var)
                    if a is None: continue
                    rm = a["rmse"]
                elif var == "base":
                    a = t4agg(E7[n][c]["rows"], k)
                    rm = np.mean([r["rmse_mean"] for r in E7[n][c]["rows"] if r["method"] == k])
                else:
                    q = [r for r in E9["semi"] if r["dataset"] == n and r["cond"] == c and r["variant"] == var and r["method"] == k]
                    if not q: continue
                    g = lambda x: float(np.nanmean([r[x] for r in q]))
                    a = dict(true=g("true_R"), rep=g("R_means"), ratio=g("epi_ratio"), cov=g("epi_cov90"), ale=g("ale_ratio"),
                             recall=g("recall"), ba=g("ba")); rm = g("rmse")
                rows.append(f"{LAB[k]} & {VL[var]} & {rm*UNIT[n]:.1f} & {a['true']:.2f} & {a['rep']:.3f} & {a['ratio']:.2f} & {a['cov']:.2f} & "
                            f"{a['ale']:.2f} & {a['recall']:.2f} & {a['ba']:.2f} \\\\")
            if not rows: continue
            rows[0] = f"\\multirow{{{len(rows)}}}{{*}}{{\\makecell[l]{{{lab}\\\\{cl}}}}} & " + rows[0]
            lines += [rows[0]] + [" & " + r for r in rows[1:]] + [r"\midrule"]
    w("remedy", "\n".join(lines[:-1]))
    acc = E9["covid_mc_p10"]
    json.dump({x: [float(np.mean([r[x] for r in acc])), float(np.std([r[x] for r in acc], ddof=1))] for x in ("MAE", "CRPS", "ECE", "ECE_roll", "R")},
              open(os.path.join(TAB, "covid_mc_p10.json"), "w"))

# =============================================================== E12: what the variance head estimates
E12 = J("e12_absorption")
if E12:
    lines = []
    CL = {"uniform 5%": "Uniform 5\\,\\%", "uniform 25%": "Uniform 25\\,\\%", "uniform 100%": "Uniform 100\\,\\%",
          "recent 1 month": "1 month, val.\\ year stop", "recent 1 month (window)": "1 month, window stop"}
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID")):
        conds = [c for c in CL if any(r["dataset"] == n and r["cond"] == c for r in E12)]
        for i, c in enumerate(conds):
            rr = [r for r in E12 if r["dataset"] == n and r["cond"] == c]; g = lambda x: np.mean([r[x] for r in rr])
            first = f"\\multirow{{{len(conds)}}}{{*}}{{{lab}}}" if i == 0 else ""
            lines.append(f"{first} & {CL[c]} & {g('lvl_noise'):.2f} & {g('lvl_total'):.2f} & {g('rho_noise'):+.2f} & {g('rho_total'):+.2f} \\\\")
        lines.append(r"\midrule")
    w("absorption", "\n".join(lines[:-1]))

# =============================================================== E11: learning-curve decomposition
E11 = {n: J(f"e11_lcd_{n}") for n in ("gef", "covid")}
if any(E11.values()):
    def fmt(x):
        if x < 100: return f"{x:.2f}"
        if x < 1e5: return f"{x:.0f}"
        return f"${x:.0e}".replace("e+0", "\\times10^{").replace("e+", "\\times10^{") + "}$"
    CL = {"uniform 100%": "Uniform 100\\,\\%", "uniform 25%": "Uniform 25\\,\\%", "uniform 5%": "Uniform 5\\,\\%",
          "recent 1 month (val. year stop)": "1 month, val.\\ year stop", "recent 1 month (window stop)": "1 month, window stop"}
    E7c = {"uniform 100%": "uniform 100%", "uniform 25%": "uniform 25%", "uniform 5%": "uniform 5%", "recent 1 month (val. year stop)": "recent 1 month"}
    lines = []
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID")):
        if not E11[n]: continue
        conds = [c for c in CL if any(r["cond"] == c for r in E11[n])]
        for i, c in enumerate(conds):
            q = [r for r in E11[n] if r["cond"] == c and r["method"] == "LCD"]; g = lambda x: float(np.nanmean([r[x] for r in q]))
            if c in E7c:
                ref = {k: t4agg(E7[n][E7c[c]]["rows"], k)["ratio"] for k in ("MC", "LA-2")}
            else:
                ref = {k: t4agg10(n, k)["ratio"] for k in ("MC", "LA-2")}
            first = f"\\multirow{{{len(conds)}}}{{*}}{{{lab}}}" if i == 0 else ""
            lines.append(f"{first} & {CL[c]} & {g('true_R'):.2f} & {g('R_means'):.3f} & {fmt(g('epi_ratio'))} & {g('epi_cov90'):.2f} & "
                         f"{fmt(g('ale_ratio'))} & {g('ba'):.2f} & {ref['MC']:.2f} / {ref['LA-2']:.2f} \\\\")
        lines.append(r"\midrule")
    w("lcd", "\n".join(lines[:-1]))

# =============================================================== figure: behavioural verification
if E4 and E5 and E6:
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    fr = sorted({r["frac"] for r in E4}); pct = np.array(fr) * 100
    for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
        m = [np.mean([r["epi"] for r in E4 if r["method"] == k and r["frac"] == f]) for f in fr]
        s = [np.std([r["epi"] for r in E4 if r["method"] == k and r["frac"] == f], ddof=1) if k != "ENS" else 0 for f in fr]
        ax[0].errorbar(pct, m, yerr=s, fmt="o-", color=COL[k], capsize=3, lw=1.8, label=LAB[k])
    la2 = [np.mean([r["epi"] for r in E4 if r["method"] == "LA-2" and r["frac"] == f]) for f in fr]
    ax[0].plot(pct, la2[-1] * np.sqrt(fr[-1] / np.array(fr)), ":", color="0.3", label=r"$1/\sqrt{n}$ (anchored to LA-2)")
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xticks(pct); ax[0].set_xticklabels([f"{p:g}" for p in pct]); ax[0].minorticks_off()
    ax[0].set_xlabel("Training data used (%)"); ax[0].set_ylabel(r"Mean $\hat\sigma_{\rm epi}$ (MW)"); ax[0].set_title("(a) T2 reducibility (GEFCom2014)"); ax[0].legend(fontsize=7.5)
    cur = E5["gef"]["curves"]
    for k in DEC:
        v = np.array(cur[k]["dist_decile"]); ax[1].plot(range(1, 11), v / v[0], "o-", color=COL[k], lw=1.8, label=LAB[k])
    ax[1].set_xticks(range(1, 11)); ax[1].set_xlabel("Decile of distance to training inputs"); ax[1].set_ylabel(r"Mean $\hat\sigma_{\rm epi}$ / first decile")
    ax[1].set_title("(b) T3 distance probe (GEFCom2014, seed 0)"); ax[1].legend(fontsize=7.5)
    conds = ["recent 1 month", "recent 2 months", "recent 3 months", "full year"]; x = np.arange(len(conds))
    from matplotlib.lines import Line2D
    for k in ("MC", "ENS", "LA-1", "LA-2"):
        v = [np.mean([r["R"] for r in E6 if r["cond"] == c and r["method"] == k]) for c in conds]
        ax[2].plot(x, v, "o--", color=COL[k], lw=1.2, alpha=0.7, ms=4)
        if E10 and E10.get("real"):
            v2 = [np.mean([r["R"] for r in E10["real"] if r["cond"] == c and r["method"] == k]) for c in conds[:3]] + [v[-1]]
            ax[2].plot(x, v2, "s-", color=COL[k], lw=1.8, ms=4)
    hs = [Line2D([0], [0], color=COL[k], lw=2) for k in ("MC", "ENS", "LA-1", "LA-2")]
    ls_ = ["MC dropout", "Deep ensemble", "Laplace (last layer)", "Laplace (two layers)"]
    if E10 and E10.get("real"):
        hs += [Line2D([0], [0], color="0.3", ls=(0, (4, 2)), lw=1.2, marker="o", ms=4), Line2D([0], [0], color="0.3", ls="-", marker="s", ms=4)]
        ls_ += ["stopped on validation year", "stopped on window"]
    ax[2].set_xticks(x); ax[2].set_xticklabels(["1 mo", "2 mo", "3 mo", "1 yr"]); ax[2].set_xlabel("Training window (most recent)")
    ax[2].set_ylabel(r"Mean reported $R_{\rm epi}$"); ax[2].set_yscale("log"); ax[2].set_title("(c) T3 out of support: seasonal extrapolation")
    ax[2].legend(hs, ls_, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    save("verification.png")

# =============================================================== E2 + E8: sensitivity figure
E2 = J("e2_dropout_sweep"); E8 = J("e8_passes")
if E2:
    S = E2["summary"]; ps = sorted(float(p) for p in S)
    fig, ax = plt.subplots(1, 2 if E8 else 1, figsize=(10.5 if E8 else 5.5, 4.0)); ax = np.atleast_1d(ax)
    for k in ("MC", "LA-1", "LA-2"):
        ax[0].errorbar(ps, [S[str(p)][k]["R_epi"][0] for p in ps], yerr=[S[str(p)][k]["R_epi"][1] for p in ps], fmt="o-", color=COL[k], capsize=3, label=LAB[k] + r": $R_{\rm epi}$")
    ax[0].set_yscale("log"); ax[0].set_xlabel("Dropout rate $p$"); ax[0].set_ylabel(r"Test $R_{\rm epi}$ (mean $\pm$ s.d., 3 seeds)")
    a2 = ax[0].twinx(); a2.spines["right"].set_visible(True)
    a2.errorbar(ps, [S[str(p)]["MVE"]["val_CRPS"][0] for p in ps], yerr=[S[str(p)]["MVE"]["val_CRPS"][1] for p in ps], fmt="s--", color="0.4", capsize=3, label="Validation CRPS (MW)")
    a2.set_ylabel("Validation CRPS (MW)", color="0.4")
    h1, l1 = ax[0].get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels(); ax[0].legend(h1 + h2, l1 + l2, fontsize=7.5, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax[0].set_title("(a) Dropout rate: validation score vs. epistemic share")
    if E8:
        Ms = [r["M"] for r in E8]; ax[1].plot(Ms, [r["epi"] for r in E8], "o-", color=COL["MC"]); ax[1].set_xscale("log"); ax[1].set_xticks(Ms); ax[1].set_xticklabels(Ms); ax[1].minorticks_off()
        ax[1].set_xlabel("MC passes $M$"); ax[1].set_ylabel(r"Mean $\hat\sigma_{\rm epi}$ (MW)", color=COL["MC"]); ax[1].set_title("(b) MC dropout: number of passes")
        b2 = ax[1].twinx(); b2.spines["right"].set_visible(True); b2.plot(Ms, [r["MAE"] for r in E8], "s--", color="0.4"); b2.set_ylabel("MAE of the averaged mean (MW)", color="0.4")
    save("sensitivity.png")
    lines = []
    for p in ps:
        Sp = S[str(p)]
        lines.append(f"{p:.2f} & {pm(*Sp['MVE']['val_CRPS'], 2)} & {pm(*Sp['MVE']['CRPS'], 2)} & {pm(*Sp['MC']['R_epi'], 3)} & {pm(*Sp['LA-1']['R_epi'], 4)} & {pm(*Sp['LA-2']['R_epi'], 3)} \\\\")
    w("dropout", "\n".join(lines))
print("done")

# =============================================================== calibration curves (E3 predictions, seed 0)
from metrics import coverage_curve, ece_from_curve, split_conformal, rolling_conformal, LEVELS
from common import ck
from backbone import prepare
if all(E3.values()):
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.3))
    for ax, (n, title) in zip(axs, (("gef", "(a) GEFCom2014"), ("covid", "(b) post-COVID demand"))):
        pr = np.load(ck(f"e3_preds_{n}").replace(".pth", ".npz")); sc = prepare(n); Yv, Yt = sc.Y("val"), sc.Y("test")
        ax.plot([0.5, 0.95], [0.5, 0.95], "k--", lw=1, label="Ideal")
        for k, ls in (("LA-2", "-"), ("MC", "--"), ("ENS", ":")):
            g = lambda sp, q: pr[f"{k}_s0__{sp}__{q}"]
            raw = coverage_curve(Yt, g("test", "P"), g("test", "S"))
            ro = rolling_conformal(Yv, g("val", "P"), g("val", "S"), Yt, g("test", "P"), g("test", "S"), W=24 * 90, lag=24)
            ax.plot(LEVELS, raw, ls, color=COL[k], lw=1.6, alpha=0.55, label=f"{LAB[k]}, raw (ECE {ece_from_curve(raw):.3f})")
            ax.plot(LEVELS, ro["curve"], ls, color=COL[k], lw=2.2, marker="o", ms=3.5, label=f"{LAB[k]}, rolling (ECE {ro['ECE']:.3f})")
        sp = split_conformal(Yv, pr["LA-2_s0__val__P"], pr["LA-2_s0__val__S"], Yt, pr["LA-2_s0__test__P"], pr["LA-2_s0__test__S"])
        ax.plot(LEVELS, sp["curve"], "-", color="0.55", lw=1.4, marker="s", ms=3, label=f"{LAB['LA-2']}, split (ECE {sp['ECE']:.3f})")
        ax.set_xlabel("Nominal level"); ax.set_ylabel("Empirical coverage"); ax.set_title(title); ax.set_xlim(0.48, 0.97); ax.set_ylim(0.4, 1.0)
        ax.grid(alpha=0.25); ax.legend(fontsize=7, loc="upper left")
    save("calibration.png")
