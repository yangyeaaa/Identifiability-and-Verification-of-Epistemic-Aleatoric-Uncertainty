"""Compact tables for the IEEE PES Transactions version (main paper).
Usage: python ieee_assets.py OUTDIR   (writes OUTDIR/tables/ieee_*.tex)
The supplementary tables are the full tables produced by paper_assets.py."""
import os, sys, json, numpy as np
from common import RES

OUT = os.path.join(sys.argv[1], "tables"); os.makedirs(OUT, exist_ok=True)
J = lambda n: json.load(open(os.path.join(RES, n + ".json")))
UNIT = {"gef": 1.0, "covid": 1e-3}


def w(name, txt):
    open(os.path.join(OUT, name + ".tex"), "w").write(txt); print("table", name)


def t4(r):
    e2 = r["rmse_mean"] ** 2; s2 = e2 * (1 - r["true_R"]) / r["true_R"]
    b = r["true_epi_dom"]; rec = 1 - r["miss_epi"] if b > 0 else np.nan
    spec = (r["action_agree"] - b * (rec if b > 0 else 0)) / (1 - b) if b < 1 else np.nan
    return dict(true=r["true_R"], ratio=r["epi_ratio"], ale=r["ale_ratio"], ba=0.5 * (rec + spec) if b > 0 else np.nan)


E7 = {n: J(f"e7_semisynthetic_{n}") for n in ("gef", "covid")}
E10 = J("e10_window_validation"); E11 = {n: J(f"e11_lcd_{n}") for n in ("gef", "covid")}


def agg7(n, c, k):
    q = [t4(r) for r in E7[n][c]["rows"] if r["method"] == k]
    return {x: float(np.nanmean([d[x] for d in q])) for x in q[0]}


def agg10(n, k):
    q = [r for r in E10["semi"] if r["dataset"] == n and r["method"] == k and r["variant"] == "NLL"]
    return dict(true=np.mean([r["true_R"] for r in q]), ratio=np.mean([r["epi_ratio"] for r in q]),
                ale=np.mean([r["ale_ratio"] for r in q]), ba=np.mean([r["ba"] for r in q]))


def agg11(n, c):
    q = [r for r in E11[n] if r["cond"] == c and r["method"] == "LCD"]
    return dict(true=np.mean([r["true_R"] for r in q]), ratio=np.mean([r["epi_ratio"] for r in q]),
                ale=np.mean([r["ale_ratio"] for r in q]), ba=np.mean([r["ba"] for r in q]))


def fmt(x):
    if not np.isfinite(x): return "--"
    if x >= 1e4: return "div."
    if x >= 100: return f"{x:.0f}"
    if x >= 10: return f"{x:.1f}"
    return f"{x:.2f}"


CONDS = [("uniform 100%", "uniform 100%"), ("uniform 5%", "uniform 5%"), ("recent 1 month", "recent 1 month (val. year stop)"),
         ("window", "recent 1 month (window stop)")]
cols = [(n, c7, c11) for n in ("gef", "covid") for c7, c11 in CONDS]


def cell(n, c7, c11, k):
    if k == "LCD":
        return agg11(n, c11)
    return agg10(n, k) if c7 == "window" else agg7(n, c7, k)


lines = []
lines.append("True epistemic share & " + " & ".join(f"{cell(n, c7, c11, 'LA-2')['true']:.2f}" for n, c7, c11 in cols) + r" \\")
lines.append(r"\midrule")
LAB = {"FREE": "Free outputs", "MC": "MC dropout", "ENS": "Deep ensemble", "LA-1": "Laplace, last layer", "LA-2": "Laplace, two layers", "LCD": "Learning curves"}
for k in ("FREE", "MC", "ENS", "LA-1", "LA-2", "LCD"):
    lines.append(f"{LAB[k]} & " + " & ".join(fmt(cell(n, c7, c11, k)["ratio"]) for n, c7, c11 in cols) + r" \\")
lines.append(r"\midrule")
lines.append("Aleatoric ratio, head & " + " & ".join(fmt(cell(n, c7, c11, 'LA-2')["ale"]) for n, c7, c11 in cols) + r" \\")
lines.append("Aleatoric ratio, LCD & " + " & ".join(fmt(cell(n, c7, c11, 'LCD')["ale"]) for n, c7, c11 in cols) + r" \\")
w("ieee_t4", "\n".join(lines))

# balanced-accuracy range for the text
ba = [cell(n, c7, c11, k)["ba"] for n, c7, c11 in cols for k in ("FREE", "MC", "ENS", "LA-1", "LA-2", "LCD")]
print("balanced accuracy range", np.nanmin(ba), np.nanmax(ba))

# ---- accuracy, calibration and reserve table
E3 = {n: J(f"e3_main_{n}") for n in ("gef", "covid")}; E13 = J("e13_reserve")
LAB2 = {"PERS": "Persistence", "QR": "Quantile regression", "MVE": "Single network (MVE)", "MC": "MC dropout",
        "ENS": "Deep ensemble", "LA-2": "Laplace, two layers"}
lines = []
for k in ("PERS", "QR", "MVE", "MC", "ENS", "LA-2"):
    cells = []
    for n in ("gef", "covid"):
        S = E3[n]["summary"][k]; u = UNIT[n]
        crps = f"{S['CRPS'][0]*u:.1f}"
        ece = f"{S['ECE_roll'][0]:.3f}" if "ECE_roll" in S else "--"
        rr = [r for r in E13 if r["dataset"] == n and r["method"] == k]
        if rr:
            m = lambda x: np.mean([r[x] for r in rr])
            cov = f"{m('up0.99_cal'):.3f} / {m('up0.99_cal_peak'):.3f}"
            red = f"{100*(1-m('cost20_tot')/m('cost20_fixed')):.1f}"
            # MVE has no epistemic term: aleatoric-only and total reserves coincide by construction
            dale = "--" if k == "MVE" else f"${100*(m('cost20_ale')-m('cost20_tot'))/m('cost20_fixed'):+.1f}$"
        else:
            cov, red, dale = "--", "--", "--"
        cells += [crps, ece, cov, red, dale]
    lines.append(f"{LAB2[k]} & " + " & ".join(cells) + r" \\")
w("ieee_accuracy", "\n".join(lines))


# ---- supplementary: full accuracy table with split conformal calibration
def pm(m, s, d=3):
    return f"${m:.{d}f}\\pm{s:.{d}f}$"


LAB3 = {"PERS": "Persistence", "QR": "Quantile regression", "CMC": "Classical MC dropout", "MVE": "MVE",
        "FREE": "Free outputs", "MC": "MC dropout", "ENS": "Deep ensemble", "LA-1": "Laplace (last layer)",
        "LA-2": "Laplace (two layers)"}
body = []
for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID demand")):
    u = UNIT[n]; body.append(f"\\multicolumn{{12}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for k in ("PERS", "QR", "CMC", "MVE", "FREE", "MC", "ENS", "LA-1", "LA-2"):
        S = E3[n]["summary"][k]; g = lambda x, d=2, f=1.0: pm(S[x][0] * f, S[x][1] * f, d) if x in S else "--"
        tt = f"{S['train_time'][0]/60:.1f}" if "train_time" in S else "--"
        it = f"{S['infer_time'][0]:.2f}" if "infer_time" in S else "--"
        body.append(f"{LAB3[k]} & {g('MAE', 1, u)} & {g('MAPE', 2)} & {g('CRPS', 1, u)} & {g('ECE', 3)} & {g('PICP90', 3)} & "
                    f"{g('ECE_split', 3)} & {g('PICP90_split', 3)} & {g('ECE_roll', 3)} & {g('PICP90_roll', 3)} & {tt} & {it} \\\\")
    body.append(r"\midrule")
w("ieee_accuracy_supp", "\n".join(body[:-1]))

# ---- supplementary: full reserve evaluation (E13)
LAB4 = {"MVE": "MVE", "MC": "MC dropout", "ENS": "Deep ensemble", "LA-2": "Laplace (two layers)"}
body = []
for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID demand")):
    u = UNIT[n]; body.append(f"\\multicolumn{{13}}{{l}}{{\\textit{{{lab}}}}} \\\\")
    for k in ("MVE", "MC", "ENS", "LA-2"):
        rr = [r for r in E13 if r["dataset"] == n and r["method"] == k]
        m = lambda x: np.mean([r[x] for r in rr])
        c = [f"{m('up0.99_raw'):.3f}", f"{m('up0.99_raw_peak'):.3f}", f"{m('up0.99_cal'):.3f}", f"{m('up0.99_cal_peak'):.3f}",
             f"{m('up0.995_cal'):.3f}", f"{m('up0.995_cal_peak'):.3f}"]
        for cu in (10, 20):
            c += [f"{m(f'cost{cu}_fixed')*u:.1f}", f"{m(f'cost{cu}_tot')*u:.1f}",
                  "--" if k == "MVE" else f"{m(f'cost{cu}_ale')*u:.1f}"]
        body.append(f"{LAB4[k]} & " + " & ".join(c) + r" \\")
    body.append(r"\midrule")
w("ieee_reserve_supp", "\n".join(body[:-1]))


# ---- decision time scale (E14, Proposition 6): main table and supplementary tables
E14 = {n: J(f"e14_timescale_{n}") for n in ("gef", "covid") if os.path.exists(os.path.join(RES, f"e14_timescale_{n}.json"))}
if len(E14) == 2:
    LAB5 = {"FREE": "Free outputs", "MC": "MC dropout", "ENS": "Deep ensemble", "LA-1": "Laplace, last layer", "LA-2": "Laplace, two layers"}
    SC = (("hour", 1), ("day", 24), ("week", 168), ("30 days", 720))

    def pr(n, st, k, H):
        return [r for r in E14[n]["pairs"] if r["out_state"] == st and r["method"] == k and r["H"] == H]

    def mean(rows, x):
        return float(np.nanmean([r[x] for r in rows]))

    lines = []
    for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
        cells = []
        for n in ("gef", "covid"):
            cells += [f"{mean(pr(n, '1mo val', k, H), 'auc'):.2f}" for H in (1, 24, 168)]
            wk = pr(n, "1mo val", k, 168)
            cells.append(f"{mean(wk, 'flag_out'):.2f} / {mean(wk, 'flag_out_pred'):.2f}")
        lines.append(f"{LAB5[k]} & " + " & ".join(cells) + r" \\")
    w("ieee_decision", "\n".join(lines))

    lines = []
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID")):
        for st, stl in (("1mo val", "val.\\ year"), ("1mo win", "window")):
            first = True
            for k in ("FREE", "MC", "ENS", "LA-1", "LA-2"):
                c0 = f"\\multirow{{5}}{{*}}{{\\makecell[l]{{{lab}\\\\one month: {stl}}}}}" if first else ""; first = False
                auc = [f"{mean(pr(n, st, k, H), 'auc'):.2f}" for _, H in SC]
                ba = [f"{mean(pr(n, st, k, H), 'ba_rule'):.2f}" for _, H in SC]
                wk = pr(n, st, k, 168)
                lines.append(f"{c0} & {LAB5[k]} & " + " & ".join(auc + ba) +
                             f" & {mean(wk, 'flag_out'):.2f} & {mean(wk, 'flag_out_pred'):.2f} & {mean(wk, 'kappa_a'):.1f} \\\\")
            lines.append(r"\midrule")
    w("decision_full", "\n".join(lines[:-1]))

    lines = []
    for n, lab in (("gef", "GEFCom2014"), ("covid", "post-COVID")):
        for st, stl in (("full", "full year"), ("1mo val", "one month, val.\\ year stop"), ("1mo win", "one month, window stop")):
            S = lambda k: [r for r in E14[n]["states"] if r["state"] == st and r["method"] == k]
            sh = S("LA-2")
            ka = [f"{np.mean([r['kappa'] * r['a'] for r in S(k)]):.1f}" for k in ("MC", "ENS", "LA-1", "LA-2")]
            lines.append(f"{lab} & {stl} & {100 * mean(sh, 'base'):.0f} & {mean(sh, 'auc_hour'):.2f} & {mean(sh, 'acf1'):.2f} & "
                         f"{mean(sh, 'heff_day'):.1f} & {mean(sh, 'heff_week'):.1f} & " + " & ".join(ka) + r" \\")
    w("decision_states", "\n".join(lines))
