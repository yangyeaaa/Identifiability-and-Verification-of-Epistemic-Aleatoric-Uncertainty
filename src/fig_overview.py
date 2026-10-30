"""Schematic overview of the study (Fig. 1)."""
import sys, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

out = os.path.join(sys.argv[1], "figures", "overview.png")
fig, ax = plt.subplots(figsize=(12, 5.2)); ax.set_xlim(0, 12); ax.set_ylim(0, 5.2); ax.axis("off")
C = dict(net="#e8f0fa", post="#eef7ee", free="#f7ecea", cal="#fdf4e3", ver="#f1edf7", edge="#4d4d4d")


def box(x, y, w, h, text, fc, fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=C["edge"], lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def arrow(x0, y0, x1, y1):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=11, lw=1, color=C["edge"]))


box(0.1, 2.0, 1.9, 1.2, "Inputs: same hour on\n7 past days + target-day\ncalendar & temperature\n(24 h ahead)", C["net"], 8.5)
box(2.4, 2.0, 2.0, 1.2, "iTransformer encoder\n+ heteroscedastic head\n$(\\mu_\\theta, \\log\\sigma^2_\\theta)$,\ntrained by Gaussian NLL", C["net"], 8.5)
arrow(2.0, 2.6, 2.4, 2.6)
ax.text(5.9, 4.85, "Decompositions of the predictive variance", ha="center", fontsize=9, fontweight="bold")
ys = [4.0, 3.3, 2.6, 1.9]
for y, t in zip(ys, ["MC dropout (M = 40 masks)", "Deep ensemble (K = 5)", "Laplace, last layer (evidence)", "Laplace, two layers (evidence)"]):
    box(4.75, y, 2.3, 0.55, t, C["post"], 8.0); arrow(4.4, 2.6, 4.75, y + 0.27)
box(4.75, 1.05, 2.3, 0.6, "Learning curves (cross-fitted)\naleatoric = asymptote", C["cal"], 8.0)
arrow(4.4, 2.45, 4.75, 1.35)
box(4.75, 0.2, 2.3, 0.6, "Free outputs $(\\hat\\sigma_{epi},\\hat\\sigma_{ale})$\n(negative control)", C["free"], 8.0)
arrow(4.4, 2.3, 4.75, 0.5)
box(7.35, 2.0, 1.6, 1.2, "$\\sigma^2_{tot}=\\sigma^2_{ale}+\\sigma^2_{epi}$\n$R_{epi}=\\sigma^2_{epi}/\\sigma^2_{tot}$", C["post"], 9)
for y in ys + [1.05, 0.2]:
    arrow(7.05, y + 0.28, 7.35, 2.6)
box(9.25, 3.15, 2.65, 0.85, "Calibration of $\\sigma_{tot}$\nsplit / rolling conformal\n(split preserved, Remark 1)", C["cal"], 8.2)
arrow(8.95, 2.9, 9.25, 3.55)
box(9.25, 0.25, 2.65, 2.85, "", C["ver"])
ax.text(10.52, 2.85, "Verification protocol", ha="center", fontsize=9, fontweight="bold")
for i, t in enumerate(["T1  stability across seeds", "T2  reduction with more data", "T3  distance from training support\n      (incl. out-of-support inputs)",
                       "T4  reference with known truth\n      ($f^\\star$, $\\sigma^\\star$); decision test\n      over windows (Prop. 6)"]):
    ax.text(9.38, 2.45 - i * 0.58, t, ha="left", va="center", fontsize=8.3)
arrow(8.95, 2.3, 9.25, 1.8)
ax.text(10.52, 4.95, "Prop. 1: free outputs not identifiable\nProp. 2: a fitted forecast cannot fix the scale\nProps. 3-4: retraining fixes it, but only in support\nProp. 5: the variance head absorbs the mean error\nProp. 6: retraining decision identifiable only over windows", ha="center", va="top", fontsize=7.8, style="italic")
plt.savefig(out, dpi=220, bbox_inches="tight"); print("saved", out)
