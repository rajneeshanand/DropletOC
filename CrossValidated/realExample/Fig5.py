#!/usr/bin/env python

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'

# ------------------------------------------------------------------
# toggle: outline in red the bars whose lubricating film ruptured
SHOW_RUPTURE = False
# legend size knobs -- edit these to resize the panel A legend
LEGEND_FONTSIZE = 14
LEGEND_HANDLELENGTH = 1.8
LEGEND_HANDLETEXTPAD = 0.6
LEGEND_COLUMNSPACING = 1.10
# ------------------------------------------------------------------

GAMMA_LABELS = [r"$\gamma = 0.1$", r"$\gamma = 0.5$", r"$\gamma = 1.0$"]

# rows = gamma (0.1, 0.5, 1.0); cols = X_T/R_T (0.10, 0.20, 0.30)
X_FINAL = np.array([[0.232985, 0.433680, 0.424906],
                    [0.229917, 0.474616, 0.670386],
                    [0.227514, 0.448567, 0.699444]])

X_TARGET = np.array([0.25, 0.50, 0.75])

DISSIPATION = np.array([[6.288448e+01, 8.764512e+04, 5.415689e+07],
                        [5.243524e+01, 2.909294e+03, 1.775820e+07],
                        [1.887379e+02, 5.874819e+02, 1.821774e+06]])

# True where the film ruptured before the horizon completed
RUPTURED = np.array([[False, True,  True ],
                     [False, False, True ],
                     [False, False, False]])

ATTAIN = X_FINAL / X_TARGET[None, :]

# ratio label, colour, hatch
TASKS = [(0.10, "#9B3FA0", ""),
         (0.20, "#1B6E2E", "///"),
         (0.30, "#E8802E", "xxx")]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
width = 0.26
xpos = np.arange(3)

for ax, data, logscale, ylab, title, panel_label in (
        (axes[0], ATTAIN,      False, r"$X(T)\,/\,X_T$", "Displacement attained", "A"),
        (axes[1], DISSIPATION, True,  r"$\mathcal{W}$",  "Transport dissipation", "B")):

    for ti, (ratio, colour, hatch) in enumerate(TASKS):
        off = (ti - 1) * width
        for gi in range(3):
            rup = SHOW_RUPTURE and RUPTURED[gi, ti]
            ax.bar(xpos[gi] + off, data[gi, ti], width,
                   color=colour, hatch=hatch,
                   edgecolor="crimson" if rup else "black",
                   linewidth=2.6 if rup else 1.1, zorder=3)

    if logscale:
        ax.set_yscale("log")
        ax.set_ylim(1e1, 1e9)
    else:
        ax.axhline(1.0, color="k", ls="-.", lw=1.3, alpha=0.7, zorder=2)
        ax.text(-0.46, 1.02, "target", fontsize=14, alpha=0.75)
        ax.set_ylim(0, 1.4)

    ax.set_xticks(xpos)
    ax.set_xticklabels(GAMMA_LABELS, fontsize=18)
    ax.set_ylabel(ylab, fontsize=19)
    ax.set_title(title, fontsize=19, pad=12)
    ax.text(-0.06, 1.05, panel_label, transform=ax.transAxes,
            fontsize=20, fontweight="bold", ha="left", va="bottom")
    ax.tick_params(axis="y", which="both", labelsize=16, direction="in", right=True, width=1.4)
    ax.tick_params(axis="x", length=0)
    for sp in ax.spines.values():
        sp.set_linewidth(1.6)
    #ax.grid(axis="y", alpha=0.22, zorder=0)

handles = [Patch(facecolor=c, hatch=h, edgecolor="black",
                 label=rf"$X_T/R_T = {r:.2f}$")
           for r, c, h in TASKS]
if SHOW_RUPTURE:
    handles.append(Patch(facecolor="white", edgecolor="crimson", linewidth=2.6,
                         label="film rupture"))

for ax in axes:
    ax.legend(handles=handles, fontsize=LEGEND_FONTSIZE, loc="upper right",
              bbox_to_anchor=(0.99, 1.0), ncol=3, handlelength=LEGEND_HANDLELENGTH,
              handletextpad=LEGEND_HANDLETEXTPAD, columnspacing=LEGEND_COLUMNSPACING,
              borderaxespad=0.5, frameon=False)

plt.tight_layout(w_pad=4.0)
plt.savefig("fig_bar_summary.pdf", bbox_inches="tight")
plt.savefig("fig_bar_summary.png", dpi=300, bbox_inches="tight")
print("wrote fig_bar_summary.pdf / .png\n")

# ---- table for the manuscript ----
print(f"{'gamma':>8} {'XT/RT':>7} {'X(T)':>9} {'X(T)/X_T':>10} {'W':>12}  status")
for gi, glbl in enumerate(["0.1", "0.5", "1.0"]):
    for ti, (ratio, _, _) in enumerate(TASKS):
        print(f"{glbl:>8} {ratio:7.2f} {X_FINAL[gi,ti]:9.4f} "
              f"{ATTAIN[gi,ti]:10.3f} {DISSIPATION[gi,ti]:12.3e}"
              f"  {'RUPTURED' if RUPTURED[gi,ti] else 'completed'}")