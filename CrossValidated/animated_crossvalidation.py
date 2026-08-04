
# Animated version


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid


rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 16
rcParams['xtick.labelsize'] = 14
rcParams['ytick.labelsize'] = 14
rcParams['legend.fontsize'] = 10
rcParams['axes.linewidth'] = 1.0
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'
rcParams['xtick.major.width'] = 0.8
rcParams['ytick.major.width'] = 0.8
rcParams['xtick.major.size'] = 4
rcParams['ytick.major.size'] = 4


# Parameters

R_0 = 2.0
R_T = 2.5
MU  = 0.5
T   = 1.0

cases = [
    {"XT": 0.25, "color": "#2ca02c", "label": r"$X_T=0.25$",
     "dir": "PMP_on_PDE_XT0p25_RT2p5_gamma0p1/rundir"},
    {"XT": 0.5,  "color": "#1f77b4", "label": r"$X_T=0.5$",
     "dir": "PMP_on_PDE_XT0p5_RT2p5_gamma0p1/rundir"},
    {"XT": 0.75,  "color": "#d62728", "label": r"$X_T=0.75$",
     "dir": "PMP_on_PDE_XT0p75_RT2p5_gamma0p1/rundir"},
]

# Number of animation frames spanning t = 0 -> T
N_FRAMES = 90
FRAME_INTERVAL_MS = 60


# Load PDE data

def load_dat(path, cols=(0, 1)):
    try:
        data = np.loadtxt(path, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return tuple(data[:, c] for c in cols)
    except:
        return None


for case in cases:
    d = case["dir"]

    ctrl = load_dat(d + "/controls.dat", (0, 1, 2))
    case["t_ctrl"], case["G0"], case["DG"] = ctrl if ctrl else (np.array([]),) * 3

    xdata = load_dat(d + "/X.dat")
    case["t_X"], case["X"] = xdata if xdata else (np.array([]), np.array([]))

    rdata = load_dat(d + "/R_support.dat")
    case["t_R"], case["R"] = rdata if rdata else (np.array([]), np.array([]))

    cdata = load_dat(d + "/cost.dat")
    W = cdata[1][-1] if cdata else 0.0

    if len(case["t_X"]) and len(case["t_R"]):
        X_final = case["X"][-1]
        R_final = case["R"][-1]
        XT_val = case["XT"]
        T_cost = (1e3 / XT_val**2 * (X_final - XT_val)**2
                  + 1e3 / R_T**2 * (R_final - R_T)**2)
    else:
        T_cost = 0.0

    case["W_final"] = W
    case["T_final"] = T_cost

    # Time at which this case's own curves finish — the bar for this case
    # grows to full height in sync with its own curves, not the global clock.
    t_ends = [arr[-1] for arr in (case["t_ctrl"], case["t_X"], case["t_R"]) if len(arr)]
    case["t_end"] = max(t_ends) if t_ends else T

bar_labels = [r'$\mathbf{X_T=' + f'{c["XT"]:g}' + r'}$' for c in cases]
colors_bar = [c["color"] for c in cases]



fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))


def add_panel_heading(ax, label, title,
                       label_x=-0.12, label_y=1.05,
                       title_x=0.5,   title_y=1.05):
    ax.text(label_x, label_y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='left', va='bottom')
    ax.text(title_x, title_y, title, transform=ax.transAxes,
            fontsize=20, fontweight='normal', ha='center', va='bottom')


#  Panel A: Controls 
axA = axes[0]
linesA = []  # (line_artist, t_array, y_array)
for case in cases:
    XT, c = case["XT"], case["color"]
    l_G0, = axA.plot([], [], '--', color=c, lw=2.0, label=r'$G_0(t)_{' + f'{XT:g}' + r'}$')
    l_DG, = axA.plot([], [], '-',  color=c, lw=2.0, label=r'$\Delta G(t)_{' + f'{XT:g}' + r'}$')
    linesA.append((l_G0, case["t_ctrl"], case["G0"]))
    linesA.append((l_DG, case["t_ctrl"], case["DG"]))

axA.set_xlabel(r'$t$')
axA.set_ylabel(r'$G_0(t),\; \Delta G(t)$')
add_panel_heading(axA, 'A', 'Controls')
axA.set_xlim(0, 1)

all_ctrl_vals = np.concatenate([c["G0"] for c in cases if len(c["G0"])] +
                                [c["DG"] for c in cases if len(c["DG"])])
pad = 0.08 * (all_ctrl_vals.max() - all_ctrl_vals.min())
axA.set_ylim(all_ctrl_vals.min() - pad, all_ctrl_vals.max() + pad)
axA.legend(loc='best', ncol=1, framealpha=0.9, fontsize=10,
           handlelength=3.5, handletextpad=0.6, columnspacing=0.8)
axA.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.3)


#  Panel B: State variables 
axB = axes[1]
linesB = []
x_handles, r_handles = [], []
for case in cases:
    XT, c = case["XT"], case["color"]
    l_X, = axB.plot([], [], '-',  color=c, lw=2.0, label=r'$X(t)_{' + f'{XT:g}' + r'}$')
    l_R, = axB.plot([], [], '--', color=c, lw=2.0, label=r'$R(t)_{' + f'{XT:g}' + r'}$')
    linesB.append((l_X, case["t_X"], case["X"]))
    linesB.append((l_R, case["t_R"], case["R"]))
    x_handles.append(l_X)
    r_handles.append(l_R)
    axB.axhline(XT, color=c, lw=0.6, ls=':', alpha=0.4)

axB.axhline(R_T, color='gray', lw=0.8, ls=':', alpha=0.5)
axB.text(0.02, R_T + 0.05, r'$R_T$', fontsize=9, color='gray')

axB.set_xlabel(r'$t$')
axB.set_ylabel(r'$X(t),\; R(t)$')
add_panel_heading(axB, 'B', 'State variables')
axB.set_xlim(0, 1)

all_state_vals = np.concatenate([c["X"] for c in cases if len(c["X"])] +
                                 [c["R"] for c in cases if len(c["R"])] + [np.array([R_T])])
axB.set_ylim(0, all_state_vals.max() * 1.08)
axB.legend(handles=x_handles + r_handles, loc='center left', ncol=2, framealpha=0.9,
           fontsize=10, handlelength=3.0, handletextpad=0.5, columnspacing=0.8)


#  Panel C: Cost bar chart (grows bottom -> top) ----
axC = axes[2]
x_pos = np.arange(len(cases))
bar_width = 0.55

total_vals = [c["W_final"] + c["T_final"] for c in cases]
Y_FLOOR = min(total_vals) * 1e-2  # small positive floor for log scale

bars_W = axC.bar(x_pos, [Y_FLOOR] * len(cases), bar_width, color=colors_bar,
                  alpha=0.7, edgecolor='black', lw=0.8, label=r'$\mathcal{W}$ (dissipation)')
bars_T = axC.bar(x_pos, [Y_FLOOR] * len(cases), bar_width, bottom=[Y_FLOOR] * len(cases),
                  color=colors_bar, alpha=0.35, edgecolor='black', lw=0.8,
                  label=r'$\mathcal{T}$ (terminal)')
cost_texts = [axC.text(i, Y_FLOOR, '', ha='center', va='bottom', fontsize=11,
                        fontweight='bold', color=colors_bar[i]) for i in range(len(cases))]

axC.set_yscale('log')
axC.set_xticks(x_pos)
axC.set_xticklabels(bar_labels, fontsize=10, fontweight='bold')
axC.set_ylabel(r'Cost $\mathcal{J}$')
add_panel_heading(axC, 'C', 'Total cost')
axC.set_ylim(Y_FLOOR, max(total_vals) * 3)


for ax in axes:
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')

plt.tight_layout(w_pad=2.5)

t_frames = np.linspace(0.0, T, N_FRAMES + 1)


def update(frame_idx):
    t_now = t_frames[frame_idx]

    # Panels A & B: reveal each line up to the current time
    for line, t_arr, y_arr in linesA + linesB:
        if len(t_arr) == 0:
            continue
        mask = t_arr <= t_now
        line.set_data(t_arr[mask], y_arr[mask])

    # Panel C: each bar grows to full height exactly when that case's own
    # curves finish in panels A/B (so faster-finishing cases finish their bar sooner).
    for i, case in enumerate(cases):
        frac = min(t_now / case["t_end"], 1.0)
        w = max(case["W_final"] * frac, Y_FLOOR)
        t_cost = max(case["T_final"] * frac, 0.0)
        bars_W[i].set_height(w)
        bars_T[i].set_y(w)
        bars_T[i].set_height(max(t_cost, Y_FLOOR * 1e-3))
        total_now = case["W_final"] * frac + case["T_final"] * frac
        cost_texts[i].set_position((i, w + t_cost))
        cost_texts[i].set_text(f'{total_now:.1e}')

    artists = [l for l, _, _ in linesA + linesB] + list(bars_W) + list(bars_T) + cost_texts
    return artists


anim = animation.FuncAnimation(fig, update, frames=len(t_frames),
                                interval=FRAME_INTERVAL_MS, blit=False)

anim.save("animated_cross_validation.gif", writer='pillow', dpi=150)

print("Saved animated_cross_validation.gif")
