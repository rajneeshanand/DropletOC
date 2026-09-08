
# Run from the directory containing the PMP_on_PDE_* output folders: python plot_crossvalidation_figure.py

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
     "dir": "PMP_on_PDE_XT0p5_RT2p5_gamma0p1_g01_c200/rundir"},
    {"XT": 0.75,  "color": "#d62728", "label": r"$X_T=0.75$",
     "dir": "PMP_on_PDE_XT0p75_RT2p5_gamma0p1/rundir"},
]

# ODE solutions

def ode_solution(X_T):
    sigma = R_T / R_0
    target = X_T / R_0

    def F_s(s):
        val = (33.0 * s**2) / (250.0 * np.sinh(s)**2)
        val *= (sigma * (2.0 * np.cosh(s) - sigma) - 1.0)
        return np.sqrt(np.maximum(val, 0.0))

    # Find root
    guesses = np.linspace(0.01, 15.0, 5000)
    s_val = None
    for g in guesses:
        try:
            sol = root_scalar(lambda s: F_s(s) - target, x0=g, method='newton')
            if sol.converged and sol.root > 0:
                s_val = sol.root
                break
        except:
            pass
    if s_val is None:
        return None

    Nx = 2000
    x = np.linspace(0, 1, Nx)
    rho = (R_T * np.sinh(s_val * x) + R_0 * np.sinh(s_val * (1 - x))) / np.sinh(s_val)
    I_tot = np.trapz(rho**2, x)
    cum = cumulative_trapezoid(rho**2, x, initial=0)
    tau_T = T / (MU * I_tot)
    t_arr = T * cum / I_tot

    drho = np.gradient(rho, x) / tau_T
    P = (14.0 / 5.0) * drho
    p0 = X_T * 700.0 / (33.0 * tau_T)

    G0 = -(5.0 / 6.0) * P * rho**2
    DG = (11.0 / 30.0) * p0 * rho**2
    tau = x * tau_T
    X_ode = (33.0 / 700.0) * p0 * tau

    return {"t": t_arr, "G0": G0, "DG": DG, "R": rho, "X": X_ode}


# Load PDE data

def load_dat(path, cols=(0, 1)):
    try:
        data = np.loadtxt(path, comments='#')
        if data.ndim == 1:
            data = data.reshape(1, -1)
        return tuple(data[:, c] for c in cols)
    except:
        return None


fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))


def add_panel_heading(ax, label, title,
                       label_x=-0.06, label_y=1.05,
                       title_x=0.5,   title_y=1.05,
                       fontsize=12):
    ax.text(label_x, label_y, label, transform=ax.transAxes,
            fontsize=16, fontweight='bold', ha='left', va='bottom')
    ax.text(title_x, title_y, title, transform=ax.transAxes,
            fontsize=20, fontweight='normal', ha='center', va='bottom')


# Panel A: Controls

ax = axes[0]

for case in cases:
    XT = case["XT"]
    c  = case["color"]
    lab = case["label"]
    d  = case["dir"]

    # PDE data (ODE controls applied to PDE)
    ctrl = load_dat(d + "/controls.dat", (0, 1, 2))
    if ctrl:
        t_pde, G0_pde, DG_pde = ctrl
        ax.plot(t_pde, G0_pde, '--', color=c, lw=2.0, label=r'$G_0(t)_{' + f'{XT:g}' + r'}$')
        ax.plot(t_pde, DG_pde, '-',  color=c, lw=2.0, label=r'$\Delta G(t)_{' + f'{XT:g}' + r'}$')

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$G_0(t),\; \Delta G(t)$')
add_panel_heading(ax, 'A', 'Controls',
                   label_x=-0.12, label_y=1.05, title_x=0.5, title_y=1.05)
ax.set_xlim(0, 1)
ax.legend(loc='best', ncol=1, frameon=False, fontsize=10,
          handlelength=3.5, handletextpad=0.6, columnspacing=0.8)
ax.axhline(0, color='gray', lw=0.5, ls='-', alpha=0.3)


# Panel B: State variables

ax = axes[1]

x_handles, r_handles = [], []

for case in cases:
    XT = case["XT"]
    c  = case["color"]
    d  = case["dir"]

    # PDE X(t)
    xdata = load_dat(d + "/X.dat")
    if xdata:
        h, = ax.plot(xdata[0], xdata[1], '-', color=c, lw=2.0, label=r'$X(t)_{' + f'{XT:g}' + r'}$')
        x_handles.append(h)

    # PDE R_moment(t):  matches ODE definition, shows CTR
    rdata = load_dat(d + "/R_support.dat")
    if rdata:
        h, = ax.plot(rdata[0], rdata[1], '--', color=c, lw=2.0, label=r'$R(t)_{' + f'{XT:g}' + r'}$')
        r_handles.append(h)

    # Target markers
    ax.axhline(XT, color=c, lw=0.6, ls=':', alpha=0.4)

# R_T target
ax.axhline(R_T, color='gray', lw=0.8, ls=':', alpha=0.5)
ax.text(0.02, R_T + 0.05, r'$R_T$', fontsize=9, color='gray')

ax.set_xlabel(r'$t$')
ax.set_ylabel(r'$X(t),\; R(t)$')
add_panel_heading(ax, 'B', 'State variables',
                   label_x=-0.12, label_y=1.05, title_x=0.5, title_y=1.05)
ax.set_xlim(0, 1)
ax.legend(handles=x_handles + r_handles, loc='center left', ncol=2, frameon=False, fontsize=10,
          handlelength=3.0, handletextpad=0.5, columnspacing=0.8)


# Panel C: Cost bar chart

ax = axes[2]

# Collect costs from data files
bar_labels = []
dissipation_vals = []
terminal_vals = []
colors_bar = []

for case in cases:
    d = case["dir"]
    lab = case["label"]

    # Read final cost from cost.dat (last entry = total dissipation)
    cdata = load_dat(d + "/cost.dat")
    if cdata:
        W = cdata[1][-1]  # last cumulative dissipation
    else:
        W = 0.0

    # Read X(T) and R_support(T) for terminal cost
    xdata = load_dat(d + "/X.dat")
    rdata = load_dat(d + "/R_support.dat")
    if xdata and rdata:
        X_final = xdata[1][-1]
        R_final = rdata[1][-1]
        XT_val = case["XT"]
        T_cost = 1e3 / XT_val**2 * (X_final - XT_val)**2 + 1e3 / R_T**2 * (R_final - R_T)**2
    else:
        T_cost = 0.0

    bar_labels.append(r'$\mathbf{X_T=' + f'{case["XT"]:g}' + r'}$')
    dissipation_vals.append(W)
    terminal_vals.append(T_cost)
    colors_bar.append(case["color"])

x_pos = np.arange(len(cases))
bar_width = 0.55

# Use log scale since costs span orders of magnitude
total_vals = [w + t for w, t in zip(dissipation_vals, terminal_vals)]

bars_W = ax.bar(x_pos, dissipation_vals, bar_width, color=colors_bar,
                alpha=0.7, edgecolor='black', lw=0.8, label=r'$\mathcal{W}$ (dissipation)')
bars_T = ax.bar(x_pos, terminal_vals, bar_width, bottom=dissipation_vals,
                color=colors_bar, alpha=0.35, edgecolor='black', lw=0.8,
                label=r'$\mathcal{T}$ (terminal)')

ax.set_yscale('log')
ax.set_xticks(x_pos)
ax.set_xticklabels(bar_labels, fontsize=10, fontweight='bold')
ax.set_ylabel(r'Cost $\mathcal{J}$')
add_panel_heading(ax, 'C', 'Total cost',
                   label_x=-0.12, label_y=1.05, title_x=0.5, title_y=1.05)

# Add value labels on top of bars
for i, (w, t, total) in enumerate(zip(dissipation_vals, terminal_vals, total_vals)):
    ax.text(i, total * 1.5, f'{total:.1e}', ha='center', va='bottom', fontsize=11,
            fontweight='bold', color=colors_bar[i])


for ax in axes:
    ax.xaxis.label.set_fontweight('bold')
    ax.yaxis.label.set_fontweight('bold')
    for tick_label in ax.get_xticklabels() + ax.get_yticklabels():
        tick_label.set_fontweight('bold')


plt.tight_layout(w_pad=2.5)

plt.savefig("cross_validation_figure.png", dpi=300, bbox_inches='tight')
plt.savefig("fig_bar_summary.pdf", bbox_inches="tight")

