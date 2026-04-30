import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from scipy.integrate import solve_ivp

# ============================================================
# Format
# ============================================================
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif']
rcParams['mathtext.fontset'] = 'stix'
rcParams['font.size'] = 10
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 8
rcParams['lines.linewidth'] = 1.4
rcParams['axes.linewidth'] = 0.8
rcParams['xtick.direction'] = 'in'
rcParams['ytick.direction'] = 'in'

# ============================================================
# Parameters
# ============================================================
mu = 0.1; X_T = 0.5; R_T = 2.5; A_w = 1e3; B_w = 1e3
p1 = A_w / X_T**2; p2 = B_w / R_T**2

def clf_ctrl(X, R):
    al = 9*A_w*(X - X_T)/(35*mu*X_T**2*R**4)
    be = 6*B_w*(R - R_T)/(7*mu*R_T**2*R**4)
    G0 = (35*mu*R**6/18)*be
    DG = -(77*mu*R**6/27)*al
    return G0, DG

def closed_loop_vec(X, R):
    G0, DG = clf_ctrl(X, R)
    Xdot = 9*DG/(70*mu*R**4)
    Rdot = -3*G0/(7*mu*R**4)
    return Xdot, Rdot

def T_cost(X, R):
    return A_w*((X - X_T)/X_T)**2 + B_w*((R - R_T)/R_T)**2

def dynamics(t, s):
    X, R = s
    R = max(R, 0.15)
    Xd, Rd = closed_loop_vec(X, R)
    return [Xd, Rd]

# ============================================================
# Vector field grid
# ============================================================
nx, ny = 24, 24
X_grid = np.linspace(-0.35, 1.35, nx)
R_grid = np.linspace(1.0, 4.0, ny)
Xg, Rg = np.meshgrid(X_grid, R_grid)

U = np.zeros_like(Xg)
V = np.zeros_like(Rg)

for i in range(ny):
    for j in range(nx):
        xd, rd = closed_loop_vec(Xg[i,j], Rg[i,j])
        U[i,j] = xd
        V[i,j] = rd

speed = np.sqrt(U**2 + V**2)
speed_safe = np.where(speed > 0, speed, 1.0)
Un = U / speed_safe
Vn = V / speed_safe

# ============================================================
# Trajectories
# ============================================================
ics = [
    (0.0, 2.0),
    (-0.2, 1.5),
    (1.2, 3.5),
    (0.9, 1.2),
    (-0.3, 3.5),
]
ic_labels = [
    r'$(0.0,\; 2.0)$',
    r'$(-0.2,\; 1.5)$',
    r'$(1.2,\; 3.5)$',
    r'$(0.9,\; 1.2)$',
    r'$(-0.3,\; 3.5)$',
]
traj_colors = ['#2166ac', '#d62728', '#1b7837', '#ff7f0e', '#762a83']

t_eval = np.linspace(0, 1.5, 10000)
trajs = []
for ic in ics:
    sol = solve_ivp(dynamics, (0, 1.5), list(ic), t_eval=t_eval,
                    method='RK45', max_step=5e-5, rtol=1e-12, atol=1e-14)
    trajs.append(sol)

# ============================================================
# FIGURE: Single panel
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(4.5, 3.8))

# Background shading: log(T) contourf
nx2, ny2 = 200, 200
X_fine = np.linspace(-0.35, 1.35, nx2)
R_fine = np.linspace(1.0, 4.0, ny2)
Xf, Rf = np.meshgrid(X_fine, R_fine)
T_grid = np.zeros_like(Xf)
for i in range(ny2):
    for j in range(nx2):
        T_grid[i,j] = T_cost(Xf[i,j], Rf[i,j])

ax.contourf(Xf, Rf, np.log10(T_grid + 1e-6), levels=30, cmap='Blues', alpha=0.25)

# CLF level sets
theta = np.linspace(0, 2*np.pi, 400)
level_vals = [20, 80, 250, 700, 1800, 4000]
for lev in level_vals:
    a_ell = np.sqrt(lev/p1)
    b_ell = np.sqrt(lev/p2)
    ax.plot(X_T + a_ell*np.cos(theta), R_T + b_ell*np.sin(theta),
            'k-', lw=0.6, alpha=0.4)

# Vector field (quiver) - muted color
ax.quiver(Xg, Rg, Un, Vn, speed, cmap='Reds', alpha=0.55,
          scale=32, width=0.0035, headwidth=4, headlength=4,
          headaxislength=3.5, zorder=2)

# Trajectories - bold and distinct
for i, sol in enumerate(trajs):
    ax.plot(sol.y[0], sol.y[1], '-', color=traj_colors[i], lw=2.0,
            zorder=4, label=ic_labels[i])
    # Start marker
    ax.plot(sol.y[0,0], sol.y[1,0], 'o', color=traj_colors[i], ms=6,
            mec='k', mew=0.6, zorder=5)
    # Arrow at ~12% of trajectory
    idx = len(sol.t) // 8
    stp = max(3, idx // 8)
    ax.annotate('', xy=(sol.y[0,idx+stp], sol.y[1,idx+stp]),
                xytext=(sol.y[0,idx], sol.y[1,idx]),
                arrowprops=dict(arrowstyle='->', color=traj_colors[i],
                               lw=2.0, mutation_scale=12),
                zorder=6)

# Target
ax.plot(X_T, R_T, 'k*', ms=14, mew=1.0, zorder=7)
ax.annotate(r'$\boldsymbol{x}_T$', xy=(X_T, R_T),
            xytext=(X_T + 0.07, R_T + 0.13), fontsize=11, fontweight='bold')

# Sublevel set annotations
ax.text(X_T + 0.01, R_T + 0.38, r'$\Omega_{r_1}$', fontsize=8.5,
        color='#555555', ha='center', style='italic')
ax.text(X_T + 0.01, R_T + 0.82, r'$\Omega_{r_2}$', fontsize=8.5,
        color='#555555', ha='center', style='italic')

# Key annotation: the CLF property
ax.annotate(r'$\dot{\mathcal{T}} < 0\;\; \forall\, \boldsymbol{x} \neq \boldsymbol{x}_T$',
            xy=(0.97, 1.23), fontsize=9.5, color='#333333',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='#888888', alpha=0.92))

# Legend
leg = ax.legend(loc='center right', framealpha=0.93, borderpad=0.4,
                handlelength=1.5, title=r'$(X_0,\; R_0)$', title_fontsize=8.5)
leg.get_frame().set_edgecolor('#aaaaaa')

ax.set_xlabel(r'Position $X(t)$')
ax.set_ylabel(r'Width $R(t)$')
ax.set_xlim([-0.35, 1.35])
ax.set_ylim([1.0, 4.0])

plt.tight_layout()
#plt.savefig('clf_single_panel.pdf', dpi=300, bbox_inches='tight')
plt.savefig('clf_single_panel.png', dpi=300, bbox_inches='tight')
print("Single panel figure saved!")
