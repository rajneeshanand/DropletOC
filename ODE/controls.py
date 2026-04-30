import numpy as np
import matplotlib
matplotlib.use("QtAgg")   # stable
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import cumulative_trapezoid


# --------------------------------------------------
# Parameters
# --------------------------------------------------
X_T = 0.5
R_0 = 2.0
R_T = 2.5
T   = 1.0
MU  = 0.5

# --------------------------------------------------
# Define functions
# --------------------------------------------------
def F_s(s):
    val = (33*s**2)/(250*np.sinh(s)**2)
    val *= ((R_T/R_0)*(2*np.cosh(s) - R_T/R_0) - 1)
    return np.sqrt(np.maximum(val, 0.0))

target = X_T / R_0

def root_fun(s):
    return F_s(s) - target

# --------------------------------------------------
# Root finding (scan initial guesses)
# --------------------------------------------------
found = False
guesses = np.linspace(1, 10, 2000)

for g in guesses:
    try:
        sol = root_scalar(root_fun, x0=g, method='newton')
        if sol.converged and sol.root > 0 and np.isreal(sol.root):
            s = sol.root
            found = True
            break
    except Exception:
        pass

if not found:
    raise RuntimeError("Could not find any positive root")

print(f"solved s = {s:g}")

# --------------------------------------------------
# Spatial discretization
# --------------------------------------------------
Nx = 1200
x = np.linspace(0, 1, Nx)

rho_x = (R_T*np.sinh(s*x) + R_0*np.sinh(s*(1 - x))) / np.sinh(s)

# --------------------------------------------------
# Time mapping
# --------------------------------------------------
I = np.trapezoid(rho_x**2, x)
cumulative_integral = cumulative_trapezoid(rho_x**2, x, initial=0)

tau_T = T / (MU * I)
t_x = T * cumulative_integral / I

# --------------------------------------------------
# Derivatives
# --------------------------------------------------
drho_dx = np.gradient(rho_x, x)
drho_dtau = drho_dx / tau_T

# --------------------------------------------------
# Controls
# --------------------------------------------------
P_tau = (14/5) * drho_dtau
p_0 = X_T * 700 / (33 * tau_T)

G0_tau = -(5/6) * P_tau * (rho_x**2)
delta_G_tau = (11/30) * p_0 * (rho_x**2)

# --------------------------------------------------
# Plot
# --------------------------------------------------
plt.figure(figsize=(9, 6))
ax = plt.gca()

plt.plot(t_x/T, delta_G_tau, 'r-', linewidth=5, label=r'$\Delta G(t)$')
plt.plot(t_x/T, G0_tau, '#00ff00', linewidth=5, label=r'$G_0(t)$')

plt.xlim(0, 1)

plt.xlabel(r'$t$', fontsize=24, fontweight='bold', fontfamily='cambria')

ax.annotate(
    '',                 # no text, just arrow
    xy=(0.68, -0.1),         # arrow head (right side)
    xytext=(0.56, -0.1),     # arrow tail
    xycoords='axes fraction',
    arrowprops=dict(
        arrowstyle='->',
        linewidth=3,
        color='black'
    )
)

#plt.ylabel(r'$G_0(t), \Delta G(t)$', fontsize=16, fontweight='bold')
plt.title('Controls', fontsize=24, fontweight='bold', color='black', pad = 16, fontfamily='cambria')

plt.legend(
    #loc='upper left',
    loc = 'best',
    fontsize=24,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    edgecolor= "white",
)
for spine in ax.spines.values():
    spine.set_linewidth(2.5)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)


for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontweight('bold')
    label.set_fontfamily('cambria')
    label.set_color('black')

ax.tick_params(
    axis='both',
    which='major',
    length=6,   # tick length (points)
    width=2    # tick thickness
)

ax.tick_params(direction='in', top=False, right=False)

plt.tight_layout()
plt.show()

