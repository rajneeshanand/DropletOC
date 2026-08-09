import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
from scipy.integrate import trapezoid, cumulative_trapezoid

# Parameters

X_T = 0.5
R_0 = 2.0
R_T = 2.5
T   = 1.0
MU  = 0.5

# Define functions


def F_s(s):
    val = (33*s**2) / (250*np.sinh(s)**2)
    val *= ( (R_T/R_0)*(2*np.cosh(s) - R_T/R_0) - 1 )
    return np.sqrt(np.maximum(val, 0.0))

target = X_T / R_0
root_fun = lambda s: F_s(s) - target

guesses = np.linspace(1e-3, 5, 2000)
F_vals = F_s(guesses)

crossings = np.where(np.diff(np.sign(F_vals - target)))[0]
if len(crossings) == 0:
    raise RuntimeError("No root found")

s_guess = np.mean(guesses[crossings[0]:crossings[0]+2])

sol = root_scalar(root_fun, x0=s_guess, method="newton")
s = sol.root

print(f"Solved s = {s}")

# State profile construction


Nx = 1500
x = np.linspace(0, 1, Nx)

rho_x = (
    R_T*np.sinh(s*x) + R_0*np.sinh(s*(1 - x))
) / np.sinh(s)

I = trapezoid(rho_x**2, x)
tau_T = T / (MU * I)

cumulative_integral = cumulative_trapezoid(rho_x**2, x, initial=0)
t_x = T * cumulative_integral / I

R_t = rho_x

# Compute X(t)

p0 = X_T * 700 / (33 * tau_T)

tau_vals = x * tau_T
X_tau = (33/700) * p0 * tau_vals

X_t = np.interp(np.linspace(0, T, Nx), t_x, X_tau)

# Plot

plt.figure(figsize=(9, 6))
ax = plt.gca()

#  Curves (different colors than controls plot)
ax.plot(t_x/T, R_t, color='#d62728', linewidth=5, label=r'$R(t)$')   # red
ax.plot(np.linspace(0,1,Nx), X_t, color='#1f77b4', linewidth=5, label=r'$X(t)$')  # blue

ax.set_xlim(0, 1)

#  X label with arrow
ax.set_xlabel(r'$t$', fontsize=24, fontweight='bold', fontfamily='cambria')

ax.annotate(
    '',
    xy=(0.68, -0.1),
    xytext=(0.56, -0.1),
    xycoords='axes fraction',
    arrowprops=dict(
        arrowstyle='->',
        linewidth=3,
        color='black'
    )
)

#ax.set_ylabel(r'$R(t), X(t)$', fontsize=18, fontweight='bold')
ax.set_title('State variables', fontsize=24, fontweight='bold', pad=16, fontfamily='cambria')

ax.legend(
    loc='best',
    fontsize=24,
    frameon=True,
    fancybox=True,
    framealpha=0.9,
    edgecolor='white'
)

#  Spine thickness
for spine in ax.spines.values():
    spine.set_linewidth(2.5)

ax.spines['bottom'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)

#  Tick label styling
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(20)
    label.set_fontweight('bold')
    label.set_fontfamily('cambria')
    label.set_color('black')

#  Tick geometry
ax.tick_params(
    axis='both',
    which='major',
    length=6,
    width=2,
    direction='in',
    top=False,
    right=False
)

plt.tight_layout()
plt.show()
