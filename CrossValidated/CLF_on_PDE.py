
# Cross-validation: ODE-derived CLF controller applied to the full PDE system.

from __future__ import print_function
import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fenics import *
import ufl

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["std_out_all_processes"] = False
set_log_level(30)


cli = argparse.ArgumentParser(
    description="CLF feedback controller applied to the full thin-film PDE"
)
cli.add_argument("params", type=str, help="Path to parameter file (XT, RT, gamma)")
cli.add_argument("--mu", type=float, default=0.5, help="Viscosity (default 0.5)")
cli.add_argument("--A", type=float, default=1.0e3, help="Terminal cost weight A (position)")
cli.add_argument("--B", type=float, default=1.0e3, help="Terminal cost weight B (size)")
cli.add_argument("--T", type=float, default=1.0, help="Time horizon")
cli.add_argument("--dt", type=float, default=5e-3, help="Time step")
cli.add_argument("--control-clamp", type=float, default=20.0,
                 help="Symmetric saturation bound on CLF controls (default 20, matching CMA-ES)")
cli.add_argument("--mode", type=str, default="clf",
                 choices=["clf", "pmp"],
                 help="Controller mode: 'clf' (feedback) or 'pmp' (feedforward from ODE)")
args = cli.parse_args()


# Read targets from parameter file

with open(args.params, "r") as f:
    target_position = float(f.readline().strip())   # X_T
    target_size     = float(f.readline().strip())   # R_T
    surface_tension_val = float(f.readline().strip())   # gamma

print(f"Target: X_T={target_position}, R_T={target_size}, gamma={surface_tension_val}")
print(f"Viscosity mu={args.mu}, A={args.A}, B={args.B}")
print(f"Controller mode: {args.mode}")


# Physical and numerical parameters 

time_step       = args.dt
final_time      = args.T
output_dt       = 2.0 * time_step
theta_scheme_val = 0.5
output_stride   = int(output_dt / time_step)

viscosity_val       = args.mu
equilibrium_angle   = np.pi / 4.0
initial_radius      = 2.0 / np.tan(equilibrium_angle)   # R_0
initial_position    = 0.0                                # X_0

precursor_thickness_val = 1.0e-2
regularization_eps_val  = 1.0e-8

hamaker_like_val = (
    3.0 * surface_tension_val
    * (precursor_thickness_val * np.tan(equilibrium_angle)) ** 2
)

film_cutoff = 1.1 * precursor_thickness_val
control_clamp = args.control_clamp

# CLF weights
A_w = args.A
B_w = args.B
X_T = target_position
R_T = target_size


# Output directory

tag = f"CLF_on_PDE_XT{target_position}_RT{target_size}_gamma{surface_tension_val}".replace(".", "p")
if args.mode == "pmp":
    tag = tag.replace("CLF", "PMP")

os.makedirs(tag, exist_ok=True)
os.makedirs(tag + "/rundir", exist_ok=True)


# 1-D mesh and function spaces 

domain_length = 8
num_cells     = 800
mesh_1d = IntervalMesh(num_cells, -3.0, 5.0)
spatial_coord = SpatialCoordinate(mesh_1d)

height_element = FiniteElement("Lagrange", mesh_1d.ufl_cell(), degree=2)
global_control_element = FiniteElement("Real", mesh_1d.ufl_cell(), degree=0)

height_space  = FunctionSpace(mesh_1d, height_element)
state_space   = FunctionSpace(mesh_1d, MixedElement([height_element, height_element]))
control_space = FunctionSpace(mesh_1d, MixedElement([global_control_element, global_control_element]))


# FEniCS constants

theta_scheme        = Constant(theta_scheme_val)
facet_normal        = FacetNormal(mesh_1d)
time_step_const     = Constant(time_step)
viscosity           = Constant(viscosity_val)
regularization_eps  = Constant(regularization_eps_val)
precursor_thickness = Constant(precursor_thickness_val)
surface_tension     = Constant(surface_tension_val)
hamaker_like        = Constant(hamaker_like_val)


# Initial condition  (identical to PDE_solver.py)
class SymmetricDroplet(UserExpression):
    def __init__(self, center, radius, length, film, **kwargs):
        super().__init__(**kwargs)
        self.center = center
        self.radius = radius
        self.length_scale = length
        self.film = film

    def eval(self, values, pt):
        x = pt[0]
        left  = self.center - 0.5 * self.radius
        right = self.center + 0.5 * self.radius
        if x < left or x > right:
            values[0] = self.film
            values[1] = 0.0
        else:
            amp = 1.0 - self.length_scale * self.film
            shape = 6.0 * amp * (x - left) * (right - x) / (self.radius ** 3)
            values[0] = self.film + shape
            values[1] = 3.0 * amp / (self.radius ** 3)

    def value_shape(self):
        return (2,)

# Physics functions  (identical to PDE_solver.py)
def thin_film_mobility(h):
    h3 = h**3
    h4 = h * h3
    base = h3 / (12.0 * viscosity)
    return (h4 * base) / (h4 + regularization_eps * base)

def interfacial_stress(h, kappa, c_off, c_grad):
    return h * (c_off + spatial_coord[0] * c_grad) + surface_tension * kappa

def thin_film_flux(h_mob, h_str, kappa, c_off, c_grad):
    stress = interfacial_stress(h_str, kappa, c_off, c_grad)
    mob    = thin_film_mobility(h_mob)
    dpidh  = (hamaker_like * (4.0 * precursor_thickness - 3.0 * h_mob) / (h_mob**5))
    return mob * (stress.dx(0) - dpidh * h_str.dx(0))


# CLF controller  

def clf_controls(X_cur, R_cur):
    # prevent getting R too small
    R_cur = max(R_cur, 0.3)

    G0 = (5.0 * B_w / 3.0) * R_cur**2 * (R_cur - R_T) / R_T**2
    DG = -(11.0 * A_w / 15.0) * R_cur**2 * (X_cur - X_T) / X_T**2

    # Saturate
    G0 = np.clip(G0, -control_clamp, control_clamp)
    DG = np.clip(DG, -control_clamp, control_clamp)

    return float(G0), float(DG)

# PMP



def pmp_controls_from_state(X_cur, R_cur, t_remaining):
    from scipy.optimize import root_scalar

    R_0_local = R_cur
    T_local   = t_remaining

    if T_local < 1e-10 or R_0_local < 0.3:
        return 0.0, 0.0

    sigma = R_T / R_0_local
    X_remaining = X_T - X_cur

    if abs(X_remaining) < 1e-12:
        # Only need size correction
        # From Eq. S15: R_dot = -3 G0 / (7 mu R^4); Simple proportional control for R
        G0 = -(7.0 * viscosity_val * R_0_local**4 / (3.0 * T_local)) * (R_T - R_0_local) / R_0_local
        return float(np.clip(G0, -control_clamp, control_clamp)), 0.0

    target_ratio = abs(X_remaining) / R_0_local

    def F_s(s):
        val = (33.0 * s**2) / (250.0 * np.sinh(s)**2)
        val *= (sigma * (2.0 * np.cosh(s) - sigma) - 1.0)
        return np.sqrt(np.maximum(val, 0.0))

    # Find root s
    guesses = np.linspace(0.01, 15.0, 3000)
    s_val = None
    for g in guesses:
        try:
            sol = root_scalar(lambda s: F_s(s) - target_ratio, x0=g, method='newton')
            if sol.converged and sol.root > 0:
                s_val = sol.root
                break
        except Exception:
            pass

    if s_val is None:
        # Fallback: use CLF controls if PMP can't find a solution
        G0 = (5.0 * B_w / 3.0) * R_0_local**2 * (R_0_local - R_T) / R_T**2
        DG = -(11.0 * A_w / 15.0) * R_0_local**2 * (X_cur - X_T) / X_T**2
        return float(np.clip(G0, -control_clamp, control_clamp)), \
               float(np.clip(DG, -control_clamp, control_clamp))

    # Evaluate controls at tau = 0 (i.e., right now)
    Nx = 500
    x_arr = np.linspace(0, 1, Nx)
    rho = (R_T * np.sinh(s_val * x_arr)
           + R_0_local * np.sinh(s_val * (1.0 - x_arr))) / np.sinh(s_val)

    I_total = np.trapz(rho**2, x_arr)
    tau_T = T_local / (viscosity_val * I_total)

    # Derivatives at tau = 0
    drho_dx = np.gradient(rho, x_arr)
    drho_dtau_0 = drho_dx[0] / tau_T

    P_0 = (14.0 / 5.0) * drho_dtau_0          # p_R at tau=0
    p_0 = X_remaining * 700.0 / (33.0 * tau_T)  # p_X = const

    # Controls at tau = 0
    G0 = -(5.0 / 6.0) * P_0 * R_0_local**2
    DG = (11.0 / 30.0) * p_0 * R_0_local**2

    G0 = float(np.clip(G0, -control_clamp, control_clamp))
    DG = float(np.clip(DG, -control_clamp, control_clamp))

    return G0, DG



# Convert ODE controls to PDE parameterisation
def ode_to_pde_controls(G0, DG, X_com, R_width):

    R_safe = max(abs(R_width), 0.3)
    offset   = G0 - DG * X_com / R_safe
    gradient = DG / R_safe
    return float(offset), float(gradient)

# Measure droplet observables from PDE height field
def measure_droplet(height_func):

    h_proj = project(height_func, height_space)
    h_arr  = h_proj.vector().get_local()

    # Total mass (full domain — for conservation check)
    total_mass = assemble(h_proj * dx)
    if total_mass < 1e-12:
        return 0.0, initial_radius, initial_radius, 0.0

    # Build indicator: 1 inside droplet, 0 in precursor
    support_idx = np.flatnonzero(h_arr >= film_cutoff)
    indicator = project(Constant(0.0), height_space)
    if len(support_idx) > 0:
        indicator.vector()[support_idx] = 1.0

    # Support width
    support_width = assemble(indicator * dx)

    # Masked height: h_drop = h * indicator  (zero outside droplet)
    h_drop = project(h_proj * indicator, height_space)

    # Droplet mass (support only)
    drop_mass = assemble(h_drop * dx)
    if drop_mass < 1e-12:
        return 0.0, initial_radius, support_width, float(total_mass)

    # COM over droplet support only
    com = assemble(spatial_coord[0] * h_drop * dx) / drop_mass

    # Second moment over droplet support only
    m2  = assemble(spatial_coord[0]**2 * h_drop * dx) / drop_mass
    var = m2 - com * com
    moment_width = np.sqrt(20.0 * max(var, 1e-12))

    return float(com), float(moment_width), float(support_width), float(total_mass)

# forward simulation of ODE (for comparison with PDE)
def run_clf_pde():

    #  FEM setup 
    h_trial, k_trial = TrialFunctions(state_space)
    w_h, w_k         = TestFunctions(state_space)

    prev_state = Function(state_space)
    h_prev, k_prev = split(prev_state)

    old_state = Function(state_space)
    h_old, k_old = split(old_state)

    cur_state = Function(state_space)

    # Temporal interpolation
    h_extrap = 1.5 * h_prev - 0.5 * h_old
    h_theta  = theta_scheme * h_prev + (1.0 - theta_scheme) * h_trial
    k_theta  = theta_scheme * k_prev + (1.0 - theta_scheme) * k_trial

    # Control functions
    ctrl_func = Function(control_space)
    c_off, c_grad = split(ctrl_func)

    # Weak form
    F_form = (
        (h_trial - h_prev) * w_h / time_step_const
        - thin_film_flux(h_extrap, h_theta, k_theta, c_off, c_grad) * w_h.dx(0)
        + h_trial.dx(0) * w_k.dx(0)
        + k_trial * w_k
    ) * dx

    a_mat = lhs(F_form)
    b_vec = rhs(F_form)

    # Initial condition
    ic = SymmetricDroplet(initial_position, initial_radius, domain_length, precursor_thickness_val, degree=2)
    prev_state.interpolate(ic)
    old_state.interpolate(ic)

    # Measure initial state
    X_cur, R_mom, R_sup, mass0 = measure_droplet(prev_state.sub(0))
    print(f"Initial: X={X_cur:.4f}, R_moment={R_mom:.4f}, R_support={R_sup:.4f}, mass={mass0:.6f}")

    # Output writers
    hf_pvd  = File(tag + "/rundir/height.pvd", "compressed")
    hf_xdmf = XDMFFile(tag + "/rundir/height.xdmf")

    # Storage arrays
    time_hist     = [0.0]
    X_hist        = [X_cur]
    R_mom_hist    = [R_mom]
    R_sup_hist    = [R_sup]
    G0_hist       = [0.0]
    DG_hist       = [0.0]
    offset_hist   = [0.0]
    gradient_hist = [0.0]
    cost_hist     = [0.0]

    # Initial dissipation (half-step weight)
    h_prev_proj = project(h_prev, height_space)
    # We need an initial stress for dissipation — with zero control at t=0
    ctrl_func.interpolate(Constant([0.0, 0.0]))
    stress0 = interfacial_stress(h_prev, k_prev, c_off, c_grad) \
              - hamaker_like * (1.0 - precursor_thickness / h_prev) / (h_prev**3)
    diss_cost = 0.5 * time_step * assemble(h_prev**3 * stress0.dx(0)**2 * dx) / (12.0 * viscosity_val)


    current_time = time_step
    step = 0

    while current_time <= final_time + 1e-10:

        # Compute controls from current PDE state
        X_cur, R_mom, R_sup, mass = measure_droplet(prev_state.sub(0))


        # Convert to PDE parameterisation

        if args.mode == "clf":
            G0_ode, DG_ode = clf_controls(X_cur, R_sup)
        else:
            # Receding-horizon PMP: re-solve from current PDE state
            t_remaining = final_time - current_time + time_step
            G0_ode, DG_ode = pmp_controls_from_state(X_cur, R_sup, t_remaining)

        # Convert to PDE parameterisation (always use actual PDE state)
        pde_off, pde_grad = ode_to_pde_controls(G0_ode, DG_ode, X_cur, R_sup)
        # Apply controls
        ctrl_func.interpolate(Constant([pde_off, pde_grad]))

        # Solve one PDE time step 
        solve(a_mat == b_vec, cur_state)

        # Rotate history
        assign(old_state, prev_state)
        assign(prev_state, cur_state)

        h_cur, k_cur = split(cur_state)

        # Positivity check
        h_proj = project(h_cur, height_space)
        if np.amin(h_proj.vector().get_local()) < 0.0:
            print(f"WARNING: Negative height at t={current_time:.4f}, stopping.")
            break

        # Accumulate dissipation
        total_stress = interfacial_stress(h_cur, k_cur, c_off, c_grad) \
                       - hamaker_like * (1.0 - precursor_thickness / h_cur) / (h_cur**3)
        wt = 0.5 if current_time > final_time - time_step else 1.0
        diss_cost += wt * time_step * assemble(h_cur**3 * total_stress.dx(0)**2 * dx) / (12.0 * viscosity_val)

        if step % output_stride == 0:
            X_m, R_m, R_s, m = measure_droplet(cur_state.sub(0))

            time_hist.append(current_time)
            X_hist.append(X_m)
            R_mom_hist.append(R_m)
            R_sup_hist.append(R_s)
            G0_hist.append(G0_ode)
            DG_hist.append(DG_ode)
            offset_hist.append(pde_off)
            gradient_hist.append(pde_grad)
            cost_hist.append(float(diss_cost))

            hf_pvd << (cur_state.sub(0), current_time)

            if step % (5 * output_stride) == 0:
                print(f"  t={current_time:.3f}  X={X_m:.4f}  R_mom={R_m:.4f}  "
                      f"R_sup={R_s:.4f}  G0={G0_ode:.2f}  DG={DG_ode:.2f}  "
                      f"cost={float(diss_cost):.4e}")

        current_time += time_step
        step += 1

    #  Terminal diagnostics 
    X_f, R_mom_f, R_sup_f, mass_f = measure_droplet(prev_state.sub(0))
    terminal_cost = (A_w / X_T**2) * (X_f - X_T)**2 + (B_w / R_T**2) * (R_sup_f - R_T)**2
    total_cost = float(diss_cost) + float(terminal_cost)

    print("\n" + "="*60)
    print("FINAL STATE")
    print(f"  X(T) = {X_f:.6f}   (target {X_T})")
    print(f"  R_moment(T) = {R_mom_f:.6f}   (target {R_T})")
    print(f"  R_support(T) = {R_sup_f:.6f}   (target {R_T})")
    print(f"  Mass(T) = {mass_f:.6f}")
    print(f"  Dissipation W = {float(diss_cost):.6e}")
    print(f"  Terminal   T  = {float(terminal_cost):.6e}")
    print(f"  Total cost    = {total_cost:.6e}")
    print("="*60)

    # Save data files 
    np.savetxt(tag + "/rundir/X.dat",
               np.column_stack([time_hist, X_hist]),
               header="time  X(t)", fmt="%.8e")
    np.savetxt(tag + "/rundir/R_moment.dat",
               np.column_stack([time_hist, R_mom_hist]),
               header="time  R_moment(t)", fmt="%.8e")
    np.savetxt(tag + "/rundir/R_support.dat",
               np.column_stack([time_hist, R_sup_hist]),
               header="time  R_support(t)", fmt="%.8e")
    np.savetxt(tag + "/rundir/controls.dat",
               np.column_stack([time_hist, G0_hist, DG_hist, offset_hist, gradient_hist]),
               header="time  G0_ode  DG_ode  PDE_offset  PDE_gradient", fmt="%.8e")
    np.savetxt(tag + "/rundir/cost.dat",
               np.column_stack([time_hist, cost_hist]),
               header="time  cumulative_dissipation", fmt="%.8e")

    #  Plots 
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    # (A) Controls
    ax = axes[0, 0]
    ax.plot(time_hist, DG_hist, 'r-', lw=2, label=r'$\Delta G(t)$')
    ax.plot(time_hist, G0_hist, 'g-', lw=2, label=r'$G_0(t)$')
    ax.set_xlabel('t')
    ax.set_title(f'Controls ({args.mode.upper()} on PDE)')
    ax.legend()
    ax.set_xlim(0, final_time)

    # (B) State variables
    ax = axes[0, 1]
    ax.plot(time_hist, R_mom_hist, 'r-', lw=2, label=r'$R(t)$ [moment]')
    ax.plot(time_hist, R_sup_hist, 'r--', lw=1.5, label=r'$R(t)$ [support]')
    ax.plot(time_hist, X_hist, 'b-', lw=2, label=r'$X(t)$')
    ax.axhline(R_T, color='r', ls=':', alpha=0.5, label=f'$R_T={R_T}$')
    ax.axhline(X_T, color='b', ls=':', alpha=0.5, label=f'$X_T={X_T}$')
    ax.set_xlabel('t')
    ax.set_title('State variables')
    ax.legend(fontsize=8)
    ax.set_xlim(0, final_time)

    # (C) Terminal cost (T) over time
    ax = axes[1, 0]
    T_vals = [A_w * ((xv - X_T) / X_T)**2 + B_w * ((rv - R_T) / R_T)**2
              for xv, rv in zip(X_hist, R_mom_hist)]
    ax.semilogy(time_hist, T_vals, 'k-', lw=2)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\mathcal{T}(\tilde{x})$')
    ax.set_title('Terminal cost (CLF) decay')
    ax.set_xlim(0, final_time)

    # (D) Cumulative dissipation
    ax = axes[1, 1]
    ax.plot(time_hist, cost_hist, 'k-', lw=2)
    ax.set_xlabel('t')
    ax.set_ylabel(r'$\mathcal{W}(t)$')
    ax.set_title('Cumulative dissipation')
    ax.set_xlim(0, final_time)

    plt.suptitle(
        f'{args.mode.upper()} controller on PDE  |  '
        f'$X_T$={X_T}, $R_T$={R_T}, $\\gamma$={surface_tension_val}, $\\mu$={viscosity_val}',
        fontsize=13
    )
    plt.tight_layout()
    plt.savefig(tag + "/summary.png", dpi=200, bbox_inches='tight')
    plt.savefig(tag + "/summary.pdf", dpi=200, bbox_inches='tight')
    print(f"\nPlots saved to {tag}/summary.png")

    return total_cost


if __name__ == "__main__":
    cost = run_clf_pde()
    print(f"\nDone. Total cost = {cost:.6e}")
