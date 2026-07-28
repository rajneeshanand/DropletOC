#!/usr/bin/env python3
# ------------------------------------------------------------------
# PDE_visualize.py
#
# Post-processing / visualization companion to PDE_Solver.py.
# Reads the CMA-ES output (xmean / xrecentbest) for one or more
# initial-condition runs, identifies the best control candidate,
# replays the forward thin-film PDE under that candidate, and
# writes:
#   - height field PVD
#   - per-timestep zavg, zgrad, Xt, Rt PVDs
#   - summary line in outdir/<case>.dat
#   - control plot (zeta.eps) and state plot (state.eps)
#
# Variable naming, parameters, weak form and dissipation prefactor are kept consistent with PDE_Solver.py.
# ------------------------------------------------------------------

from __future__ import print_function

import argparse
from subprocess import Popen, PIPE

import numpy as np
import matplotlib
matplotlib.rcParams.update({'font.size': 24})
import matplotlib.pyplot as plt
import scipy.interpolate as interp1d_lib

from fenics import *
import ufl

# ------------------------------------------------------------------
# FEniCS compilation and logging configuration
# ------------------------------------------------------------------
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["std_out_all_processes"] = False
set_log_level(30)

# ------------------------------------------------------------------
# Command-line interface
# ------------------------------------------------------------------
cli = argparse.ArgumentParser()
cli.add_argument(
    "params",
    type=str,
    help="Path to parameter file (XT, RT, gamma)"
)
args = cli.parse_args()

# ------------------------------------------------------------------
# Target objectives and surface tension from parameter file (params.txt)
# ------------------------------------------------------------------
with open(args.params, "r") as param_file:
    target_position = float(param_file.readline().strip())
    target_size     = float(param_file.readline().strip())
    surface_tension = float(param_file.readline().strip())

# ------------------------------------------------------------------
# Temporal discretization and output control
# ------------------------------------------------------------------
time_step     = 5e-3
final_time    = 1.0
control_dt    = 0.01
output_dt     = 2.0 * time_step
theta_scheme  = 0.5
output_stride = int(output_dt / time_step)

# ------------------------------------------------------------------
# Fluid and interface properties
# ------------------------------------------------------------------
viscosity            = 0.5
equilibrium_angle    = np.pi / 4.0
initial_radius       = 2.0 / np.tan(equilibrium_angle)
initial_position     = 0.0

penalty_weight       = 1.0e3                  # mu (viscosity) is also a natural scale for the penalty weight, but we keep it fixed here since viscosity is fixed
precursor_thickness  = 1.0e-2
regularization_eps   = 1.0e-8

hamaker_like = (
    3.0 * surface_tension * (precursor_thickness * np.tan(equilibrium_angle)) ** 2
)

film_cutoff         = 1.1 * precursor_thickness
temporal_smoothing  = control_dt

# ------------------------------------------------------------------
# One-dimensional computational domain
# ------------------------------------------------------------------
domain_length = 8
num_cells     = 800

mesh_1d       = IntervalMesh(num_cells, -3.0, 5.0)
spatial_coord = SpatialCoordinate(mesh_1d)

# ------------------------------------------------------------------
# Directory structure
# ------------------------------------------------------------------
parent = (
    "XT" + str(target_position)
    + "_RT" + str(target_size)
    + "/gamma" + str(surface_tension)
).replace(".", "p")


subdir   = "%s/rundirBest" % parent
hfname   = "%s/height" % subdir

outfile  = ("outdir/XT" + str(target_position)
            + "_RT" + str(target_size)).replace(".", "p") + ".dat"

initfile = ("init/best/XT" + str(target_position)
            + "_RT" + str(target_size)
            + "_gamma" + str(surface_tension)).replace(".", "p") + ".dat"

# ------------------------------------------------------------------
# Pull best candidate controls from each available CMA-ES run.
#
# In PDE_Solver.py, run_tag is fixed to "misc" (with other possible
# tags being "rand", "sol1", "sol2", "opt", "bincpop"). Edit the
# run_tags list below to include whichever sub-runs you have under
# this case.
# ------------------------------------------------------------------
run_tags = ("misc",)
# run_tags = ("misc", "rand", "sol1", "sol2", "opt")

xmean_candidates       = []   # population mean from each run
xrecentbest_candidates = []   # most recent best from each run

for tag in run_tags:
    dirname   = parent + "/" + tag
    cmaesdir  = "%s/outcmaes/" % dirname
    xmean_path       = cmaesdir + "xmean.dat"
    xrecentbest_path = cmaesdir + "xrecentbest.dat"

    # tail -1 xmean.dat
    proc = Popen(["tail", "-1", xmean_path],
                 shell=False, stderr=PIPE, stdout=PIPE)
    res, err = proc.communicate()
    if err:
        print(err.decode())
    else:
        line_xmean = np.fromstring(res.decode().strip("\n"), sep=" ")
        # CMA logger prefixes each row with 5 metadata columns
        xmean_candidates.append(np.delete(line_xmean, [0, 1, 2, 3, 4]))

    # tail -1 xrecentbest.dat
    proc = Popen(["tail", "-1", xrecentbest_path],
                 shell=False, stderr=PIPE, stdout=PIPE)
    res, err = proc.communicate()
    if err:
        print(err.decode())
    else:
        line_xrec = np.fromstring(res.decode().strip("\n"), sep=" ")
        xrecentbest_candidates.append(np.delete(line_xrec, [0, 1, 2, 3, 4]))

# ------------------------------------------------------------------
# Output writers for replay (height + per-step diagnostics)
# ------------------------------------------------------------------
height_pvd_writer = File(hfname + ".pvd", "compressed")
zavg_pvd_writer   = File(subdir + "/zavg.pvd",  "compressed")
zgrad_pvd_writer  = File(subdir + "/zgrad.pvd", "compressed")
Xt_pvd_writer     = File(subdir + "/Xt.pvd",    "compressed")
Rt_pvd_writer     = File(subdir + "/Rt.pvd",    "compressed")

# ------------------------------------------------------------------
# Finite element definitions and associated function spaces
# ------------------------------------------------------------------
height_element = FiniteElement(
    "Lagrange", mesh_1d.ufl_cell(), degree=2
)
global_control_element = FiniteElement(
    "Real", mesh_1d.ufl_cell(), degree=0
)

height_space = FunctionSpace(mesh_1d, height_element)
state_space  = FunctionSpace(mesh_1d, MixedElement([height_element, height_element]))
control_space = FunctionSpace(mesh_1d, MixedElement([global_control_element,
                                                     global_control_element]))

# Scalar space for uniform scalar fields (zavg, zgrad, Xt, Rt) in PVDs
scalar_space = FunctionSpace(mesh_1d, height_element)

# ------------------------------------------------------------------
# Promote numerical and physical parameters to FEniCS Constant objects
# ------------------------------------------------------------------
theta_scheme         = Constant(theta_scheme)
facet_normal         = FacetNormal(mesh_1d)
time_step_const      = Constant(time_step)
viscosity            = Constant(viscosity)
regularization_eps   = Constant(regularization_eps)
precursor_thickness  = Constant(precursor_thickness)
surface_tension      = Constant(surface_tension)
hamaker_like         = Constant(hamaker_like)

# ------------------------------------------------------------------
# Symmetric initial droplet profile (mirrors PDE_Solver.py)
# ------------------------------------------------------------------
class SymmetricDroplet(UserExpression):
    """
    Quadratic-cap droplet sitting on a precursor film, with
    second component holding the curvature field.
    """
    def __init__(self, center_position, droplet_radius,
                 domain_length, film_thickness, **kwargs):
        super().__init__(**kwargs)
        self.center       = center_position
        self.radius       = droplet_radius
        self.length_scale = domain_length
        self.film         = film_thickness

    def eval(self, values, spatial_point):
        x_coord    = spatial_point[0]
        left_edge  = self.center - 0.5 * self.radius
        right_edge = self.center + 0.5 * self.radius

        if (x_coord < left_edge) or (x_coord > right_edge):
            values[0] = self.film
            values[1] = 0.0
        else:
            amplitude = 1.0 - self.length_scale * self.film
            shape_factor = (
                6.0 * amplitude
                * (x_coord - left_edge)
                * (right_edge - x_coord)
                / (self.radius ** 3)
            )
            values[0] = self.film + shape_factor
            values[1] = 3.0 * amplitude / (self.radius ** 3)

    def value_shape(self):
        return (2,)

# ------------------------------------------------------------------
# Hydrodynamic mobility with precursor regularization
# ------------------------------------------------------------------
def thin_film_mobility(height_field):
    h_cubed  = height_field**3
    h_fourth = height_field * h_cubed
    base_mobility = h_cubed / (3.0 * viscosity)
    return (h_fourth * base_mobility) / (
        h_fourth + regularization_eps * base_mobility
    )

# ------------------------------------------------------------------
# Interfacial stress (capillarity + active forcing)
# ------------------------------------------------------------------
def interfacial_stress(height_field, curvature_field,
                       control_offset, control_gradient):
    active_component = height_field * (
        control_offset + spatial_coord[0] * control_gradient
    )
    capillary_component = surface_tension * curvature_field
    return active_component + capillary_component

# ------------------------------------------------------------------
# Total stress including disjoining pressure
# ------------------------------------------------------------------
def total_interfacial_stress(height_field, curvature_field,
                             control_offset, control_gradient):
    disjoining_pressure = (
        hamaker_like
        * (1.0 - precursor_thickness / height_field)
        / (height_field**3)
    )
    return (
        interfacial_stress(height_field, curvature_field,
                           control_offset, control_gradient)
        - disjoining_pressure
    )

# ------------------------------------------------------------------
# Conservative thin-film flux operator
# ------------------------------------------------------------------
def thin_film_flux(height_for_mobility, height_for_stress,
                   curvature_field, control_offset, control_gradient):
    stress_field        = interfacial_stress(height_for_stress, curvature_field,
                                             control_offset, control_gradient)
    mobility_field      = thin_film_mobility(height_for_mobility)
    disjoining_gradient = (
        hamaker_like
        * (4.0 * precursor_thickness - 3.0 * height_for_mobility)
        / (height_for_mobility**5)
    )
    return mobility_field * (
        stress_field.dx(0) - disjoining_gradient * height_for_stress.dx(0)
    )

# ------------------------------------------------------------------
# Forward thin-film PDE solver — replay mode for visualization
#
# Differs from PDE_Solver.run_forward_simulation only in the
# write_output branch, which logs zavg / zgrad / Xt / Rt history
# arrays and writes per-step PVD files for ParaView.
# ------------------------------------------------------------------
def run_forward_simulation(control_sequence, initial_profile, write_output=False):
    height_trial,    curvature_trial    = TrialFunctions(state_space)
    test_height,     test_curvature     = TestFunctions(state_space)

    previous_state = Function(state_space)
    height_prev, curvature_prev = split(previous_state)

    older_state = Function(state_space)
    height_old, curvature_old = split(older_state)

    current_state = Function(state_space)
    height_curr, curvature_curr = split(current_state)

    extrapolated_height = 1.5 * height_prev - 0.5 * height_old
    height_theta    = theta_scheme * height_prev    + (1.0 - theta_scheme) * height_trial
    curvature_theta = theta_scheme * curvature_prev + (1.0 - theta_scheme) * curvature_trial

    previous_state.interpolate(initial_profile)
    older_state.interpolate(initial_profile)

    control_field = Function(control_space)
    control_offset, control_gradient = split(control_field)

    residual_form = (
        (height_trial - height_prev) * test_height / time_step_const
        - thin_film_flux(extrapolated_height, height_theta, curvature_theta,
                         control_offset, control_gradient) * test_height.dx(0)
        + height_trial.dx(0) * test_curvature.dx(0)
        + curvature_trial * test_curvature
    ) * dx

    system_matrix = lhs(residual_form)
    system_rhs    = rhs(residual_form)

    # Initial viscous dissipation contribution
    initial_stress = total_interfacial_stress(height_prev, curvature_prev,
                                              control_offset, control_gradient)
    dissipation_cost = (
        0.5 * time_step
        * assemble(height_prev**3 * initial_stress.dx(0)**2 * dx)
        / (12.0 * float(viscosity))
    )

    # Interpolate control in time
    control_times = np.linspace(0.0, final_time, int(len(control_sequence) / 2))

    index = 0
    control_offset_history   = []
    control_gradient_history = []
    while index < len(control_sequence):
        control_offset_history.append(control_sequence[index])
        control_gradient_history.append(control_sequence[index + 1])
        index += 2
    control_offset_history   = np.asarray(control_offset_history)
    control_gradient_history = np.asarray(control_gradient_history)

    control_offset_interp   = interp1d_lib.interp1d(control_times, control_offset_history)
    control_gradient_interp = interp1d_lib.interp1d(control_times, control_gradient_history)

    # Temporal regularization
    offset_jump   = np.diff(control_offset_history)
    gradient_jump = np.diff(control_gradient_history)
    regularization_term = (
        (temporal_smoothing / control_dt)
        * (np.sum(offset_jump * offset_jump) + np.sum(gradient_jump * gradient_jump))
    )

    # Time loop
    current_time  = time_step
    step_counter  = 0

    # History arrays for state + control plots
    Xt_history     = [initial_position]
    Rt_history     = [initial_radius]
    zavg_history   = [control_offset_history[0]]
    zgrad_history  = [initial_radius * control_gradient_history[0]]

    # Track maximum height for the Jmin lower-bound estimator
    height_proj0 = project(height_prev, height_space)
    h_max = max(height_proj0.vector().get_local())

    # Initial droplet width (used in Jmin)
    H0 = assemble(height_proj0 * dx)
    D0 = np.sqrt(assemble(spatial_coord[0]**2 * height_proj0 * dx) / H0)
    D_avg = D0
    D_curr = D0  # in case the loop runs zero times

    while current_time <= final_time:
        # Update controls
        control_field.interpolate(
            Constant([
                float(control_offset_interp(current_time)),
                float(control_gradient_interp(current_time)),
            ])
        )

        solve(system_matrix == system_rhs, current_state)

        assign(older_state, previous_state)
        assign(previous_state, current_state)

        height_curr, curvature_curr = split(current_state)

        height_projected = project(height_curr, height_space)
        if np.amin(height_projected.vector().get_local()) < 0.0:
            return np.nan

        weight = 0.5 if current_time > final_time - time_step else 1.0

        stress_now = total_interfacial_stress(height_curr, curvature_curr,
                                              control_offset, control_gradient)
        dissipation_cost += (
            weight * time_step
            * assemble(height_curr**3 * stress_now.dx(0)**2 * dx)
            / (12.0 * float(viscosity))
        )

        # Track max height
        h_now = max(height_projected.vector().get_local())
        if h_now > h_max:
            h_max = h_now

        # Track droplet width (for Jmin lower bound)
        H_now = assemble(height_projected * dx)
        x_mean = assemble(spatial_coord[0] * height_projected * dx) / H_now
        x2_mean = assemble(spatial_coord[0]**2 * height_projected * dx) / H_now
        D_curr = np.sqrt(x2_mean - x_mean * x_mean)
        D_avg += D_curr

        # Diagnostics + PVD output
        if write_output and (step_counter % output_stride == 0):

            # Save height field
            height_pvd_writer << (current_state.sub(0), current_time)

            # Droplet support mask
            support_indices = np.flatnonzero(
                height_projected.vector().get_local() >= film_cutoff
            )
            support_indicator = project(Constant(0.0), height_space)
            support_indicator.vector()[support_indices] = 1.0
            droplet_size = assemble(support_indicator * dx)

            # Center of mass
            center_of_mass = (
                assemble(spatial_coord[0] * height_projected * dx)
                / assemble(height_projected * dx)
            )

            # Local control values
            z_offset_now   = float(control_offset_interp(current_time))
            z_gradient_now = float(control_gradient_interp(current_time))

            # zavg(t) only over droplet support
            zavg_now = (
                assemble(
                    (Constant(z_offset_now)
                     + spatial_coord[0] * Constant(z_gradient_now))
                    * support_indicator * dx
                ) / droplet_size
            )

            # zgrad(t): gradient control weighted by droplet size
            zgrad_now = float(z_gradient_now) * droplet_size

            # Save scalar fields as uniform fields for ParaView
            zavg_field  = project(Constant(zavg_now),       scalar_space)
            zgrad_field = project(Constant(zgrad_now),      scalar_space)
            Xt_field    = project(Constant(center_of_mass), scalar_space)
            Rt_field    = project(Constant(droplet_size),   scalar_space)

            zavg_pvd_writer  << (zavg_field,  current_time)
            zgrad_pvd_writer << (zgrad_field, current_time)
            Xt_pvd_writer    << (Xt_field,    current_time)
            Rt_pvd_writer    << (Rt_field,    current_time)

            Xt_history.append(center_of_mass)
            Rt_history.append(droplet_size)
            zavg_history.append(zavg_now)
            zgrad_history.append(zgrad_now)

        current_time += time_step
        step_counter += 1

    # ------------- Terminal cost --------------
    final_height = project(height_curr, height_space)
    support_indices = np.flatnonzero(
        final_height.vector().get_local() >= film_cutoff
    )
    support_indicator = project(Constant(0.0), height_space)
    support_indicator.vector()[support_indices] = 1.0

    final_center_of_mass = (
        assemble(spatial_coord[0] * final_height * dx)
        / assemble(final_height * dx)
    )
    final_droplet_size = assemble(support_indicator * dx)

    terminal_cost = (
        (penalty_weight / (target_position * target_position))
        * (final_center_of_mass - target_position)**2
        + (penalty_weight / (target_size * target_size))
        * (final_droplet_size - target_size)**2
    )

    # Average droplet width (for Jmin)
    D_avg = D_avg / final_time
    dD = D_curr - D0

    # Lower bound on dissipation (mirrors analysis.py)
    Jmin = (
        0.5 * float(viscosity)
        * (12.0 * final_center_of_mass**2 + 3.0 * (dD * dD / D_avg))
        / (final_time * h_max * h_max)
    )

    total_cost = dissipation_cost + terminal_cost + regularization_term
    if total_cost < 0.0 or total_cost == np.inf:
        total_cost = np.nan

    Xt_history    = np.asarray(Xt_history)
    Rt_history    = np.asarray(Rt_history)
    zavg_history  = np.asarray(zavg_history)
    zgrad_history = np.asarray(zgrad_history)

    if write_output:
        return (total_cost, dissipation_cost, terminal_cost,
                Jmin / dissipation_cost,
                Xt_history, Rt_history, zavg_history, zgrad_history)
    else:
        return total_cost

# ------------------------------------------------------------------
# Plot color palette
# ------------------------------------------------------------------
cerulean_blue       = "#056eee"
dark_green          = "#087804"
amber               = "#feb308"
vermillion          = "#f4320c"
tomato_red          = "#ec2d01"
scarlet             = "#be0119"
bright_yellow       = "#fffd01"
lavender            = "#c79fef"
violet              = "#9a0eea"
royal_purple        = "#4b006e"
lighter_royal_purple = "#660095"
indigo              = "#380282"
blue                = "#0006b1"
azure               = "#069af3"
tangerine           = "#ff9408"
light_carmine       = "#D7021E"
chestnut            = "#742802"
light_red           = "#ff474c"
bluish_purple       = "#703be7"
emerald             = "#01a049"
dark_olive          = "#3c4d03"

# ------------------------------------------------------------------
# Score every candidate, pick the best, then replay it
# ------------------------------------------------------------------
print("Initializing control sequence")

initial_profile = SymmetricDroplet(
    initial_position, initial_radius, domain_length,
    float(precursor_thickness), degree=2
)

candidate_costs = np.zeros(len(xmean_candidates))
for k in range(len(xmean_candidates)):
    candidate_costs[k] = run_forward_simulation(xmean_candidates[k], initial_profile)

print("Found minimum")
best_index = np.nanargmin(candidate_costs)

(best_total_cost,
 best_dissipation,
 best_terminal,
 best_efficiency,
 Xt_best, Rt_best,
 zavg_best, zgrad_best) = run_forward_simulation(
    xmean_candidates[best_index], initial_profile, write_output=True
)
print("Finished optimal forward run")

# ------------------------------------------------------------------
# Bond and capillary numbers (uses the FEniCS Constant -> float cast)
# ------------------------------------------------------------------
gamma_val = float(surface_tension)
Bo = np.sum(np.abs(zavg_best)  * Rt_best**2 / gamma_val) / len(zavg_best)
Ca = np.sum(np.abs(zgrad_best) * Rt_best**2 / gamma_val) / len(zgrad_best)

import os as _os
_os.makedirs("outdir", exist_ok=True)
with open(outfile, "a") as out0:
    out0.write(
        f"{gamma_val} {Bo} {Ca} "
        f"{best_total_cost} {best_dissipation} {best_terminal} {best_efficiency} "
        f"{Xt_best[-1]} {Rt_best[-1]}\n"
    )

# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------
t_plot = np.linspace(0.0, 1.0, len(zavg_best))

# Control plot (zeta_0(t) and Delta zeta(t))
plt.plot(t_plot, zavg_best,  color=blue,       linewidth=2, label=r"$\zeta_0(t)$")
plt.plot(t_plot, zgrad_best, color=vermillion, linewidth=2, label=r"$\Delta\zeta(t)$")
plt.legend()
plt.savefig("%s/zeta.eps" % parent, format="eps", dpi=1000)
plt.gcf().clear()

# State plot (X(t) and R(t))
plt.plot(t_plot, Xt_best, color=blue,       linewidth=2, label=r"$X(t)$")
plt.plot(t_plot, Rt_best, color=vermillion, linewidth=2, label=r"$R(t)$")
plt.legend()
plt.savefig("%s/state.eps" % parent, format="eps", dpi=1000)
plt.gcf().clear()
