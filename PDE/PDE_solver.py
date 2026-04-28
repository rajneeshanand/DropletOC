from __future__ import print_function
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp1d_lib

from fenics import *
import ufl
import cma

np.random.seed(1)                       # For reproducibility


# If we want to restart the calculation from a previous checkpoint comment on these lines

import os
import pickle
import time

start_time = time.time()
max_time = 1                   # 3 days minus 30 minutes buffer   # NOTE- WE CAN CHANGE THE max_time HERE AND WE NEED TO ALSO need to update THE max_time DOWN BELOW ####


# ------------------------------------------------------------------
# FEniCS compilation and logging configuration
# ------------------------------------------------------------------

parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["cpp_optimize"] = True
parameters["std_out_all_processes"] = False                        # Suppress duplicated output in parallel runs
set_log_level(30)                                                  # Set log level to WARNING to reduce output verbosity

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
# Target objectives and surface tension from parameter file
# ------------------------------------------------------------------

with open(args.params, "r") as param_file:
    target_position = float(param_file.readline().strip())             # desired droplet position (XT)
    target_size = float(param_file.readline().strip())                 # desired droplet size  (RT)
    surface_tension = float(param_file.readline().strip())             # surface tension coefficient (gamma)


# ------------------------------------------------------------------
# Temporal discretization and output control
# ------------------------------------------------------------------

time_step = 5e-3                                                       # time step size
final_time = 1.0                                                       # total simulation time

control_dt = 0.01                                                      # interval for control actuation
output_dt = 2.0 * time_step                                            # interval for output writing

theta_scheme = 0.5                                                     # 0 = implicit, 1 = explicit
output_stride = int(output_dt / time_step)                             # output writing stride

# ------------------------------------------------------------------
# Fluid and interface properties
# ------------------------------------------------------------------

viscosity = 0.5                                                        # dynamic viscosity
equilibrium_angle = np.pi / 4.0                                        # equilibrium contact angle

initial_radius = 2.0 / np.tan(equilibrium_angle)                       # initial droplet radius
initial_position = 0.0                                                 # initial droplet position

penalty_pos = 1.0e3                                                    # penalty weights for droplet position
penalty_size = 1.0e3                                                   # penalty weights for droplet size

precursor_thickness = 1.0e-2                                           # precursor film thickness
regularization_eps = 1.0e-8                                            # regularization parameter for disjoining pressure

hamaker_like = (                                                       # disjoining pressure strength
    3.0 * surface_tension * (precursor_thickness * np.tan(equilibrium_angle)) ** 2)

film_cutoff = 1.1 * precursor_thickness                                # cutoff thickness for film region
temporal_smoothing = control_dt                                        # temporal smoothing parameter for control actuation


# ------------------------------------------------------------------
# Evolutionary optimization (CMA-ES) configuration
# ------------------------------------------------------------------

cma_initial_sigma = 0.2                                                # initial search spread
#cma_max_iterations = 100000                                            # maximum CMA iterations
cma_max_iterations = 22                                                 # dummy max iterations
control_bound = 20.0                                                   # symmetric bound on control variables
population_size = 20                                                   # CMA population
num_parallel_cores = population_size                                   # number of parallel CPUs for CMA evaluations
#min_iter = 75000                                                       # minimum number of iterations before applying stopping criteria
min_iter = 20								# this iternation is just for testing

# ------------------------------------------------------------------
# One-dimensional computational domain
# ------------------------------------------------------------------

domain_length = 8                                                      # length of the 1D domain
num_cells = 800                                                        # number of mesh cells

mesh_1d = IntervalMesh(num_cells, -3.0, 5.0)                           # 1D uniform mesh
spatial_coord = SpatialCoordinate(mesh_1d)                             # spatial coordinate


run_tag = "misc"
# ------------------------------------------------------------------
# Directory structure for storing simulation and optimization results
# ------------------------------------------------------------------
parent = (                                                              # Encode target objectives and surface tension into a unique case path
    "XT" + str(target_position)
    + "_RT" + str(target_size)
    + "/gamma" + str(surface_tension)).replace(".", "p")

dirname = parent + "/" + run_tag                                        # Base directory for the current run

subdir = dirname + "/rundir"                                            # Directory for forward thin-film PDE outputs

cmaesdir = dirname + "/outcmaes/"                                       # Directory for CMA-ES optimization results

hfname = subdir + "/height"                                             # Base filename for droplet height outputs


#---------------------------------------------------
# Create necessary directories if they do not exist
#---------------------------------------------------
os.makedirs(dirname, exist_ok=True)
os.makedirs(subdir, exist_ok=True)
os.makedirs(cmaesdir, exist_ok=True)


# ------------------------------------------------------------------
# Output writers for droplet height evolution (ParaView compatible)
# ------------------------------------------------------------------
height_pvd_writer = File(hfname + ".pvd", "compressed")
height_xdmf_writer = XDMFFile(hfname + ".xdmf")

# ------------------------------------------------------------------
# Save computational mesh for reproducibility and post-processing
# ------------------------------------------------------------------
mesh_output_path = subdir + "/mesh.xml.gz" 
File(mesh_output_path) << mesh_1d

# ------------------------------------------------------------------
# Finite element definitions and associated function spaces
# ------------------------------------------------------------------

height_element = FiniteElement(                                          # Scalar quadratic Lagrange element for droplet height representation
    "Lagrange",
    mesh_1d.ufl_cell(),
    degree=2
)

global_control_element = FiniteElement(                                  # Global real-valued element for control variables
    "Real",
    mesh_1d.ufl_cell(),
    degree=0
)

height_space = FunctionSpace(                                            # Function space for scalar height projection
    mesh_1d,
    height_element
)

state_space = FunctionSpace(                                             # Mixed function space for droplet height and auxiliary variable
    mesh_1d,
    MixedElement([height_element, height_element])
)

control_space = FunctionSpace(                                           # Mixed function space for global control variables    
    mesh_1d,
    MixedElement([global_control_element, global_control_element])
)

# ------------------------------------------------------------------
# Promote numerical and physical parameters to FEniCS Constant objects
# ------------------------------------------------------------------

theta_scheme = Constant(theta_scheme)                                           # time-integration weighting
facet_normal = FacetNormal(mesh_1d)                                             # outward unit normal on boundaries
time_step_const = Constant(time_step)                                           # time step size

viscosity = Constant(viscosity)                                                 # dynamic viscosity
regularization_eps = Constant(regularization_eps)                               # regularization parameter
precursor_thickness = Constant(precursor_thickness)                             # precursor film thickness

surface_tension = Constant(surface_tension)                                     # surface tension coefficient
hamaker_like = Constant(hamaker_like)                                           # disjoining pressure strength

# ------------------------------------------------------------------
# Symmetric initial droplet profile (height and curvature field)
# ------------------------------------------------------------------

class SymmetricDroplet(UserExpression):
    """
    Defines a compact-support symmetric droplet profile represented by a quadratic cap sitting on a precursor film.
    The second component stores the associated curvature field.
    """

    def __init__(                                                        # Constructor
        self,                                                            # Instance reference
        center_position,                                                 # droplet center position                  
        droplet_radius,                                                  # droplet radius
        domain_length,                                                   # computational domain length
        film_thickness,                                                  # precursor film thickness
        **kwargs                                                         # additional arguments for UserExpression parent class
    ):
        super().__init__(**kwargs)                                       # Initialize parent class
        self.center = center_position                                   
        self.radius = droplet_radius
        self.length_scale = domain_length
        self.film = film_thickness

    def eval(self, values, spatial_point):                               # Evaluate droplet profile at given spatial location
        x_coord = spatial_point[0]                                       # Extract x-coordinate

        left_edge = self.center - 0.5 * self.radius                      # Compute droplet left edges
        right_edge = self.center + 0.5 * self.radius                     # Compute droplet right edges

        if (x_coord < left_edge) or (x_coord > right_edge):              
            # Outside droplet: precursor film only
            values[0] = self.film
            values[1] = 0.0
        else:
            # Inside droplet support: quadratic cap
            amplitude = 1.0 - self.length_scale * self.film
            shape_factor = (
                6.0 * amplitude
                * (x_coord - left_edge)
                * (right_edge - x_coord)
                / (self.radius ** 3)
            )

            values[0] = self.film + shape_factor                          # droplet height profile
            values[1] = 3.0 * amplitude / (self.radius ** 3)              # droplet curvature

    def value_shape(self):                                                # Shape of the vector-valued expression
        return (2,)

# ------------------------------------------------------------------
# Hydrodynamic mobility with precursor regularization
# ------------------------------------------------------------------

def thin_film_mobility(height_field):
    """
    Regularized mobility for thin-film flow.
    Prevents singular behavior as film thickness approaches zero.
    """
    h_cubed = height_field**3                                              # h^3 term
    h_fourth = height_field * h_cubed                                      # h^4 term

    base_mobility = h_cubed / (3.0 * viscosity)                            # base mobility expression

    return (h_fourth * base_mobility) / (                                  # regularized mobility
        h_fourth + regularization_eps * base_mobility
    )


# ------------------------------------------------------------------
# Interfacial stress including capillarity and active forcing
# ------------------------------------------------------------------

def interfacial_stress(height_field, curvature_field, control_offset, control_gradient):
    """
    Computes interfacial stress composed of:
    - controls
    - capillary curvature contribution
    """
    active_component = height_field * (
        control_offset + spatial_coord[0] * control_gradient               # stress component 
    )
    capillary_component = surface_tension * curvature_field                # capillary stress component

    return active_component + capillary_component                          # total interfacial stress


# ------------------------------------------------------------------
# Total stress corrected by disjoining pressure
# ------------------------------------------------------------------

def total_interfacial_stress(height_field, curvature_field, control_offset, control_gradient):
    """
    Stress corrected by long-range intermolecular (disjoining) forces.
    """
    disjoining_pressure = (
        hamaker_like
        * (1.0 - precursor_thickness / height_field)
        / (height_field**3)
    )
    return (
        interfacial_stress(height_field, curvature_field, control_offset, control_gradient)- disjoining_pressure
    )


# ------------------------------------------------------------------
# Conservative thin-film flux operator
# ------------------------------------------------------------------

def thin_film_flux(height_for_mobility, height_for_stress, curvature_field, control_offset, control_gradient):
    """
    Computes conservative thin-film flux with:
    - regularized mobility
    - controls
    - capillarity
    - disjoining pressure gradient
    """
    stress_field = interfacial_stress(height_for_stress, curvature_field, control_offset, control_gradient)
    mobility_field = thin_film_mobility(height_for_mobility)
    disjoining_gradient = (hamaker_like * (4.0 * precursor_thickness - 3.0 * height_for_mobility)/ (height_for_mobility**5))
    
    return mobility_field * (stress_field.dx(0) - disjoining_gradient * height_for_stress.dx(0))

# ------------------------------------------------------------------
# Forward thin-film PDE solver with prescribed controls
# ------------------------------------------------------------------

def run_forward_simulation(
    control_sequence,
    initial_profile,
    write_output=False
):

    """
    Advances the thin-film equation forward in time under a given
    control sequence and returns the total objective functional value.
    """  
    height_trial, curvature_trial = TrialFunctions(state_space)                 # Mixed trial functions
    test_height, test_curvature = TestFunctions(state_space)                    # Mixed test functions

    # Solution containers (current and previous steps)  
    previous_state = Function(state_space)                                      # Previous time step solution
    height_prev, curvature_prev = split(previous_state)                         # Extract previous step components

    older_state = Function(state_space)                                         # Solution from two time steps prior
    height_old, curvature_old = split(older_state)                              # Extract older step components

    current_state = Function(state_space)                                       # Current time step solution
    height_curr, curvature_curr = split(current_state)                          # Extract current step components

    # Temporal interpolation
    extrapolated_height = 1.5 * height_prev - 0.5 * height_old  
    height_theta = (theta_scheme * height_prev + (1.0 - theta_scheme) * height_trial)
    curvature_theta = (theta_scheme * curvature_prev + (1.0 - theta_scheme) * curvature_trial)

    # Initialize solution fields
    previous_state.interpolate(initial_profile)                                 # Initialize previous step
    older_state.interpolate(initial_profile)                                    # Initialize older step
    current_state  = Function(state_space)                                      # Initialize current step
 
    # Control variables (global in space, time-dependent)
    control_field = Function(control_space)                                     # Control function
    control_offset, control_gradient = split(control_field)                     # Extract control components

    # Weak form of governing equation
    residual_form = (
        (height_trial - height_prev) * test_height / time_step_const
        - thin_film_flux(extrapolated_height, height_theta, curvature_theta, control_offset, control_gradient) * test_height.dx(0)
        + height_trial.dx(0) * test_curvature.dx(0)
        + curvature_trial * test_curvature
    ) * dx

    system_matrix = lhs(residual_form)                                          # Compute system matrix left-hand side
    system_rhs = rhs(residual_form)                                             # Compute system right-hand side

    # Initial viscous dissipation contribution
    initial_stress = total_interfacial_stress(height_prev, curvature_prev, control_offset, control_gradient)

    dissipation_cost = (                                                              # Initial viscous dissipation contribution
        0.5 * time_step * assemble(height_prev**3 * initial_stress.dx(0)**2 * dx) / (3.0 * float(viscosity))
    )

    # Interpolate control in time
    control_times = np.linspace(0.0, final_time, int(len(control_sequence) / 2))

    index = 0
    control_offset_history = []
    control_gradient_history = []

    while index < len(control_sequence):
        control_offset_history.append(control_sequence[index])
        control_gradient_history.append(control_sequence[index + 1])
        index += 2

    control_offset_history = np.asarray(control_offset_history)
    control_gradient_history = np.asarray(control_gradient_history)

    # Interpolants
    control_offset_interp = interp1d_lib.interp1d(control_times, control_offset_history)
    control_gradient_interp = interp1d_lib.interp1d(control_times, control_gradient_history)

    # Temporal regularization (EXACT match)
    offset_jump = np.diff(control_offset_history)
    gradient_jump = np.diff(control_gradient_history)

    regularization_term = ((temporal_smoothing / control_dt) * (np.sum(offset_jump * offset_jump) + np.sum(gradient_jump * gradient_jump)))

    # Time loop
    current_time = time_step
    step_counter = 0

    while current_time <= final_time:

        # Update control parameters
        control_field.interpolate(
            Constant(
                [
                    float(control_offset_interp(current_time)),
                    float(control_gradient_interp(current_time)),
                ]
            )
        )

        # Solve nonlinear system
        solve(system_matrix == system_rhs, current_state)

        # Rotate solution history
        assign(older_state, previous_state)                                        # older <- previous
        assign(previous_state, current_state)                                      # previous <- current

        height_curr, curvature_curr = split(current_state)                         # Extract current step components

        # Positivity check
        height_projected = project(height_curr, height_space)   
        if np.amin(height_projected.vector().get_local()) < 0.0:
            print("Negative height")
            return np.nan

        # Trapezoidal integration weight
        weight = 0.5 if current_time > final_time - time_step else 1.0

        # Increment viscous dissipation
        stress_now = total_interfacial_stress(height_curr, curvature_curr, control_offset, control_gradient)

        dissipation_cost += (weight * time_step * assemble(height_curr**3 * stress_now.dx(0)**2 * dx) / (3.0 * float(viscosity)))

        # Output diagnostics and visualization
        if write_output and (step_counter % output_stride == 0):

            # Write droplet height field for visualization
            height_pvd_writer << (current_state.sub(0), current_time)

            # Identify droplet support region (height >= precursor cutoff)
            support_indices = np.flatnonzero(
                height_projected.vector().get_local() >= film_cutoff
            )
            support_indicator = project(Constant(0.0), height_space)
            support_indicator.vector()[support_indices] = 1.0

            # Compute global droplet observables
            total_mass = assemble(height_projected * dx)
            center_of_mass = assemble(spatial_coord[0] * height_projected * dx) / total_mass
            second_moment = assemble(spatial_coord[0] * spatial_coord[0] * height_projected * dx) / total_mass
            droplet_width = np.sqrt(20.0 * (second_moment - center_of_mass * center_of_mass))
            effective_radius = assemble(support_indicator * dx)

            # Persist diagnostics to disk
            with open(subdir + "/h.dat", "a") as fh:
                fh.write(f"{current_time} {total_mass}\n")

            with open(subdir + "/X.dat", "a") as fx:
                fx.write(f"{current_time} {center_of_mass}\n")

            with open(subdir + "/R.dat", "a") as fr:
                fr.write(f"{current_time} {effective_radius}\n")

            with open(subdir + "/D.dat", "a") as fd:
                fd.write(f"{current_time} {droplet_width}\n")
        
        # iterate time
        current_time += time_step
        step_counter += 1

    # End of time loop
    # Terminal penalty terms
    final_height = project(height_curr, height_space)

    # Identify support region using cutoff (CRITICAL STEP)
    support_indices = np.flatnonzero(final_height.vector().get_local() >= film_cutoff)
    support_indicator = project(Constant(0.0), height_space)
    support_indicator.vector()[support_indices] = 1.0

    final_center_of_mass = (assemble(spatial_coord[0] * final_height * dx) / assemble(final_height * dx))       # Final droplet center of mass
    final_droplet_size = assemble(support_indicator * dx)                                                       # Final droplet size

    # Terminal penalty (position + size)
    terminal_cost = (penalty_pos / (target_position * target_position) * (final_center_of_mass - target_position) * (final_center_of_mass - target_position)
        + penalty_size / (target_size * target_size) * (final_droplet_size - target_size) * (final_droplet_size - target_size)
        )
    
    # Total objective value
    total_cost = dissipation_cost + terminal_cost + regularization_term

    # Numerical safety check
    if total_cost < 0.0 or total_cost == np.inf:
        print("NAN/INF cost")
        total_cost = np.nan

    return total_cost


# ------------------------------------------------------------------
# Objective function for CMA-ES
# ------------------------------------------------------------------

def evaluate_cost(control_vector):
    """
    Computes the objective value corresponding to a given
    control vector by running the forward thin-film solver.
    """

    # Construct initial droplet configuration
    initial_state = SymmetricDroplet(initial_position, initial_radius, domain_length, precursor_thickness, degree=2)

    # Forward solve and cost evaluation
    cost_value = run_forward_simulation(control_vector, initial_state)

    return cost_value

# ------------------------------------------------------------------
# Deterministic initialization of control variables
# ------------------------------------------------------------------

def initialize_control_sequence(control_offset_new, control_gradient_new):
    """
    Generate a constant-in-time control sequence.
    Each control actuation consists of a pair (G_, dG_),
    repeated uniformly over the control horizon.
    """
    control_vector = []
    current_time = 0.0

    while current_time < final_time + control_dt:
        control_vector.append(control_offset_new)
        control_vector.append(control_gradient_new)

        current_time += control_dt

    return control_vector

# ------------------------------------------------------------------
# Initialize random controls uniformly distributed within bounds
# ------------------------------------------------------------------

def initialize_random_control():
    """
    Construct a randomly initialized control sequence.
    Each control entry is drawn uniformly from a bounded interval
    and repeated over the control actuation horizon.
    """
    control_vector = []
    current_time = 0.0

    while current_time < final_time + control_dt:
        random_value_0 = 0.5 * control_bound * (np.random.rand() - 0.5)
        random_value_1 = 0.5 * control_bound * (np.random.rand() - 0.5)

        control_vector.append(random_value_0)
        control_vector.append(random_value_1)

        current_time += control_dt

    return control_vector

# ------------------------------------------------------------------
# Initialize control from external file
# ------------------------------------------------------------------

def initialize_control_from_file(file_path):
    """
    Load a time-discretized control sequence from a text file.
    Each row is expected to contain two control components
    corresponding to offset and gradient terms.
    """

    loaded_controls = np.genfromtxt(file_path).tolist()

    control_vector = []
    current_time = 0.0
    row_index = 0

    while current_time < final_time + control_dt:
        control_vector.append(loaded_controls[row_index][0])
        control_vector.append(loaded_controls[row_index][1])
        row_index += 1
        current_time += control_dt

    return control_vector

print("Initializing control sequence")

if (run_tag == "rand") or (run_tag == "misc"):
    control_vector = initialize_random_control()
else:
    control_vector = initialize_control_from_file(args.init)



print("Starting CMA-ES optimization")

cma_options = {
    "tolfunrel": 1.0e-4,
    "tolfun": 1.0e-3,
    "maxiter": cma_max_iterations,
    "verb_filenameprefix": cmaesdir,
    "bounds": [-control_bound, control_bound],
    "popsize": population_size,
}

###   ---------------- If we want to restart the calculation from a previous checkpoint keep comment on these lines ----------------   ###

# CMA-ES checkpointing
if not os.path.isfile('dump.pkl'):
    cma_engine = cma.CMAEvolutionStrategy(control_vector, cma_initial_sigma, cma_options)
else:
    # if we have a checkpoint file, resume from checkpoint
    with open('dump.pkl','rb') as fp:
       cma_engine = pickle.loads(fp.read())
   
C0_median = None
Cmin_median = np.inf
#cma_engine = cma.CMAEvolutionStrategy(control_vector, cma_initial_sigma, cma_options)
with cma.fitness_transformations.EvalParallel2(evaluate_cost,number_of_processes=num_parallel_cores) as parallel_evaluator:

###   ---------------- If we do NOT want to restart the calculation from a previous checkpoint comment on these lines and comment out next loop ----------------   ###

    '''
    while (not cma_engine.stop()) or (cma_engine.countiter < min_iter): 
        candidate_controls = cma_engine.ask()                                           # get a new candidate solution
        cma_engine.tell(candidate_controls, parallel_evaluator(candidate_controls))     # evaluate and update the strategy
        
        # cma_engine.fit.fit contains the fitness values of the current population (same order as candidate_controls)        
        fits = np.array(cma_engine.fit.fit, dtype=float)    
        if cma_engine.countiter == 1:   
            C0_median = float(np.median(fits))                                          # median fitness of initial population
            Cmin_median = C0_median 

        Cmedian = float(np.median(fits))                                                # median fitness of current population
        Cmin_median = min(Cmin_median, Cmedian)                                         # min median fitness so far
        DeltaC = float(np.max(fits) - np.min(fits))                                     # max fitness - min fitness
        eps_tol = 1e-4                                                                  # tolerance for population collapse

        # Stop if population has collapsed relative to achieved improvement
        if (cma_engine.countiter >= min_iter) and (DeltaC <= eps_tol * (C0_median - Cmin_median)):
            print("Stopping: Population criterion satisfied (DeltaC test).")
            break

       
        cma_engine.disp(100)       # display info every 100th iteration
        cma_engine.logger.add(cma_engine)
    '''

###   ---------------- If we want to restart the calculation from a previous checkpoint keep comment on these lines ----------------   ###

    while ((not cma_engine.stop()) or (cma_engine.countiter < min_iter)) \
	 and ((time.time() - start_time)/60 <= max_time):                               # NOTE- ((time.time() - start_time)/60**2 <= max_time) if we want to set max_time in hours
        candidate_controls = cma_engine.ask()
        cma_engine.tell(candidate_controls, parallel_evaluator(candidate_controls))
        
        # cma_engine.fit.fit contains the fitness values of the current population 
        fits = np.array(cma_engine.fit.fit, dtype=float)
        Cmedian = float(np.median(fits))

        if C0_median is None:
            C0_median = Cmedian

#        if es.countiter == 1:
#            C0_median = float(np.median(fits))
#            Cmin_median = C0_median

        Cmin_median = min(Cmin_median, Cmedian)
        DeltaC = float(np.max(fits) - np.min(fits))
        eps_tol = 1e-4

        # Stop if population has collapsed relative to achieved improvement
        if (cma_engine.countiter >= min_iter) and (DeltaC <= eps_tol * (C0_median - Cmin_median)):
            print("Stopping: Population criterion satisfied (DeltaC test).")
            break

        cma_engine.disp(100)       # display info every 100th iteration
        cma_engine.logger.add(cma_engine)

    # always save the result in case a restart is necessary
    with open('dump.pkl','wb') as fp:
        fp.write(cma_engine.pickle_dumps())





# Report termination status and optimal objective value
print("Termination reason:", cma_engine.stop())
print("Optimal cost value:", cma_engine.result[1])

# Generate and save CMA-ES convergence history
cma_engine.logger.plot()
plt.savefig(dirname + "/cmaes_plot.eps")



###   ---------------- If we want to restart the calculation from a previous checkpoint keep comment on these lines ----------------   ###


import subprocess
import datetime

SBATCH = "/usr/local/bin/sbatch"
JOBFILE = "/home/raa524/mvk2_proj/raa524/translated_code/pde.job"

# compute time 1 minute from now
t = datetime.datetime.now() + datetime.timedelta(minutes=1)
begin_time = t.strftime("%H:%M")

cmd = [SBATCH, f"--begin={begin_time}", JOBFILE]

print("Submitting restart job:")
print(" ".join(cmd))

res = subprocess.run(cmd, capture_output=True, text=True)

print("sbatch return code:", res.returncode)
print("sbatch stdout:", res.stdout)
print("sbatch stderr:", res.stderr)

