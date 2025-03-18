import numpy as np
from elements import Mesh, MeshBuilder
from plotting import Plotter
from solver import Solver
from logger import Logger

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in one line
n = 100
# Length of the square (physical)
L = 1
# Angle of the peak (rad)
theta0 = 3 * np.pi / 4
# Maximum potential at the boudary
potential = 1.0

# ------------------------------ MAIN ------------------------------
Logger().log("Creating mesh...")
# Creating mesh
mesh : Mesh = MeshBuilder().build_circular_mesh(n, L, theta0)

Logger().log("Setting boundary conditions...")
# Fixing nodes that are subject to a boudary condition
for i in range(mesh.size()):
        if mesh.is_in_peak(i):
            mesh[i].value = 0
        elif mesh.is_on_border(i):
            mesh[i].value = potential * (1 - (mesh.angle_from_center(i))**2
                                         / (theta0)**2 )

# Initializing solver
solver = Solver(mesh)

Logger().log("Solving...")
# Solving
u = solver.solve_mesh()

Logger().log("Computing continuous solutions...")
# Computing the continuous solution
# z = solver.compute_continous_solutions(mesh, u)
e = solver.compute_element_gradients(mesh, u)

Logger().log("Plotting...")
# Plotting
# Plotter.plot_continous(mesh, z)
Plotter.plot_discontinuous_field(mesh, e)