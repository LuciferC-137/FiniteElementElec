import numpy as np
from elements import Mesh, MeshBuilder
from plotting import Plotter
from solver import Solver

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in one line
n = 40
# Length of the square (physical)
L = 1
# Angle of the peak (rad)
theta0 = 3 * np.pi / 4
# Maximum potential at the boudary
potential = 1.0

# ------------------------------ MAIN ------------------------------

# Creating mesh
mesh : Mesh = MeshBuilder().build_circular_mesh(n, L, theta0)

# Fixing nodes that are subject to a boudary condition
for i in range(mesh.size()):
        if mesh.is_in_peak(i):
            mesh[i].value = 0
        elif mesh.is_on_border(i):
            mesh[i].value = potential * (1 - (mesh.angle_from_center(i))**2
                                         / (theta0)**2 )

# Initializing solver
solver = Solver(mesh)

# Solving
u = solver.solve_mesh()

# Computing the continuous solution
# z = solver.compute_continous_solutions(mesh, u)
e = solver.compute_element_gradients(mesh, u)

# Plotting
Plotter.plot_continous(mesh, z)
Plotter.plot_discontinuous_field(mesh, e)