import os
import sys
from matplotlib import pyplot as plt
import numpy as np

# ------------------------ SIMULATED IMPORT ------------------------
# This line allows to launch the code without the need
# to install the package. This is bad practice but effective for
# demonstration purposes.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from fem_peak25.elements import Mesh, MeshBuilder
from fem_peak25.plotting import Plotter
from fem_peak25.solver import Solver

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in the circumference
n = 20
# Diameter of the circle (physical)
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
z = solver.compute_continous_solutions(mesh, u, res=200)

e = solver.compute_element_gradients(mesh, u)

# Plotting
fig, ((ax0, ax1), (ax2, ax3)) = plt.subplots(2, 2)
Plotter.plot_mesh_circle_boundary(mesh, ax = ax0)
Plotter.plot_mesh_all_elements(mesh, ax = ax1)
Plotter.plot_continous(mesh, z, ax = ax2)
Plotter.plot_discontinuous_field(mesh, e, ax = ax3)


plt.show()
Plotter.savefig(fig, 'image')