import numpy as np
from fem_peak25.elements import Mesh, MeshBuilder
from fem_peak25.plotting import Plotter
from fem_peak25.solver import Solver

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in one line
n = 10
# Length of the square (physical)
L = 1
# Maximum potential at the boudary
potential = 1.0

# ------------------------------ MAIN ------------------------------

# Creating mesh
mesh : Mesh = MeshBuilder().build_square_mesh(n, L)

# Fixing nodes that are subject to a boudary condition
for i in range(mesh.size()):
        if mesh.is_in_peak(i):
            mesh[i].value = 0
        elif mesh.is_on_border(i):
            # Here, the potential is calculated as a function of the angle
            # from the center of the square (the peak). The angle
            # 3 pi / 4 corresponds to the bottom left corner witch is already
            # at V=0, so we must ensure the border's potential decreases from
            # its maximum value to 0 when it reaches the peak.
            mesh[i].value = potential * (1 - (mesh.angle_from_center(i)
                                              - np.pi / 4)**2
                                         / (3 * np.pi / 4)**2 )

# Initializing solver
solver = Solver(mesh)

u = solver.solve_mesh()

# Plotting
Plotter.plot_potential(mesh, u)
Plotter.plot_electric_field(mesh, u)