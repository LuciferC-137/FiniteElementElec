import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

# ------------------------ SIMULATED IMPORT ------------------------
# This line allows to launch the code without the need
# to install the package. This is bad practice but effective for
# demonstration purposes.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fem_peak25.elements import Mesh, MeshBuilder
from fem_peak25.solver import Solver
from fem_peak25.plotting import Plotter
from fem_peak25.logger import Logger

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in the circumference
n = 60
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

# --------------- ANALYTICAL SOLUTION --------------------------------

order = 10

def a_n(n, theta0):
    denom = (2 * n + 1) * np.pi
    return  1 / (denom * denom)

def phi_n(n, r, theta, theta0):
    return r**((2 * n + 1) * np.pi / 2 / theta0) \
        * np.cos((2 * n + 1) * np.pi * theta / 2 / theta0)

def V(r, theta, theta0, order):
    result = 0
    for n in range(order):
        result += a_n(n, theta0) * phi_n(n, r, theta, theta0)
    return result

# Computing the analytical solution
res = 200

X = np.linspace(-L/2, L/2, res)
Y = np.linspace(-L/2, L/2, res)

Z = np.zeros((res, res))

cnt = 0
for i in range(res):
    for j in range(res):
        Logger().log_prc("Computing analytical solutions", cnt, res * res)
        cnt += 1
        x, y = X[j], Y[i]
        r = np.sqrt(x**2 + y**2)
        if r <= L / 2:
            theta = np.arctan2(y, x)
            if np.abs(theta) >= theta0:
                Z[i,j] = 0
            else:
                Z[i, j] = V(r, theta, theta0, order)
        else:
            Z[i, j] = np.nan
Logger().log_prc_done("Computing analytical solutions")

# Plotting
fig, (ax_fem, ax_ana) = plt.subplots(1, 2)
Plotter.plot_continous(mesh, z, ax = ax_fem)
ax_ana : Axes
im = ax_ana.imshow(Z, extent=[-L, L, -L, L], origin='lower')
ax_ana.get_figure().colorbar(im, label='Potential')
ax_ana.set_xlabel('x')
ax_ana.set_ylabel('y')
ax_ana.set_title('Analytical solution')
ax_ana.axis("equal")
ax_ana.get_figure().tight_layout()
plt.show()