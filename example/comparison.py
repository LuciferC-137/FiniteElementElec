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
n = 180
# Diameter of the circle (physical) Do not change this value for consistency
L = 2
# Angle of the peak (rad)
theta0 = 3 * np.pi / 4
# Maximum potential at the boudary
potential = 1.0

# Resolution of the pictures
res = 400

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
#z = solver.compute_continous_solutions(mesh, u, res=res)
z = 0

# Computing electric field
e = solver.compute_element_gradients(mesh, u)

# --------------- ANALYTICAL SOLUTION --------------------------------

order = 10

def a_n(n):
    denom = (2 * n + 1)
    pi_cub = np.pi * np.pi * np.pi
    res = 32 / (denom * denom * denom * pi_cub)
    if (n % 2 == 0):
        return res
    return -res

def phi_n(n, r, theta, theta0):
    return r**((2 * n + 1) * np.pi / 2 / theta0) \
        * np.cos((2 * n + 1) * np.pi * theta / 2 / theta0)

def V(r, theta, theta0, order):
    result = 0
    for n in range(order + 1):
        result += a_n(n) * phi_n(n, r, theta, theta0)
    return result

# Computing the analytical solution

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

# Computing electric field
Ex, Ey = np.gradient(Z)
E = np.sqrt(Ex**2 + Ey**2)

E_minus_e = np.zeros((res, res))
cnt = 0
for i in range(res):
    for j in range(res):
        Logger().log_prc("Computing electric field difference",
                         cnt, res * res)
        cnt += 1
        x, y = X[j], Y[i]
        found = False
        for element_idx, element in mesh.elements.items():
            nodes = element.nodes
            x1, y1 = nodes[0].x, nodes[0].y
            x2, y2 = nodes[1].x, nodes[1].y
            x3, y3 = nodes[2].x, nodes[2].y

            denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
            if abs(denominator) < 1e-10:
                continue

            lambda1 = ((y2 - y3) * (x - x3) + (x3 - x2) * (y - y3)) / denominator
            lambda2 = ((y3 - y1) * (x - x3) + (x1 - x3) * (y - y3)) / denominator
            lambda3 = 1 - lambda1 - lambda2

            if 0 <= lambda1 <= 1 and 0 <= lambda2 <= 1 and 0 <= lambda3 <= 1:
                E_minus_e[i, j] = E[i, j] - e[element_idx]
                found = True
                break

        if not found:
            E_minus_e[i, j] = np.nan
Logger().log_prc_done("Computing electric field difference")


# --------------------------- PLOTTING ------------------------------

plot_potential = False
plot_potential_difference = False

plot_electric_field = True
plot_electric_field_difference = True

if plot_potential:
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
    Plotter.savefig(fig, "comparison")
    plt.close()

if plot_potential_difference:
    fig, ax = plt.subplots()
    ax : Axes
    im = ax.imshow(np.abs(Z - z.reshape(res, res)),
                extent=[-L, L, -L, L], origin='lower', cmap='coolwarm')
    fig.colorbar(im, label='Difference')
    ax.axis("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Difference')
    plt.show()
    Plotter.savefig(fig, "difference")

if plot_electric_field:
    fig, (ax_ana, ax_fem) = plt.subplots(1, 2)
    Plotter.plot_discontinuous_field(mesh, e, ax = ax_fem)
    ax_ana : Axes
    im = ax_ana.imshow(E, extent=[-L, L, -L, L], origin='lower')
    ax_ana.get_figure().colorbar(im, label='Potential')
    ax_ana.set_xlabel('x')
    ax_ana.set_ylabel('y')
    ax_ana.set_title('Analytical solution')
    ax_ana.axis("equal")
    ax_ana.get_figure().tight_layout()
    plt.show()
    Plotter.savefig(fig, "comparison_elec")
    plt.close()

if plot_electric_field_difference:
    fig, ax = plt.subplots()
    ax : Axes
    im = ax.imshow(np.abs(E_minus_e),
                extent=[-L, L, -L, L], origin='lower', cmap='coolwarm')
    fig.colorbar(im, label='Difference')
    ax.axis("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Difference')
    plt.show()
    Plotter.savefig(fig, "difference_elec")
    plt.close()