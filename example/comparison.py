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

# --------------------------------- PARAMETERS ---------------------------------

# Number of nodes in the circumference
n = 180
# Diameter of the circle (physical) Do not change this value for consistency
L = 2
# Angle of the peak (rad)
theta0 = 3 * np.pi / 4
# Maximum potential at the boudary
potential = 1.0

# Resolution of the pictures
res = 500

# ------------------------------------ MAIN ------------------------------------
# Creating mesh
mesh : Mesh = MeshBuilder().build_circular_mesh(n, L, theta0)
print(f"Number of nodes: {mesh.size()}")

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
z = solver.compute_continous_solutions(mesh, u, res=res)

# Computing electric field
e = solver.compute_element_gradients(mesh, u)

# ----------------------- ANALYTICAL SOLUTION ----------------------------------

order = 10

def a_n(n):
    denom = (2 * n + 1)
    pi_cub = np.pi * np.pi * np.pi
    result = 32 / (denom * denom * denom * pi_cub)
    if (n % 2 == 0):
        return result
    return -result

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
        Logger().log_prc("Computing analytical potential", cnt, res * res)
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
Logger().log_prc_done("Computing analytical potential")

# Difference between the analytical and numerical solutions

pot_diff = np.nan * np.ones((res, res))
for i in range(res):
    for j in range(res):
        if np.isnan(Z[i, j]):
            pot_diff[i, j] = np.nan
        elif Z[i, j] == 0:
            pot_diff[i, j] = 0
        else:
            pot_diff[i, j] = np.abs((Z[i, j] - z[i, j]) / Z[i, j]) * 100

# ------------------------------ ELECTRIC FIELD --------------------------------
# Computing the electric field (numerical)
Ex, Ey = np.gradient(Z)
E = np.sqrt(Ex**2 + Ey**2)

def E_r_n(r, theta, n, theta0):
    return a_n(n) * (2 * n + 1) * np.pi / 2 / theta0 \
        * r**((2 * n + 1) * np.pi / 2 / theta0 - 1) \
        * np.cos((2 * n + 1) * np.pi * theta / 2 / theta0)

def E_theta_n(r, theta, n, theta0):
    return  a_n(n) * (2 * n + 1) * np.pi / 2 / theta0 \
        * r**((2 * n + 1) * np.pi / 2 / theta0 - 1) \
        * np.sin((2 * n + 1) * np.pi * theta / 2 / theta0)

Er = np.zeros((res, res))
Eth = np.zeros((res, res))

cnt = 0
for i in range(res):
    for j in range(res):
        Logger().log_prc("Computing analytical electric field", cnt, res * res)
        cnt += 1
        x, y = X[j], Y[i]
        r = np.sqrt(x**2 + y**2)
        if r <= L / 2:
            theta = np.arctan2(y, x)
            if np.abs(theta) < theta0:
                for n in range(order + 1):
                    Er[i, j] -= E_r_n(r, theta, n, theta0)
                    Eth[i, j] -= E_theta_n(r, theta, n, theta0)
        else:
            Er[i, j] = np.nan
            Eth[i, j] = np.nan
Logger().log_prc_done("Computing analytical electric field")
E = np.sqrt(Er**2 + Eth**2)

# ---------------------------- ELECTRIC FIELD DIFFERENCE -----------------------

E_minus_e = np.nan * np.ones((res, res))

x = np.array([node.x for node in mesh])
y = np.array([node.y for node in mesh]) 
x_min, x_max = np.min(x), np.max(x)
y_min, y_max = np.min(y), np.max(y)

cnt = 0
for element in mesh.iter_elements():
    Logger().log_prc("Computing electric field difference",
                     cnt, len(mesh.elements))
    pixels = Solver._pixels_in_el(element, res, x_min, x_max, y_min, y_max)
    for i, j in pixels:
        if np.isnan(E[j, i]):
            E_minus_e[j, i] = np.nan
        elif E[j, i] == 0:
            E_minus_e[j, i] = 0
        else:
            E_minus_e[j, i] = np.abs(E[j, i] - e[cnt]) / np.abs(E[j, i]) * 100
    cnt += 1
Logger().log_prc_done("Computing electric field difference")


# --------------------------------- PLOTTING -----------------------------------

plot_potential = True
plot_potential_difference = True

plot_electric_field = True
plot_electric_field_difference = True

if plot_potential:
    fig, (ax_fem, ax_ana) = plt.subplots(1, 2)
    Plotter.plot_continous(mesh, z, ax = ax_fem)
    ax_ana : Axes
    im = ax_ana.imshow(Z, origin='lower')
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
    im = ax.imshow(pot_diff, vmin = 0, vmax = 100,
                   origin='lower', cmap='coolwarm')
    fig.colorbar(im, label='Relative Difference (%)')
    ax.axis("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Potential Difference (%)')
    plt.show()
    Plotter.savefig(fig, "difference")

if plot_electric_field:
    fig, (ax_fem, ax_ana) = plt.subplots(1, 2)
    Plotter.plot_discontinuous_field(mesh, e, ax = ax_fem)
    ax_ana : Axes
    im = ax_ana.imshow(E, origin='lower')
    ax_ana.get_figure().colorbar(im, label='Electric field intensity (V/m)')
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
    im = ax.imshow(E_minus_e, origin='lower', cmap='coolwarm')
    fig.colorbar(im, label='Relative Difference (%)')
    ax.axis("equal")
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Electric field Difference (%)')
    plt.show()
    Plotter.savefig(fig, "difference_elec")
    plt.close()