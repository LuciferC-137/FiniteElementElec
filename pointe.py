import numpy as np
from elements import Mesh, Node, Element
from plotting import plot_mesh_anim, plot_mesh_boundary_conditions,\
    plot_mesh_nodes_with_controls, \
    plot_mesh_elements_with_controls, plot_potential, plot_electric_field

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in one line
n = 30
# Length of the square (physical)
L = 1
# Maximum potential at the boudary
potential = 1.0

# ------------------------------ MAIN ------------------------------

H = L / n

# Coordinates of the nodes
x = np.linspace(0, L, n)
y = np.linspace(0, L, n)


# Creating dictionnary of nodes
nodes : dict[Node] = {}
node_index = 0
for i in range(n):
    for j in range(n):
        nodes[node_index] = Node(x[i], y[j])
        node_index+=1

# Creating mesh
mesh : Mesh = Mesh(n, nodes)

# Indexing neighbors
for i in range(n*n):
    node : Node = mesh[i]
    if not mesh.is_on_border_right(i):
        node.right = nodes[i+n]
    if not mesh.is_on_border_left(i):
        node.left = nodes[i-n]
    if not mesh.is_on_border_bottom(i):
        node.under = nodes[i-1]
    if not mesh.is_on_border_top(i):
        node.above = nodes[i+1]
    if not mesh.is_on_border_bottom(i) and not mesh.is_on_border_right(i):
        node.diag_down_right = nodes[i+n-1]
    if not mesh.is_on_border_top(i) and not mesh.is_on_border_left(i):
        node.diag_up_left = nodes[i-n+1]

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

# Creating Elements
mesh.build_elements()


def compute_rigidity_matrix(mesh: Mesh) -> np.ndarray:
    n_nodes = mesh.size()
    K = np.zeros((n_nodes, n_nodes))
    for element in mesh.elements.values():
        element : Element
        Ke = compute_element_stiffness_matrix(element)
        for i_local, node_i in enumerate(element.nodes):
            for j_local, node_j in enumerate(element.nodes):
                K[node_i.index, node_j.index] += Ke[i_local, j_local]
    return K


def compute_element_stiffness_matrix(element: Element) -> np.ndarray:
    nodes = element.nodes
    x = [node.x for node in nodes]
    y = [node.y for node in nodes]

    Pe = np.array([[1, x[0], y[0]],
                   [1, x[1], y[1]],
                   [1, x[2], y[2]]])
    Ae = np.array([[1, 0], [0, 1]])
    He = np.linalg.inv(Pe)
    Te = 0.5 / H / H
    D = np.array([[0, 1, 0], [0, 0, 1]])
    DT = np.array([[0, 0], [1, 0], [0, 1]])

    Ke = np.transpose(He) @ DT @ Ae @ D @ He * Te
    
    return Ke


def apply_boundary_conditions(K: np.ndarray, F: np.ndarray,
                               mesh: Mesh) -> None:
    for i in range(mesh.size()):
        if mesh[i].value is not None:
            # Removing the contributions
            K[i, :] = 0 
            K[i, i] = 1
            # Applying the boundary condition
            F[i] = mesh[i].value

K = compute_rigidity_matrix(mesh)

F = np.zeros(mesh.size())

apply_boundary_conditions(K, F, mesh)

u = np.linalg.solve(K, F)

#plot_mesh_boundary_conditions(mesh)
plot_potential(mesh, u)
plot_electric_field(mesh, u)