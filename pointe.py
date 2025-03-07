import numpy as np
from elements import Mesh, Node, Element
from plotting import plot_mesh_anim, plot_mesh_boundary_conditions,\
    plot_mesh_nodes_with_controls, \
    plot_mesh_elements_with_controls, plot_potential, plot_electric_field

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in one line
n = 100
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
            mesh[i].value = potential
            mesh[i].value = potential * (1 - (mesh.angle_from_center(i)
                                              - np.pi / 4)**2
                                / (3 * np.pi / 4)**2 )

# Creating Elements
mesh.build_elements()

def compute_rigidity_matrix(mesh: Mesh) -> np.ndarray:
    n_nodes = mesh.size()
    K = np.zeros((n_nodes, n_nodes))
    for element in mesh.elements.values():
        element: Element
        nodes = [element.node1, element.node2, element.node3]
        Ke = compute_element_stiffness_matrix(nodes)
        for i in range(3):
            for j in range(3):
                K[nodes[i].index, nodes[j].index] += Ke[i, j]
    return K

def compute_element_stiffness_matrix(nodes: list[Node]) -> np.ndarray:
    x1, y1 = nodes[0].x, nodes[0].y
    x2, y2 = nodes[1].x, nodes[1].y
    x3, y3 = nodes[2].x, nodes[2].y

    # Computing the derivatives of the shape functions
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Compute the element stiffness matrix
    Ke = (1 / (4 * 2 / H / H)) * (np.outer(b, b) + np.outer(c, c))
    
    return Ke

def apply_boundary_conditions(K: np.ndarray, F: np.ndarray,
                               mesh: Mesh, potential: float) -> None:
    for i in range(mesh.size()):
        if mesh[i].value is not None:
            # Removing the contributions
            K[i, :] = 0 
            K[i, i] = 1
            # Applying the boundary condition
            F[i] = mesh[i].value

K = compute_rigidity_matrix(mesh)

F = np.zeros(mesh.size())

apply_boundary_conditions(K, F, mesh, potential)

u = np.linalg.solve(K, F)

#plot_mesh_boundary_conditions(mesh)
plot_potential(mesh, u)
plot_electric_field(mesh, u)