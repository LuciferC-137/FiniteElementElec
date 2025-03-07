import numpy as np
import matplotlib.pyplot as plt
from elements import Mesh, Node, Element
from plotting import plot_mesh_anim, plot_mesh_boundary_conditions, plot_mesh_nodes_with_controls, \
    plot_mesh_elements_with_controls, plot_potential

# --------------------------- PARAMETERS ---------------------------

# Number of nodes in one line
n = 10
# Length of the square (physical)
L = 1

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
    # Compute the area of the triangle
    x1, y1 = nodes[0].x, nodes[0].y
    x2, y2 = nodes[1].x, nodes[1].y
    x3, y3 = nodes[2].x, nodes[2].y
    area = 0.5 * abs(x1*(y2 - y3) + x2*(y3 - y1) + x3*(y1 - y2))

    # Compute the derivatives of the shape functions
    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Compute the element stiffness matrix
    Ke = (1 / (4 * area)) * (np.outer(b, b) + np.outer(c, c))
    
    return Ke

def apply_boundary_conditions(K: np.ndarray, F: np.ndarray, mesh: Mesh, potential: float) -> None:
    for i in range(mesh.size()):
        node = mesh[i]
        if mesh.is_in_peak(i):
            K[i, :] = 0
            K[i, i] = 1
            F[i] = 0
        elif mesh.is_on_border(i):
            K[i, :] = 0
            K[i, i] = 1
            F[i] = potential

plot_mesh_boundary_conditions(mesh)

# Creating the rigidity matrix
K = compute_rigidity_matrix(mesh)

# Initialize the right-hand side vector
F = np.zeros(mesh.size())

# Apply boundary conditions
potential = 1.0
apply_boundary_conditions(K, F, mesh, potential)

# Now you can solve the system K * u = F to find the potential u
u = np.linalg.solve(K, F)

# Plot the potential
plot_potential(mesh, u)
