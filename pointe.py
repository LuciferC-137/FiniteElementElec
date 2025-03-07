import numpy as np
from elements import Mesh, Node, Element
from plotting import plot_mesh_anim, plot_mesh_nodes_with_controls, \
    plot_mesh_elements_with_controls

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

# Plotting
plot_mesh_nodes_with_controls(mesh)
plot_mesh_elements_with_controls(mesh)