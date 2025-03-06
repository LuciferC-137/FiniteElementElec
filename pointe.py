import time
import numpy as np
import matplotlib.pyplot as plt
from elements import Mesh, Node, Element
from plotting import plot_mesh_anim, plot_mesh_with_controls

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
        node.next = nodes[i+1]
    if not mesh.is_on_border_left(i):
        node.prev = nodes[i-1]
    if not mesh.is_on_border_bottom(i):
        node.under = nodes[i-n]
    if not mesh.is_on_border_top(i):
        node.above = nodes[i+n]
    if not mesh.is_on_border_bottom(i) and not mesh.is_on_border_right(i):
        node.diag_down_right = nodes[i-n+1]
    if not mesh.is_on_border_top(i) and not mesh.is_on_border_left(i):
        node.diag_up_left = nodes[i+n-1]

# Creating Elements
elements : dict[Element] = {}
for i in range(n*n):
    if i % 2 == 0:
        if (i+1) % n != 0 and i < n*(n-1):
            elements[i] = Element(nodes[i], nodes[i+1], nodes[i+n])
    else:
        if i % n != 0 and i < n*(n-1):
            elements[i] = Element(nodes[i], nodes[i+n], nodes[i+n-1])

# Plotting
plot_mesh_with_controls(mesh)