from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np
from math import pi, cos, sin, floor
from logger import Logger

RED = '\033[91m'
RESET = '\033[0m'

class Node:
    def __init__(self, x, y, value = None, neighbors: list['Node'] = [],
                 index: int = None):
        self._x : float = x
        self._y : float = y
        self.value : float = value
        self._neighbors : list['Node'] = []
        self._index : int = index
    
    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Node({self.x}, {self.y})"
    
    def __eq__(self, value):
        if not isinstance(value, Node):
            return False
        return self.x == value.x and self.y == value.y

    def distance(self, node : 'Node') -> float:
        return np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)
    
    @property
    def index(self) -> int:
        return self._index
    
    @index.setter
    def index(self, index: int) -> None:
        self._index = index

    @property
    def neighbors(self) -> list['Node']:
        return self._neighbors

    @neighbors.setter
    def neighbors(self, neighbors: list['Node']) -> None:
        self._neighbors = neighbors

    @property
    def x(self) -> float:
        return self._x
    
    @property
    def y(self) -> float:
        return self._y


class Element:
    def __init__(self, node1 : Node, node2 : Node, node3 : Node):
        self._node1 : Node = node1
        self._node2 : Node = node2
        self._node3 : Node = node3

    def __eq__(self, value):
        if not isinstance(value, Element):
            return False
        return self._node1 == value.node1 and self._node2 == value.node2\
            and self._node3 == value.node3

    def __str__(self):
        return f"Element({self._node1}, {self._node2}, {self._node3})"
    
    @property
    def node1(self) -> Node:
        return self._node1
    
    @property
    def node2(self) -> Node:
        return self._node2
    
    @property
    def node3(self) -> Node:
        return self._node3
    
    @property
    def nodes(self) -> list[Node]:
        return [self._node1, self._node2, self._node3]


class Mesh(ABC):
    def __init__(self, n: int, nodes : dict[Node]):
        self._n = n
        self._nodes : dict[Node] = nodes
        self._elements : dict[Element] = {}
        self._logger = Logger()
    
    def __getitem__(self, key) -> Node:
        return self._nodes[key]
    
    def __iter__(self) -> Iterator[Node]:
        return iter(self._nodes.values())
    
    def iter_elements(self) -> Iterator[Element]:
        return iter(self._elements.values())
    
    def size(self) -> int:
        return len(self._nodes)
    
    @property
    def n(self) -> int:
        return self._n
    
    @property
    def elements(self) -> dict[Element]:
        return self._elements

    def build_elements(self):
        self._elements = {}
        element_index = 0
        i = 0
        for node1_index, node1 in self._nodes.items():
            for node2_index, node2 in self._nodes.items():
                self._logger.log_prc("Building Elements",
                                     i, len(self._nodes)*len(self._nodes))
                i += 1
                node1 : Node
                node2 : Node
                if node2_index <= node1_index:
                    continue
                if node2 not in node1.neighbors:
                    continue
                for node3_index, node3 in self._nodes.items():
                    if node3_index <= node2_index \
                       or node3_index <= node1_index:
                        continue
                    if node3 not in node1.neighbors \
                       or node3 not in node2.neighbors:
                        continue
                    element = Element(node1, node2, node3)
                    self._elements[element_index] = element
                    element_index += 1
        self._logger.log_prc_done("Building Elements")
        if not self._elements:
            self._logger.raise_error("No element could be created. "
                                     "Check the nodes connectivity")

    @abstractmethod
    def is_in_peak(self, i: int) -> bool:
        pass

    @abstractmethod
    def is_on_border(self, i: int) -> bool:
        pass

    @abstractmethod
    def angle_from_center(self, i) -> float:
        pass



class SquareMesh(Mesh):
    def __init__(self, n: int, nodes : dict[Node]):
        super().__init__(n, nodes)
    
    def build_elements(self) -> None:
        # Faster method that the default one.
        element_index = 0
        for i in range(self._n-1):
            for j in range(self._n-1):
                self._logger.log_prc("Building Elements",
                                     element_index, (self._n-1) * (self._n-1))
                node1 = self._nodes[i * self._n + j]
                node2 = self._nodes[i * self._n + j + 1]
                node3 = self._nodes[(i + 1) * self._n + j]
                node4 = self._nodes[(i + 1) * self._n + j + 1]
                self._elements[element_index] = Element(node1, node2, node3)
                element_index += 1
                self._elements[element_index] = Element(node2, node3, node4)
                element_index += 1
        self._logger.log_prc_done("Building Elements")
    
    def is_in_peak(self, i: int) -> bool:
        return i % self._n <= self._n // 2 - self._n % 2\
            and i < (self._n * self._n // 2)

    def is_on_border(self, i: int) -> bool:
        return self.is_on_border_top(i) \
            or self.is_on_border_right(i) \
            or self.is_on_border_bottom(i) \
            or self.is_on_border_left(i)
    
    def is_on_border_top(self, i: int) -> bool:
        return (i+1) % self._n == 0
    
    def is_on_border_right(self, i: int) -> bool:
        return i >= (self._n - 1) * self._n
    
    def is_on_border_bottom(self, i: int) -> bool:
        return i % self._n == 0
    
    def is_on_border_left(self, i: int) -> bool:
        return i < self._n
    
    def angle_from_center(self, i) -> float:
        dx = self._nodes[i].x - self.peak_node.x
        dy = self._nodes[i].y - self.peak_node.y
        return np.arctan2(dy, dx)

    @property
    def peak_node(self) -> Node:
        if self._n % 2 == 0:
            return self._nodes[(self._n * (self._n - 1)) // 2]
        return self._nodes[(self._n * self._n) // 2 - 1]


class CircularMesh(Mesh):
    def __init__(self, n, nodes : dict[Node], theta: float = 0,):
        super().__init__(n, nodes)
        self._theta = theta

    def _ring_start(self, k: int) -> int:
        if k == 0:
            return 0
        return 1 + 3 * k * (k - 1)

    def is_on_border(self, i: int) -> bool:
        start = self._ring_start(self._n_layers)
        return i >= start

    def is_in_peak(self, i: int) -> bool:
        return np.abs(self.angle_from_center(i)) >= self._theta

    def angle_from_center(self, i: int) -> float:
        return np.arctan2(self[i].y, self[i].x)

    @property
    def _n_layers(self) -> int:
        return int(floor(self._n / 6))


class MeshBuilder:
    def __init__(self):
        self._logger = Logger()

    def build_square_mesh(self, n: int, L: float) -> SquareMesh:
        """
        Method to build a square mesh of size n x n with a side length of L.

        Parameters
        ----------
            n: int
                The number of nodes on one side of the square.
            L: float
                The side length of the square.

        Returns
        -------
            SquareMesh: The mesh object.
        """
        mesh = SquareMesh(n, self._create_square_node_dict(n, L))
        self._index_square_neighbors(mesh, n)
        mesh.build_elements()
        return mesh
    
    def _index_square_neighbors(self, mesh: SquareMesh, n) -> None:
        for i in range(n*n):
            self._logger.log_prc("Indexing Square Mesh", i, n*n)
            node : Node = mesh[i]
            node.index = i
            if not mesh.is_on_border_right(i):
                node.neighbors.append(mesh[i+n])
            if not mesh.is_on_border_left(i):
                node.neighbors.append(mesh[i-n])
            if not mesh.is_on_border_bottom(i):
                node.neighbors.append(mesh[i-1])
            if not mesh.is_on_border_top(i):
                node.neighbors.append(mesh[i+1])
            if not mesh.is_on_border_bottom(i) \
               and not mesh.is_on_border_right(i):
                node.neighbors.append(mesh[i+n-1])
            if not mesh.is_on_border_top(i) and not mesh.is_on_border_left(i):
                node.neighbors.append(mesh[i-n+1])
        self._logger.log_prc_done("Indexing Square Mesh")

    def _create_square_node_dict(self, n, L) -> dict[Node]:
        x = np.linspace(0, L, n)
        y = np.linspace(0, L, n)
        nodes : dict[Node] = {}
        node_index = 0
        for i in range(n):
            for j in range(n):
                nodes[node_index] = Node(x[i], y[j])
                node_index+=1
        return nodes

    def build_circular_mesh(self, n: int, L: float,
                            theta: float = 0) -> CircularMesh:
        """
        Method to build a square mesh of size n x n with a side length of L.

        Parameters
        ----------
            n: int
                The number of nodes on the circumference of the circle.
            L: float
                Diamater of the circle.

        Returns
        -------
            CircularMesh: The mesh object.
        """
        mesh = CircularMesh(n, self._create_circular_node_dict(n, L), theta)
        self._index_circular_neighbors(mesh)
        mesh.build_elements()
        return mesh

    def _create_circular_node_dict(self, n, L) -> dict[Node]:
        nodes : dict[Node] = {}
        r = L / 2
        center = Node(0.0, 0.0)
        center.index = 0
        nodes[0] = center

        n_layers = int(floor((n := n / 6)))
        if n_layers == 0:
            self._logger.raise_error("The number of nodes on the edge"
                                     " must be at least 6")

        current_index = 1
        layer_radius = r / n_layers
        p = 0
        for k in range(1, n_layers + 1):
            num_points = 6 * k
            angle_step = 2 * pi / num_points
            r = layer_radius * k
            for i in range(num_points):
                self._logger.log_prc("Building Circular Mesh",
                                     p, n_layers * (n_layers + 1) * 3)
                p += 1
                theta = i * angle_step
                x = r * cos(theta)
                y = r * sin(theta)
                node = Node(x, y, index = current_index)
                nodes[current_index] = node
                current_index += 1
        self._logger.log_prc_done("Building Circular Mesh")
        return nodes
    
    def _index_circular_neighbors(self, mesh: CircularMesh):
        i = 0
        for node in mesh:
            closest_nodes = []
            for neighbor in mesh:
                self._logger.log_prc("Indexing Circular Mesh",
                                     i, mesh.size() * mesh.size())
                i += 1
                if node != neighbor:
                    distance = node.distance(neighbor)
                    closest_nodes.append((neighbor, distance))

            closest_nodes.sort(key=lambda x: x[1])

            if not mesh.is_on_border(node.index):
                number_of_neighbors = 6
            else:
                number_of_neighbors = 3

            node.neighbors = [neighbor for neighbor,
                               _ in closest_nodes[:number_of_neighbors]]
        self._logger.log_prc_done("Indexing Circular Mesh")