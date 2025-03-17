from abc import ABC, abstractmethod
from typing import Iterator
import numpy as np


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
    
    @abstractmethod
    def build_elements(self) -> None:
        pass

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
        self._n = n
        self._nodes : dict[Node] = nodes
        self._elements : dict[Element] = {}
    
    def build_elements(self) -> None:
        element_index = 0
        for i in range(self._n-1):
            for j in range(self._n-1):
                node1 = self._nodes[i * self._n + j]
                node2 = self._nodes[i * self._n + j + 1]
                node3 = self._nodes[(i + 1) * self._n + j]
                node4 = self._nodes[(i + 1) * self._n + j + 1]
                self._elements[element_index] = Element(node1, node2, node3)
                element_index += 1
                self._elements[element_index] = Element(node2, node3, node4)
                element_index += 1
    
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


class MeshBuilder:
    def __init__(self):
        pass

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
            Mesh: The mesh object.
        """
        mesh = SquareMesh(n, self._create_square_node_dict(n, L))
        self._index_square_neighbors(mesh, n)
        return mesh

    def _index_square_neighbors(self, mesh: SquareMesh, n) -> None:
        for i in range(n*n):
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