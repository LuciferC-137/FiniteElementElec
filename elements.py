import numpy as np


class Node:
    def __init__(self, x, y, value = 0, node_right = None, node_left = None,
                 node_above = None, node_under = None,
                 node_diag_up_left = None, node_diag_down_right = None):
        self.x = x
        self.y = y
        self.value = value
        self._right : Node = node_right
        self._left : Node = node_left
        self._above : Node = node_above
        self._under : Node = node_under
        self._diag_up_left : Node = node_diag_up_left
        self._diag_down_right : Node = node_diag_down_right

    def distance(self, node) -> float:
        return np.sqrt((self.x - node.x) ** 2 + (self.y - node.y) ** 2)

    def __str__(self):
        return f"({self.x}, {self.y})"

    def __repr__(self):
        return f"Node({self.x}, {self.y})"
    
    # Getters and setters for nodes
    @property
    def right(self):
        return self._right
    
    @right.setter
    def right(self, node):
        self._right = node

    @property
    def left(self):
        return self._left
    
    @left.setter
    def left(self, node):
        self._left = node

    @property
    def above(self):
        return self._above
    
    @above.setter
    def above(self, node):
        self._above = node

    @property
    def under(self):
        return self._under
    
    @under.setter
    def under(self, node):
        self._under = node
    
    @property
    def diag_up_left(self):
        return self._diag_up_left
    
    @diag_up_left.setter
    def diag_up_left(self, node):
        self._diag_up_left = node

    @property
    def diag_down_right(self):
        return self._diag_down_right
    
    @diag_down_right.setter
    def diag_down_right(self, node):
        self._diag_down_right = node

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

class Mesh:
    def __init__(self, n: int, nodes : dict[Node]):
        self._n = n
        self._nodes = nodes
    
    def is_on_border_top(self, i: int) -> bool:
        return (i+1) % self._n == 0
    
    def is_on_border_right(self, i: int) -> bool:
        return i >= (self._n - 1) * self._n
    
    def is_on_border_bottom(self, i: int) -> bool:
        return i % self._n == 0
    
    def is_on_border_left(self, i: int) -> bool:
        return i < self._n

    def size(self) -> int:
        return len(self._nodes)
    
    def __getitem__(self, key) -> Node:
        return self._nodes[key]
    
    @property
    def n(self):
        return self._n