import numpy as np
from elements import Mesh, Element

RED = '\033[91m'
RESET = '\033[0m'

class Solver:
    def __init__(self, mesh: Mesh = None):
        self._mesh = mesh
    
    def set_mesh(self, mesh: Mesh):
        self._mesh = mesh
    
    def _check_mesh(self) -> bool:
        return self._mesh is not None
    
    def _check_mesh_or_abort(self) -> None:
        if not self._check_mesh():
            raise ValueError(f"{RED}Mesh if not defined, cannot solve!{RESET}")
    
    def solve_mesh(self) -> np.ndarray:
        """
        Method to solve the mesh.

        Returns
        -------
            np.ndarray[float]: The solution of the mesh (dimension: n_nodes)
        
        """
        self._check_mesh_or_abort()
        K = self._compute_rigidity_matrix()
        F = np.zeros(self._mesh.size())
        self._apply_boundary_conditions(K, F, self._mesh)
        return np.linalg.solve(K, F)

    def _compute_rigidity_matrix(self) -> np.ndarray:
        """
        Method to compute the rigity matrix. This does not take into account
        the boundary conditions, all interactions between nodes are calculated.

        Returns
        -------
            np.ndarray[float]: The rigidity matrix
            (dimension: n_nodes x n_nodes)
        """
        self._check_mesh_or_abort()
        n_nodes = self._mesh.size()
        K = np.zeros((n_nodes, n_nodes))
        for element in self._mesh.elements.values():
            element : Element
            Ke = self.compute_element_stiffness_matrix(element)
            for i_local, node_i in enumerate(element.nodes):
                for j_local, node_j in enumerate(element.nodes):
                    K[node_i.index, node_j.index] += Ke[i_local, j_local]
        return K
    
    def _apply_boundary_conditions(self, K: np.ndarray, F: np.ndarray,
                                  mesh: Mesh) -> None:
        """
        Method to apply the boundary conditions to the rigidity matrix and
        the boundary vector.

        Parameters
        ----------
            K: np.ndarray[float]
                The rigidity matrix.
            F: np.ndarray[float]
                The boundary vector.
            mesh: Mesh
                The mesh object.
        
        Returns
        -------
            None
        """
        for i in range(self._mesh.size()):
            if self._mesh[i].value is not None:
                # Removing the contributions
                K[i, :] = 0 
                K[i, i] = 1
                # Applying the boundary condition
                F[i] = mesh[i].value

    @staticmethod
    def compute_element_stiffness_matrix(element: Element) -> np.ndarray:
        """
        Method to compute the stiffness matrix of an element.

        Parameters
        ----------
            element: Element
                The element for which the stiffness matrix is to be calculated.
        
        Returns
        -------
            np.ndarray[float]: The stiffness matrix of the element
            (dimension 3 x 3)
        """
        nodes = element.nodes
        x = [node.x for node in nodes]
        y = [node.y for node in nodes]

        Pe = np.array([[1, x[0], y[0]],
                       [1, x[1], y[1]],
                       [1, x[2], y[2]]])
        Ae = np.array([[1, 0], [0, 1]])
        He = np.linalg.inv(Pe)
        h = 1 / 30
        Te = 0.5 * np.abs(np.linalg.det(Pe))
        D = np.array([[0, 1, 0], [0, 0, 1]])
        DT = np.array([[0, 0], [1, 0], [0, 1]])

        Ke = np.transpose(He) @ DT @ Ae @ D @ He * Te
        
        return Ke