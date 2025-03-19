import numpy as np
from .elements import Mesh, Element, Node
from .logger import Logger

RED = '\033[91m'
RESET = '\033[0m'

class Solver:
    def __init__(self, mesh: Mesh = None):
        self._mesh = mesh
        self._logger = Logger()
    
    def set_mesh(self, mesh: Mesh):
        self._mesh = mesh
    
    def _check_mesh(self) -> bool:
        return self._mesh is not None
    
    def _check_mesh_or_abort(self) -> None:
        if not self._check_mesh():
            self._logger.raise_error("Mesh is not defined, cannot solve!")

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
        u = np.linalg.solve(K, F)
        self._logger.log("Problem solved !")
        return u

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
        i = 0
        for element in self._mesh.iter_elements():
            # Printing progression
            element : Element
            self._logger.log_prc("Computing rigidity matrix", i,
                                 len(self._mesh.elements))
            i += 1
            Ke = self.compute_element_stiffness_matrix(element)
            for i_local, node_i in enumerate(element.nodes):
                for j_local, node_j in enumerate(element.nodes):
                    K[node_i.index, node_j.index] += Ke[i_local, j_local]
        self._logger.log_prc_done("Computing rigidity matrix")
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
                self._logger.log_prc("Applying boundary conditions", i,
                                     self._mesh.size())
                # Removing the contributions
                K[i, :] = 0 
                K[i, i] = 1
                # Applying the boundary condition
                F[i] = mesh[i].value
        self._logger.log_prc_done("Applying boundary conditions")

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
        Te = 0.5 * np.abs(np.linalg.det(Pe))
        D = np.array([[0, 1, 0], [0, 0, 1]])
        DT = np.array([[0, 0], [1, 0], [0, 1]])

        Ke = np.transpose(He) @ DT @ Ae @ D @ He * Te
        
        return Ke
    
    @staticmethod
    def compute_continous_solutions(mesh: Mesh, u: np.ndarray,
                                    res: int = 500,
                                    margin: float = 0.05) -> np.ndarray:
        
        x = np.array([node.x for node in mesh])
        y = np.array([node.y for node in mesh])
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        margin_val = margin * max(x_max - x_min, y_max - y_min)
        x_min -= margin_val
        x_max += margin_val
        y_min -= margin_val
        y_max += margin_val
        
        x_grid = np.linspace(x_min, x_max, res)
        y_grid = np.linspace(y_min, y_max, res)
        X, Y = np.meshgrid(x_grid, y_grid)

        z = np.zeros_like(X)
        
        for i in range(res):
            for j in range(res):
                Logger().log_prc("Computing continuous solutions",
                                 i * res + j, res * res)
                pixel_x, pixel_y = X[i, j], Y[i, j]
                found = False
                
                for element_idx, element in mesh.elements.items():
                    nodes : list[Node] = element.nodes
                    x1, y1 = nodes[0].x, nodes[0].y
                    x2, y2 = nodes[1].x, nodes[1].y
                    x3, y3 = nodes[2].x, nodes[2].y
                    
                    denominator = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
                    
                    if abs(denominator) < 1e-10:
                        continue
                    
                    lambda1 = ((y2 - y3) * (pixel_x - x3) + (x3 - x2)
                            * (pixel_y - y3)) / denominator
                    lambda2 = ((y3 - y1) * (pixel_x - x3) + (x1 - x3)
                            * (pixel_y - y3)) / denominator
                    lambda3 = 1 - lambda1 - lambda2
                    
                    if 0 <= lambda1 <= 1 and 0 <= lambda2 <= 1 \
                       and 0 <= lambda3 <= 1:
                        potential = lambda1 * u[nodes[0].index] + lambda2 \
                            * u[nodes[1].index] + lambda3 * u[nodes[2].index]
                        z[i, j] = potential
                        found = True
                        break
                
                if not found:
                    z[i, j] = np.nan
        Logger().log_prc_done("Computing continuous solutions")
        return z
    
    @staticmethod
    def compute_element_gradients(mesh : Mesh, u : np.ndarray):
        e = {}
        for element_idx, element in mesh.elements.items():
            nodes : list[Node] = element.nodes
            x1, y1 = nodes[0].x, nodes[0].y
            x2, y2 = nodes[1].x, nodes[1].y
            x3, y3 = nodes[2].x, nodes[2].y
            A = np.array([
                [1, x1, y1],
                [1, x2, y2],
                [1, x3, y3]
            ])
            v = np.array([u[nodes[0].index], u[nodes[1].index], u[nodes[2].index]])
            
            try:
                coeffs = np.linalg.solve(A, v)
                ex, ey = -coeffs[1], -coeffs[2]
            except np.linalg.LinAlgError:
                ex, ey = 0, 0

            norm = np.sqrt(ex**2 + ey**2)
            e[element_idx] = norm
        return e