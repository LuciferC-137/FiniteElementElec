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
                                    res: int = 500) -> np.ndarray:
        """
        Method to compute the continuous solutions of the mesh
        by interpolating according to the values of the polynomes
        calculated by the solver.

        Parameters
        ----------
            mesh: Mesh
                The mesh object.
            u: np.ndarray[float]
                The solution of the mesh.
            res: int
                The resolution of the grid. default: 500 (high)

        Returns
        -------
            np.ndarray[float]: The continuous solutions of the mesh
            (dimension: res x res)        
        """
        x = np.array([node.x for node in mesh])
        y = np.array([node.y for node in mesh])
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        # Using np.nan to make transparent the values outside the mesh
        z = np.nan * np.ones((res, res))
        
        i = 0
        for element in mesh.iter_elements():
            Logger().log_prc("Computing continuous solutions", i,
                            len(mesh.elements))
            i += 1
            pixels = Solver._pixels_in_el(element, res, x_min,
                                          x_max, y_min, y_max)
            
            nodes = element.nodes
            x1, y1 = nodes[0].x, nodes[0].y
            x2, y2 = nodes[1].x, nodes[1].y
            x3, y3 = nodes[2].x, nodes[2].y
            A = np.array([[1, x1, y1],
                          [1, x2, y2],
                          [1, x3, y3]])
            
            v = np.array([u[nodes[0].index], u[nodes[1].index],
                        u[nodes[2].index]])
            
            try:
                # Finding interpolation coefficients (of polynomes B_i)
                coeffs = np.linalg.solve(A, v)
                
                for px, py in pixels:
                    # Convert pixels to physical coordinates
                    phys_x = px / (res - 1) * (x_max - x_min) + x_min
                    phys_y = py / (res - 1) * (y_max - y_min) + y_min
                    
                    interpolated_value = coeffs[0] + coeffs[1] \
                        * phys_x + coeffs[2] * phys_y
                    z[py, px] = interpolated_value
                    
            except np.linalg.LinAlgError:
                # This error could happen if the element is degenerated
                continue

        Logger().log_prc_done("Computing continuous solutions")
        return z

    @staticmethod
    def _pixels_in_el(element: Element, res: int,
                    x_min: float, x_max: float,
                    y_min: float, y_max: float) -> list[tuple[int, int]]:
        """
        Method to find the pixels inside an element.
        
        Parameters
        ----------
            element: Element
                The element.
            res: int
                The resolution of the 'continous' array.
            x_min, x_max, y_min, y_max: float
                The minimum and maximum values of the mesh (physical).
                
        Returns
        -------
            list[tuple[int, int]]: List of coordinates of the 
            pixels inside the element.
        """
        x1, y1 = element.nodes[0].x, element.nodes[0].y
        x2, y2 = element.nodes[1].x, element.nodes[1].y
        x3, y3 = element.nodes[2].x, element.nodes[2].y

        px1, py1 = Solver._coord_to_pix(x1, y1, res, x_min, x_max, y_min, y_max)
        px2, py2 = Solver._coord_to_pix(x2, y2, res, x_min, x_max, y_min, y_max)
        px3, py3 = Solver._coord_to_pix(x3, y3, res, x_min, x_max, y_min, y_max)

        x_min_pix = max(min(px1, px2, px3), 0)
        x_max_pix = min(max(px1, px2, px3), res - 1)
        y_min_pix = max(min(py1, py2, py3), 0)
        y_max_pix = min(max(py1, py2, py3), res - 1)

        pixels = []
        # Here, we will iterate over the pixels in the small box
        # that includes the element
        for px in range(x_min_pix, x_max_pix + 1):
            for py in range(y_min_pix, y_max_pix + 1):
                # Converting pixels to physical coordinates
                x = px / (res - 1) * (x_max - x_min) + x_min
                y = py / (res - 1) * (y_max - y_min) + y_min

                # Calculating barycentric coordinates
                denom = ((y2 - y3)*(x1 - x3) + (x3 - x2)*(y1 - y3))

                if abs(denom) < 1e-10:  # Avoiding division by zero
                    continue  # Degenerated triangle

                a = ((y2 - y3)*(x - x3) + (x3 - x2)*(y - y3)) / denom
                b = ((y3 - y1)*(x - x3) + (x1 - x3)*(y - y3)) / denom
                c = 1 - a - b

                # Checking for the pixel to be inside the element
                if 0 <= a <= 1 and 0 <= b <= 1 and 0 <= c <= 1:
                    pixels.append((px, py))

        return pixels

    
    @staticmethod
    def _coord_to_pix(x: float, y: float, res: int,
                    x_min: float, x_max: float,
                    y_min: float, y_max: float) -> tuple[int, int]:
        """
        Method to convert the coordinates to pixels.

        Parameters
        ----------
            x: float
                The x coordinate.
            y: float
                The y coordinate.
            res: int
                The resolution of the grid.
            x_min: float
                The minimum x value of the mesh.
            x_max: float
                The maximum x value of the mesh.
            y_min: float
                The minimum y value of the mesh.
            y_max: float
                The maximum y value of the mesh.

        Returns
        -------
            tuple[int, int]: The pixel coordinates.
        """
        # Utiliser (res - 1) pour s'assurer que les valeurs extrêmes sont mappées aux limites du tableau
        px = int((x - x_min) / (x_max - x_min) * (res - 1))
        py = int((y - y_min) / (y_max - y_min) * (res - 1))
        
        # S'assurer que les valeurs sont dans les limites
        px = max(0, min(px, res - 1))
        py = max(0, min(py, res - 1))
        
        return px, py
    
    @staticmethod
    def compute_element_gradients(mesh : Mesh,
                                  u : np.ndarray) -> dict[int, float]:
        """
        Method to compute the gradients of the elements over the
        mesh.

        Parameters
        ----------
            mesh: Mesh
                The mesh object.
            u: np.ndarray[float]
                The solution of the mesh.

        Returns
        -------
            dict[int, float]: The gradients of the elements over the mesh.
        
        """
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