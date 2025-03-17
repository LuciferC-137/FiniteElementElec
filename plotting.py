import numpy as np
from elements import CircularMesh, Element, Mesh, SquareMesh, Node
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.widgets import Slider, Button
import matplotlib.tri as tri

YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'



class Plotter:
    
    @staticmethod
    def _plot_node(node: Node, ax: Axes, color: str = "blue"):
        """
        Internal method of the plotter.
        """
        ax.scatter(node.x, node.y, color=color)

    @staticmethod
    def _plot_node_and_links(node: Node, ax: Axes,
                            color: str = "blue"):
        """
        Internal method of the plotter.
        """
        Plotter._plot_node(node, ax, color)
        for neighbor in node.neighbors:
            ax.plot([node.x, neighbor.x], [node.y, neighbor.y],
                    '-', color=color)

    @staticmethod
    def _plot_element(element: Element, ax: Axes, color: str ='blue'):
        """
        Internal method of the plotter.
        """
        x = [element.node1.x, element.node2.x, element.node3.x, element.node1.x]
        y = [element.node1.y, element.node2.y, element.node3.y, element.node1.y]
        ax.fill(x, y, color=color, alpha=0.5)
        ax.plot(x, y, '-', color=color)

    @staticmethod
    def plot_mesh_nodes_with_controls(mesh: Mesh, color_mesh: str = "blue",
                                      color_high: str = "red"):
        """
        Method to plot the mesh with one node highlithed in the defined color.
        The highlithed node will change every 0.1 seconds in the index order if
        the "auto" button is clicked (acting as a toggle button).
        The arrows "<" and ">" and the cursor can be used to select a node.

        Parameters
        ----------
            mesh : Mesh
                The mesh to be plotted.
            color_mesh : str
                The color of the mesh. Default: "blue"
            color_high : str
                The color of the highlighted node. Default: "red"
        """
        fig, ax = plt.subplots()
        ax : Axes
        plt.subplots_adjust(bottom=0.25)
        
        n = mesh.n
        current = 0

        # Initial plot
        for node in mesh:
            Plotter._plot_node_and_links(node, ax, color_mesh)
        Plotter._plot_node_and_links(mesh[0], ax, color_high)
        current_plot, = ax.plot([], [], 'ro')

        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03],
                             facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Iteration', 0, mesh.size() - 1,
                        valinit=0, valstep=1)

        # Buttons
        ax_prev = plt.axes([0.1, 0.025, 0.1, 0.04])
        ax_next = plt.axes([0.85, 0.025, 0.1, 0.04])
        ax_auto = plt.axes([0.475, 0.025, 0.05, 0.04])
        btn_prev = Button(ax_prev, '<')
        btn_next = Button(ax_next, '>')
        btn_auto = Button(ax_auto, 'Auto')

        auto_play = False

        def update(val):
            nonlocal current
            current = int(slider.val)
            ax.clear()
            for node in mesh:
                Plotter._plot_node_and_links(node, ax, color_mesh)
            Plotter._plot_node_and_links(mesh[val], ax, color_high)
            plt.draw()

        def prev_iteration(event):
            slider.set_val((slider.val - 1) % mesh.size())

        def next_iteration(event):
            slider.set_val((slider.val + 1) % mesh.size())

        def toggle_auto(event):
            nonlocal auto_play
            auto_play = not auto_play
            if auto_play:
                auto_update()

        def auto_update():
            while auto_play:
                next_iteration(None)
                plt.pause(0.1)

        slider.on_changed(update)
        btn_prev.on_clicked(prev_iteration)
        btn_next.on_clicked(next_iteration)
        btn_auto.on_clicked(toggle_auto)

        plt.show()

    @staticmethod
    def plot_mesh_elements_with_controls(mesh: Mesh, color_mesh: str = "blue",
                                         color_high: str = "red"):
        """
        Method to plot the mesh with one element highlithed in red.
        The highlithed element will change every 0.1 seconds in the
        index order if the "auto" button is clicked (acting as a toggle button).
        The arrows "<" and ">" and the cursor can be used to select an element.

        Parameters
        ----------
            mesh : Mesh
                The mesh to be plotted.
            color_mesh : str
                The color of the mesh. Default: "blue"
            color_high : str
                The color of the highlighted node. Default: "red"
    
        """
        fig, ax = plt.subplots()
        ax : Axes
        plt.subplots_adjust(bottom=0.25)
        
        n = mesh.n
        current = 0

        # Initial plot
        for node in mesh:
            Plotter._plot_node_and_links(node, ax, color_mesh)
        current_plot, = ax.plot([], [], 'ro')

        # Slider
        ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03],
                             facecolor='lightgoldenrodyellow')
        slider = Slider(ax_slider, 'Iteration', 0, len(mesh.elements) - 1,
                        valinit=0, valstep=1)

        # Buttons
        ax_prev = plt.axes([0.1, 0.025, 0.1, 0.04])
        ax_next = plt.axes([0.85, 0.025, 0.1, 0.04])
        ax_auto = plt.axes([0.475, 0.025, 0.05, 0.04])
        btn_prev = Button(ax_prev, '<')
        btn_next = Button(ax_next, '>')
        btn_auto = Button(ax_auto, 'Auto')

        auto_play = False

        def update(val):
            nonlocal current
            current = int(slider.val)
            ax.clear()
            for node in mesh:
                Plotter._plot_node_and_links(node, ax, color_mesh)
            Plotter._plot_element(mesh.elements[current], ax, color_high)
            plt.draw()

        def prev_iteration(event):
            slider.set_val((slider.val - 1) % len(mesh.elements))

        def next_iteration(event):
            slider.set_val((slider.val + 1) % len(mesh.elements))

        def toggle_auto(event):
            nonlocal auto_play
            auto_play = not auto_play
            if auto_play:
                auto_update()

        def auto_update():
            while auto_play:
                next_iteration(None)
                plt.pause(0.1)

        slider.on_changed(update)
        btn_prev.on_clicked(prev_iteration)
        btn_next.on_clicked(next_iteration)
        btn_auto.on_clicked(toggle_auto)

        plt.show()
    
    def plot_mesh_all_elements(mesh: Mesh):
        """
        Method to plot the mesh with all elements highlighted in blue.
        
        Parameters
        ----------
            mesh : Mesh
                The mesh to be plotted.
        """
        fig, ax = plt.subplots()
        ax : Axes
        for element in mesh.elements.values():
            Plotter._plot_element(element, ax, "red")
        for node in mesh:
            Plotter._plot_node_and_links(node, ax)
        plt.show()

    @staticmethod
    def plot_mesh_square_boundary(mesh: SquareMesh, color_mesh: str = "blue",
                                      color_peak: str = "green",
                                      color_bound: str = "red",
                                      color_central_node: str = "yellow"):
        """
        Method to plot the mesh with all nodes inside the peak highlighted in
        "color_peak" and the nodes of the external boundary in "color_bound".
        The central node is highlighted in "color_central_node".
        
        Parameters
        ----------
            mesh : Mesh
                The mesh to be plotted.
            color_mesh : str
                The color of the mesh. Default: "blue"
            color_peak : str
                The color of the peak nodes. Default: "green"
            color_bound : str
                The color of the boundary nodes. Default: "red"
            color_central_node : str
                The color of the central node. Default: "yellow"

        """
        if not (isinstance(mesh, SquareMesh)):
            raise ValueError(f"{RED}The mesh must be"
                             f" a SquareMesh object when using "
                             f"'plot_mesh_square_boudary'.{RESET}")
        if (mesh.n > 30):
            input(f"{YELLOW}You are trying to plot {mesh.n*mesh.n} nodes."
                f" Continue ? [Enter]{RESET}")
        fig, ax = plt.subplots()
        ax : Axes
        n = mesh.n
        # Plot blue lines in the background with lower z-order
        for node in mesh:
            Plotter._plot_node_and_links(node, ax, color_mesh)
        # Plot red nodes for border with higher z-order
        for i in range(n*n):
            if mesh.is_on_border(i):
                Plotter._plot_node(mesh[i], ax, color_bound)
                ax.scatter(mesh[i].x, mesh[i].y, color=color_bound, zorder=3)
                if mesh[i].value is not None:
                    ax.text(mesh[i].x, mesh[i].y, f'{mesh[i].value:.3f}',
                            color='black', fontsize=8, ha='center',
                            va='center', zorder=4)
        # Plot green nodes for peak with higher z-order
        for i in range(n*n):
            if mesh.is_in_peak(i):
                Plotter._plot_node(mesh[i], ax, color_peak)
                ax.scatter(mesh[i].x, mesh[i].y, color=color_peak, zorder=3)
        # Plot a yellow nodes for peak node
        ax.scatter(mesh.peak_node.x, mesh.peak_node.y,
                   color=color_central_node, zorder=3)
        plt.show()
    
    @staticmethod
    def plot_mesh_circle_boundary(mesh: CircularMesh, color_mesh: str = "blue",
                                      color_peak: str = "green",
                                      color_bound: str = "red",
                                      color_central_node: str = "yellow"):
        """
        Method to plot the mesh with all nodes inside the peak highlighted in
        "color_peak" and the nodes of the external boundary in "color_bound".
        The central node is highlighted in "color_central_node".
        
        Parameters
        ----------
            mesh : Mesh
                The mesh to be plotted.
            color_mesh : str
                The color of the mesh. Default: "blue"
            color_peak : str
                The color of the peak nodes. Default: "green"
            color_bound : str
                The color of the boundary nodes. Default: "red"
            color_central_node : str
                The color of the central node. Default: "yellow"

        """
        if not (isinstance(mesh, CircularMesh)):
            raise ValueError(f"{RED}The mesh must be"
                             f" a CircularMesh object when using "
                             f"'plot_mesh_square_boudary'.{RESET}")
        if (mesh.size() > 400):
            input(f"{YELLOW}You are trying to plot {mesh.size()} nodes."
                f" Continue ? [Enter]{RESET}")
        fig, ax = plt.subplots()
        ax : Axes
        # Plot blue lines in the background with lower z-order
        for node in mesh:
            Plotter._plot_node_and_links(node, ax, color_mesh)
        # Plot red nodes for border with higher z-order
        for node in mesh:
            if mesh.is_on_border(node.index):
                Plotter._plot_node(node, ax, color_bound)
                ax.scatter(node.x, node.y, color=color_bound, zorder=3)
                if node.value is not None:
                    ax.text(node.x, node.y, f'{node.value:.3f}',
                            color='black', fontsize=8, ha='center',
                            va='center', zorder=4)
        # Plot green nodes for peak with higher z-order
        for node in mesh:
            if mesh.is_in_peak(node.index):
                Plotter._plot_node(node, ax, color_peak)
                ax.scatter(node.x, node.y, color=color_peak, zorder=3)
        # Plot a yellow nodes for peak node
        ax.scatter(mesh[0].x, mesh[0].y,
                   color=color_central_node, zorder=3)
        plt.show()

    @staticmethod
    def plot_potential(mesh: SquareMesh, u: np.ndarray,
                       cmap: str = 'viridis') -> None:
        """
        Method to plot the potential distribution in the mesh according
        to the result array. Only nodal values are plotted.

        Parameters
        ----------
            mesh : Mesh
                The mesh object.
            u : np.ndarray[float]
                The array with the potential values for each node
                (dimension: n_nodes)
            cmap : str
                The colormap to be used. Default: 'viridis'

        """
        if not isinstance(mesh, SquareMesh):
            raise ValueError(f"{RED}The mesh must be a SquareMesh object"
                             f" when using 'plot_potential'.{RESET}")
        x = np.array([node.x for node in mesh])
        y = np.array([node.y for node in mesh])
        z = u.reshape((mesh.n, mesh.n))
        plt.figure()
        plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()),
                   origin='lower', cmap=cmap)
        plt.colorbar(label='Potential')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Potential Distribution')
        plt.show()

    @staticmethod
    def plot_electric_field(mesh: SquareMesh, u: np.ndarray,
                            cmap: str = 'viridis') -> None:
        """
        Method to plot the electric field distribution in the mesh according
        to the result array. The field is calculated by computing
        the gradient values between nodal values using
        the numpy method "gradient".

        Parameters
        ----------
            mesh : Mesh
                The mesh object.
            u : np.ndarray[float]
                The array with the potential values for each node
                (dimension: n_nodes)
            cmap : str
                The colormap to be used. Default: 'viridis'

        """
        if not isinstance(mesh, SquareMesh):
            raise ValueError(f"{RED}The mesh must be a SquareMesh object"
                             f" when using 'plot_potential'.{RESET}")
        x = np.array([node.x for node in mesh])
        y = np.array([node.y for node in mesh])
        z = u.reshape((mesh.n, mesh.n))

        dx, dy = np.gradient(z)
        magnitude = np.sqrt(dx**2 + dy**2)

        plt.figure()
        plt.imshow(magnitude, extent=(x.min(), x.max(), y.min(), y.max()),
                origin='lower', cmap=cmap)
        plt.colorbar(label='Electric Field Intensity')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Electric Field')
        plt.show()
    
    def plot_continous(mesh: Mesh, z: np.ndarray,
                                  cmap: str = 'viridis',
                                  margin: float = 0.05) -> None:
        """
        Method to plot the potential distribution in the mesh and compute 
        the intermediate values for an image of 'res' pixels in length.

        Parameters
        ----------
            mesh : Mesh
                The mesh object.
            z : np.ndarray[float]
                The array with the continuous values
            cmap : str
                The colormap to be used. Default: 'viridis'
            res : int
                The resolution of one side of the image (in pixels)

        """
        
        x = np.array([node.x for node in mesh])
        y = np.array([node.y for node in mesh])      
        
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        
        margin = margin * max(x_max - x_min, y_max - y_min)
        x_min -= margin
        x_max += margin
        y_min -= margin
        y_max += margin
        
        plt.figure(figsize=(10, 8))
        plt.imshow(z, extent=[x_min, x_max, y_min, y_max], origin='lower', 
                cmap=cmap, interpolation='bilinear')
        plt.colorbar(label='Potential')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Continous potential over the mesh')
        plt.tight_layout()
        plt.show()

    def plot_discontinuous_field(mesh: Mesh, e: np.ndarray,
                                 cmap: str ='viridis'):
        x = np.array([node.x for node in mesh])
        y = np.array([node.y for node in mesh])

        triangles = []
        E_values = []

        for idx, element in mesh.elements.items():
            n0, n1, n2 = element.nodes
            triangles.append([n0.index, n1.index, n2.index])
            E_values.append(e[idx]) 

        triangles = np.array(triangles)
        E_values = np.array(E_values)

        triang = tri.Triangulation(x, y, triangles)

        plt.figure(figsize=(10, 8))
        # 'flat' shading -> one value per triangle
        tpc = plt.tripcolor(triang, facecolors=E_values, cmap=cmap,
                            shading='flat', edgecolors='k', linewidth=0.2)
        plt.colorbar(tpc, label='|E| Electric field intensity')
        plt.title("Electric field |∇V|")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.axis("equal")
        plt.tight_layout()
        plt.show()