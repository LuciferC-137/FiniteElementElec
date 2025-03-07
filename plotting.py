import numpy as np
from elements import Element, Mesh, Node
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

YELLOW = '\033[93m'
RED = '\033[91m'
RESET = '\033[0m'

def plot_node(node: Node, ax, color):
    ax.scatter(node.x, node.y, color=color)


def plot_node_and_links(mesh: Mesh, ax, i, color):
    plot_node(mesh[i], ax, color)
    if mesh[i].right:
        ax.plot([mesh[i].x, mesh[i].right.x],
                [mesh[i].y, mesh[i].right.y], '-', color=color)
    if mesh[i].left:
        ax.plot([mesh[i].x, mesh[i].left.x],
                [mesh[i].y, mesh[i].left.y], '-', color=color)
    if mesh[i].above:
        ax.plot([mesh[i].x, mesh[i].above.x],
                [mesh[i].y, mesh[i].above.y], '-', color=color)
    if mesh[i].under:
        ax.plot([mesh[i].x, mesh[i].under.x],
                [mesh[i].y, mesh[i].under.y], '-', color=color)
    if mesh[i].diag_up_left:
        ax.plot([mesh[i].x, mesh[i].diag_up_left.x],
                [mesh[i].y, mesh[i].diag_up_left.y], '-', color=color)
    if mesh[i].diag_down_right:
        ax.plot([mesh[i].x, mesh[i].diag_down_right.x],
                [mesh[i].y, mesh[i].diag_down_right.y], '-', color=color)


def plot_element(element: Element, ax, color='blue'):
    x = [element.node1.x, element.node2.x, element.node3.x, element.node1.x]
    y = [element.node1.y, element.node2.y, element.node3.y, element.node1.y]
    ax.fill(x, y, color=color, alpha=0.5)
    ax.plot(x, y, '-', color=color)


def plot_mesh_anim(mesh: Mesh):
    plt.ion()
    fig, ax = plt.subplots()
    n = mesh.n
    current = 0
    for i in range(n*n):
        plot_node_and_links(ax, i, 'blue')
    while True:
        if not plt.fignum_exists(fig.number):
            break
        plot_node_and_links(ax, (current-1)%(n*n), 'blue')
        plot_node_and_links(ax, current, 'red')
        current = (current + 1)%(n*n)
        plt.draw()
        plt.pause(0.1)


def plot_mesh_nodes_with_controls(mesh: Mesh):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    n = mesh.n
    current = 0

    # Initial plot
    for i in range(n*n):
        plot_node_and_links(mesh, ax, i, 'blue')
    current_plot, = ax.plot([], [], 'ro')

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Iteration', 0, n*n-1, valinit=0, valstep=1)

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
        for i in range(n*n):
            plot_node_and_links(mesh, ax, i, 'blue')
        plot_node_and_links(mesh, ax, current, 'red')
        plt.draw()

    def prev_iteration(event):
        slider.set_val((slider.val - 1) % (n*n))

    def next_iteration(event):
        slider.set_val((slider.val + 1) % (n*n))

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


def plot_mesh_elements_with_controls(mesh: Mesh):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    n = mesh.n
    current = 0

    n_el = len(mesh.elements)

    # Initial plot
    for i in range(n*n):
        plot_node_and_links(mesh, ax, i, 'blue')
    current_plot, = ax.plot([], [], 'ro')

    # Slider
    ax_slider = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    slider = Slider(ax_slider, 'Iteration', 0, n_el, valinit=0, valstep=1)

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
        for i in range(n*n):
            plot_node_and_links(mesh, ax, i, 'blue')
        plot_element(mesh.elements[current], ax, 'red')
        plt.draw()

    def prev_iteration(event):
        slider.set_val((slider.val - 1) % n_el)

    def next_iteration(event):
        slider.set_val((slider.val + 1) % n_el)

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


def plot_mesh_boundary_conditions(mesh: Mesh):
    if (mesh.n > 30):
        input(f"{YELLOW}You are trying to plot {mesh.n*mesh.n} nodes."
              f" Continue ? [Enter]{RESET}")
    fig, ax = plt.subplots()
    n = mesh.n
    # Plot blue lines in the background with lower z-order
    for i in range(n*n):
        plot_node_and_links(mesh, ax, i, 'blue')
    # Plot red nodes for border with higher z-order
    for i in range(n*n):
        if mesh.is_on_border(i):
            plot_node(mesh[i], ax, 'red')
            ax.scatter(mesh[i].x, mesh[i].y, color='red', zorder=3)
            if mesh[i].value is not None:
                ax.text(mesh[i].x, mesh[i].y, f'{mesh[i].value:.3f}',
                        color='black', fontsize=8, ha='center',
                        va='center', zorder=4)
    # Plot green nodes for peak with higher z-order
    for i in range(n*n):
        if mesh.is_in_peak(i):
            plot_node(mesh[i], ax, 'green')
            ax.scatter(mesh[i].x, mesh[i].y, color='green', zorder=3)
    # Plot a yellow nodes for peak node
    ax.scatter(mesh.peak_node.x, mesh.peak_node.y, color='yellow', zorder=3)
    plt.show()


def plot_potential(mesh: Mesh, u: np.ndarray) -> None:
    x = np.array([node.x for node in mesh._nodes.values()])
    y = np.array([node.y for node in mesh._nodes.values()])
    z = u.reshape((mesh.n, mesh.n))

    plt.figure()
    plt.imshow(z, extent=(x.min(), x.max(), y.min(), y.max()),
               origin='lower', cmap='viridis')
    plt.colorbar(label='Potential')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Potential Distribution')
    plt.show()


def plot_electric_field(mesh: Mesh, u: np.ndarray) -> None:
    x = np.array([node.x for node in mesh._nodes.values()])
    y = np.array([node.y for node in mesh._nodes.values()])
    z = u.reshape((mesh.n, mesh.n))

    dx, dy = np.gradient(z)
    magnitude = np.sqrt(dx**2 + dy**2)

    plt.figure()
    plt.imshow(magnitude, extent=(x.min(), x.max(), y.min(), y.max()),
               origin='lower', cmap='viridis')
    plt.colorbar(label='Electric Field Intensity')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Electric Field')
    plt.show()
