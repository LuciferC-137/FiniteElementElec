from elements import Mesh
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button


def plot_node(mesh: Mesh, ax, i, color):
    ax.scatter(mesh[i].x, mesh[i].y, color=color)
    if mesh[i].next:
        ax.plot([mesh[i].x, mesh[i].next.x],
                [mesh[i].y, mesh[i].next.y], '-', color=color)
    if mesh[i].prev:
        ax.plot([mesh[i].x, mesh[i].prev.x],
                [mesh[i].y, mesh[i].prev.y], '-', color=color)
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
        
def plot_mesh_anim(mesh: Mesh):
    plt.ion()
    fig, ax = plt.subplots()
    n = mesh.n
    current = 0
    for i in range(n*n):
        plot_node(ax, i, 'blue')
    while True:
        if not plt.fignum_exists(fig.number):
            break
        plot_node(ax, (current-1)%(n*n), 'blue')
        plot_node(ax, current, 'red')
        current = (current + 1)%(n*n)
        plt.draw()
        plt.pause(0.1)


def plot_mesh_with_controls(mesh: Mesh):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    
    n = mesh.n
    current = 0

    # Initial plot
    for i in range(n*n):
        plot_node(mesh, ax, i, 'blue')
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
            plot_node(mesh, ax, i, 'blue')
        plot_node(mesh, ax, current, 'red')
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