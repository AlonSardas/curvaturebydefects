import os.path

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import hoomdlattice
from latticedefects.utils import plotutils
from latticedefects.hoomdlattice import Lattice, do_relaxation
from latticedefects.latticegenerator import TriangularLatticeGenerator, calc_metric_curvature_triangular_lattice

FIGURE_PATH = "../Figures/MD-simulations"


def sphere_by_inclusions():
    nx, ny = 40, 40
    lattice = TriangularLatticeGenerator(nx, ny, inclusion_d=1.1)
    lattice.set_z_to_sphere(radius=60)

    factor = 0.00007
    # factor = 0.0
    disk_width = 5
    padding = 0
    angle = 0
    # lattice.add_inclusion_defect(ny//2, nx//2)
    C0 = -0.05

    # TODO: fix the plot for this distribution, I think I didn't treat the radial
    # TODO: dependency correctly

    for r in range(disk_width, nx, disk_width):
        disk_area = np.pi * r ** 2 - np.pi * (r - disk_width) ** 2
        defects_in_disk = round((factor * (r - disk_width / 2) ** 2 + C0) * disk_area)
        print(f"r={r}, disk area: {disk_area}, defect in disk: {defects_in_disk}")
        if defects_in_disk <= 0:
            continue
        for d in range(defects_in_disk):
            x = round(nx / 2 + np.cos(angle) * (r - disk_width / 2))
            y = round(ny / 2 + np.sin(angle) * (r - disk_width / 2))
            angle += 2 * np.pi / defects_in_disk
            if x < padding or y < padding or x >= nx - padding or y >= ny - padding:
                continue
            print(f"Putting SW at {x},{y}, for angle={angle / np.pi * 180 % 360:.3f}")
            try:
                lattice.add_inclusion_defect(y, x)
            except RuntimeError as e:
                print("warning: ", e.args)
        angle += 2 * np.pi / defects_in_disk / 2

    snapshot = do_relaxation(lattice.frame, lattice.harmonic, lattice.dihedrals)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    hoomdlattice.plot_dots(ax, snapshot)
    # hoomdlattice.plot_bonds(ax, snapshot)
    plotutils.set_axis_scaled(ax)
    ax.set_zlim(-5, 5)

    fig, ax = plt.subplots()
    Ks, g11, g12, g22 = calc_metric_curvature_triangular_lattice(
        snapshot.particles.position, nx, ny
    )
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    plt.show()


def cone_by_traceless_quadrupoles():
    nx, ny = 40, 40
    lattice_gen = TriangularLatticeGenerator(nx, ny)
    # lattice.set_z_to_noise()
    lattice_gen.set_z_to_sphere(radius=100)

    defects_jumps = 3
    padding = 0
    for x in range(nx // 2, nx - padding, defects_jumps):
        for y in range(padding, ny - padding, defects_jumps):
            # if (x - nx // 2) >= abs(y - ny // 2):
            if (x - nx // 2) >= abs(y - ny // 2):
                print(f"Putting SW at {x},{y}")
                try:
                    lattice_gen.add_SW_defect(y, x)
                except RuntimeError as e:
                    print("warning: ", e.args)

    lattice = lattice_gen.generate_lattice()
    snapshot = lattice.do_relaxation()

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    # lattice.plot_dots(ax)
    lattice.plot_bonds(ax)
    ax.set_zlim(-5, 5)

    fig, ax = plt.subplots()
    Ks, g11, g12, g22 = calc_metric_curvature_triangular_lattice(
        snapshot.particles.position, nx, ny
    )
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    plt.show()


def old_generate_sphere_by_traceless_quadrupoles(nx, ny):
    defects_x_jumps = 6
    padding = 1
    factor = 0.0001
    middle_x = nx / 2
    x0 = middle_x - defects_x_jumps * int((middle_x - padding) / defects_x_jumps)
    for x in range(int(x0), nx - padding, defects_x_jumps):
        y_defects = (ny - 2 * padding) * factor * (x - nx / 2) ** 2
        if y_defects > ny:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{x}, ny={ny}, defects={y_defects}"
            )
        print(f"putting in x={x}, {y_defects} defects")
        for yy in np.linspace(padding, ny - padding, int(y_defects) + 2)[1:-1]:
            y = int(yy)
            print(f"Putting SW at {x},{y}")
            try:
                lattice.add_SW_defect(y, x)
            except RuntimeError as e:
                print("warning: ", e.args)


def create_lattice_for_sphere_by_traceless_quadrupoles(nx, ny,
                                                       defects_x_jumps=4,
                                                       padding=1,
                                                       factor=0.0001):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    middle_x = nx // 2
    max_x = nx - padding + 1
    xs_right = np.arange(middle_x, max_x, defects_x_jumps)
    print(xs_right)
    xs_left = -xs_right[1:] + 2 * middle_x
    xs = np.append(xs_left[::-1], xs_right)
    print(f"defects at {xs}")
    for x in xs:
        y_defects = round((ny - 2 * padding) * factor * (x - nx / 2) ** 2)
        x_index = round(x)
        if y_defects > ny:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{x_index}, ny={ny}, defects={y_defects}"
            )
        print(f"putting in x={x_index}, {y_defects} defects")
        for yy in np.linspace(padding, ny - padding, int(y_defects) + 2)[1:-1]:
            y = round(yy)
            print(f"Putting SW at {x_index},{y}")
            lattice_gen.add_SW_defect(y, x_index)
    return lattice_gen


def create_lattice_for_negative_K_SW(nx, ny,
                                     defects_y_jumps=4,
                                     padding=1,
                                     factor=0.0001):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    middle_y = ny // 2
    max_y = ny - padding + 1
    ys_right = np.arange(middle_y, max_y, defects_y_jumps)
    ys_left = -ys_right[1:] + 2 * middle_y
    ys = np.append(ys_left[::-1], ys_right) - 1
    print(f"defects at {ys}")
    for y in ys:
        x_defects = round((nx - 2 * padding) * factor * (y - ny / 2) ** 2)
        y_index = round(y)
        if x_defects > nx:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{y_index}, nx={nx}, defects={x_defects}"
            )
        print(f"putting in y={y_index}, {x_defects} defects")
        for xx in np.linspace(padding, nx - padding, int(x_defects) + 2)[1:-1]:
            x = round(xx)
            print(f"Putting SW at {y_index},{x}")
            lattice_gen.add_SW_defect(y_index, x)
    return lattice_gen


def plot_sphere_by_traceless_quadrupoles():
    lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(
        70, 70, padding=2, factor=0.0002, defects_x_jumps=4)

    lattice = lattice_gen.generate_lattice()
    fig: Figure = plt.figure(figsize=(7, 7))
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    print("Plotting bonds")
    lattice.plot_bonds(ax)
    print("After plotting bonds")

    ax.axis('off')
    box_size = 22
    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    # lattice.plot_dots(ax)
    print("Saving fig 2D ori")
    fig.savefig(os.path.join(FIGURE_PATH, "sphere-by-traceless-Q-config.png"))
    # plt.show()

    print("Relaxing the lattice in 2D")
    lattice.do_relaxation()
    fig: Figure = plt.figure(figsize=(7, 7))
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    lattice.plot_bonds(ax)
    ax.axis('off')
    box_size = 22
    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    print("Saving fig 2D relaxed")
    fig.savefig(os.path.join(FIGURE_PATH, "sphere-by-traceless-Q-relaxed-2D.png"))

    lattice_gen.set_z_to_sphere(radius=120)
    lattice = lattice_gen.generate_lattice()
    print("Relaxing the lattice in 3D")
    lattice.do_relaxation()
    fig: Figure = plt.figure(figsize=(7, 7))
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-75)
    lattice.plot_bonds(ax)
    ax.set_zlim(-5, 5)
    plotutils.set_3D_labels(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURE_PATH, "sphere-by-traceless-Q.png"))
    plt.show()


def plot_dots(lattice: Lattice, azim, elev):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=azim, elev=elev)
    lattice.plot_dots(ax)
    plotutils.set_3D_labels(ax)
    # plotutils.set_axis_scaled(ax)
    return fig, ax


def analyze_single_SW_displacement():
    nx, ny = 60, 60
    lattice = TriangularLatticeGenerator(nx, ny)

    x = nx // 2
    y = ny // 2
    print(f"Putting SW defect at {x},{y}")
    lattice.add_SW_defect(y, x)

    reference = lattice.dots.copy()

    snapshot = do_relaxation(lattice.frame, lattice.harmonic, lattice.dihedrals)

    relaxed = snapshot.particles.position

    print(np.mean(relaxed[:, 0]))

    if not np.all(np.isclose(relaxed[:, 2], 0)):
        print("The relaxed state is not in XY plane!!!")
    if not np.all(np.isclose(np.mean(relaxed[:, 0]), 0)):
        print("The relaxed state is not centered in X axis")
    if not np.all(np.isclose(np.mean(relaxed[:, 1]), 0)):
        print("The relaxed state is not centered in Y axis")

    fig, axes = plt.subplots(1, 2)
    dx = np.reshape(relaxed[:, 0] - reference[:, 0], (ny, nx))
    dy = np.reshape(relaxed[:, 1] - reference[:, 1], (ny, nx))
    plotutils.imshow_with_colorbar(fig, axes[0], dx, "dx")
    plotutils.imshow_with_colorbar(fig, axes[1], dy, "dy")

    dx_vs_x = dx[ny // 2, nx // 2 + 1:]
    dx_vs_y = dx[ny // 2 + 1:, nx // 2 + 1]
    dy_vs_x = dy[ny // 2, nx // 2 + 1:]
    dy_vs_y = dy[ny // 2 + 1:, nx // 2 + 1]
    fig, axes = plt.subplots(2, 2)
    axes = axes.flat
    axes[0].plot(1 + np.arange(len(dx_vs_x)), dx_vs_x, '.')
    axes[1].plot(dx_vs_y, '.')
    axes[2].plot(dy_vs_x, '.')
    axes[3].plot(dy_vs_y, '.')

    import origami.utils.fitter as fitter
    func = lambda x, a, b: a / x + b * x
    params = fitter.FitParams(func, 1 + np.arange(len(dx_vs_x)), dx_vs_x)
    fit = fitter.FuncFit(params)
    axes[0].set_xlim(0)
    fit.plot_fit(axes[0])
    fit.print_results()

    plt.show()


def plot_single_SW():
    nx, ny = 8, 8
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    x = nx // 2 - 1
    y = ny // 2
    print(f"Putting SW defect at {x},{y}")
    lattice_gen.add_SW_defect(y, x)
    lattice = lattice_gen.generate_lattice()

    fig: Figure = plt.figure(figsize=(4, 10))
    ax1: Axes3D = fig.add_subplot(311, projection='3d', azim=-90, elev=90)
    ax2: Axes3D = fig.add_subplot(312, projection='3d', azim=-90, elev=90)
    ax3: Axes3D = fig.add_subplot(313, projection='3d', azim=-24, elev=54)
    ax1.axis('off')
    ax2.axis('off')
    lattice.plot_bonds(ax1)
    lattice.plot_dots(ax1)

    snapshot = lattice.do_relaxation()

    relaxed = snapshot.particles.position

    if not np.all(np.isclose(relaxed[:, 2], 0)):
        print("The relaxed state is not in XY plane!!!")
    if not np.all(np.isclose(np.mean(relaxed[:, 0]), 0)):
        print("The relaxed state is not centered in X axis")
    if not np.all(np.isclose(np.mean(relaxed[:, 1]), 0)):
        print("The relaxed state is not centered in Y axis")

    lattice.plot_bonds(ax2)
    lattice.plot_dots(ax2)

    lattice_gen.set_z_to_sphere(5)
    lattice = lattice_gen.generate_lattice()
    snapshot = lattice.do_relaxation()
    lattice.plot_bonds(ax3)
    lattice.plot_dots(ax3)
    # plotutils.set_axis_scaled(ax)
    zlim = 0.5
    ax3.set_zlim(-zlim, zlim)
    # ax3.set(xticks=[], yticks=[], zticks=[])
    fig.tight_layout()
    fig.subplots_adjust(hspace=-0.2)

    fig.savefig(os.path.join(FIGURE_PATH, 'single-sw.png'))

    plt.show()


def main():
    # cone_by_traceless_quadrupoles()
    # sphere_by_traceless_quadrupoles()
    # plot_sphere_by_traceless_quadrupoles()
    # sphere_by_inclusions()
    analyze_single_SW_displacement()
    # plot_single_SW()


if __name__ == "__main__":
    main()
