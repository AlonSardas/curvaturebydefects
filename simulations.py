import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import hoomdlattice
import plotutils
from hoomdlattice import do_relaxation
from latticegenerator import (
    TriangularLatticeGenerator,
    calc_metric_curvature_triangular_lattice,
)


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
    nx, ny = 60, 60
    lattice = TriangularLatticeGenerator(nx, ny)
    # lattice.set_z_to_noise()
    lattice.set_z_to_sphere(radius=100)

    defects_jumps = 3
    padding = 0
    for x in range(nx // 2, nx - padding, defects_jumps):
        for y in range(padding, ny - padding, defects_jumps):
            # if (x - nx // 2) >= abs(y - ny // 2):
            if (x - nx // 2) >= abs(y - ny // 2):
                print(f"Putting SW at {x},{y}")
                try:
                    lattice.add_SW_defect(y, x)
                except RuntimeError as e:
                    print("warning: ", e.args)

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


def sphere_by_traceless_quadrupoles():
    nx, ny = 70, 80
    lattice = TriangularLatticeGenerator(nx, ny)
    # lattice.set_z_to_noise()
    lattice.set_z_to_sphere(radius=120)

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

    snapshot = do_relaxation(lattice.frame, lattice.harmonic, lattice.dihedrals)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    hoomdlattice.plot_dots(ax, snapshot)
    dots = snapshot.particles.position
    dots_pos = dots[dots[:, 2] > 0]
    dots = dots_pos
    ax.plot(dots[:, 0], dots[:, 1], dots[:, 2], ".")

    # hoomdlattice.plot_bonds(ax, snapshot)
    plotutils.set_axis_scaled(ax)
    ax.set_zlim(-5, 5)

    fig, ax = plt.subplots()
    Ks, g11, g12, g22 = calc_metric_curvature_triangular_lattice(
        snapshot.particles.position, nx, ny
    )
    Ks[Ks < 0] = 0
    Ks[Ks > 0.001] = 0.0
    im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
    # im.set_clim(0, 0.01)

    plt.show()


def analyze_single_SW():
    nx, ny = 60, 60
    lattice = TriangularLatticeGenerator(nx, ny)

    x = nx // 2
    y = ny // 2
    print(f"Putting SW defect at {x},{y}")
    lattice.add_SW_defect(y, x)

    reference = lattice.dots.copy()

    snapshot = do_relaxation(lattice.frame, lattice.harmonic, lattice.dihedrals)

    relaxed = snapshot.particles.position

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
    axes[0].plot(1+np.arange(len(dx_vs_x)), dx_vs_x, '.')
    axes[1].plot(dx_vs_y, '.')
    axes[2].plot(dy_vs_x, '.')
    axes[3].plot(dy_vs_y, '.')

    import origami.utils.fitter as fitter
    func = lambda x, a, b: a / x+b*x
    params = fitter.FitParams(func, 1+np.arange(len(dx_vs_x)), dx_vs_x)
    fit = fitter.FuncFit(params)
    axes[0].set_xlim(0)
    fit.plot_fit(axes[0])
    fit.print_results()

    plt.show()


def main():
    # cone_by_traceless_quadrupoles()
    # sphere_by_traceless_quadrupoles()
    # sphere_by_inclusions()
    analyze_single_SW()


if __name__ == "__main__":
    main()
