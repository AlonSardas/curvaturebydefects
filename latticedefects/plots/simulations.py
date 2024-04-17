import os.path

import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import plots
from latticedefects.geometry import calc_metric_curvature_triangular_lattice
from latticedefects.hoomdlattice import do_relaxation
from latticedefects.latticegenerator import TriangularLatticeGenerator
from latticedefects.plots.latticeplotutils import plot_flat_and_save
from latticedefects.swdesign import create_lattice_for_sphere_by_traceless_quadrupoles, create_cone_by_sw
from latticedefects.trajectory import load_trajectory
from latticedefects.utils import plotutils

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations")


def cone_by_traceless_quadrupoles():
    folder = os.path.join(FIGURE_PATH, 'cone-by-SW')
    traj_path = os.path.join(folder, 'trajectory.gsd')

    should_simulate = True
    if should_simulate:
        nx, ny = 54, 55
        defects_jumps = 4
        padding = 0
        lattice_gen = create_cone_by_sw(nx, ny, defects_jumps, padding)

        lattice_gen.set_dihedral_k(2.0)
        print(f"k={lattice_gen.get_spring_constant()}, kappa={lattice_gen.get_dihedral_k()}")

        plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'), 20, with_axes=False)

        lattice_gen.set_z_to_sphere(radius=1000)

        lattice = lattice_gen.generate_lattice()
        lattice.log_trajectory(traj_path, 200)
        snapshot = lattice.do_relaxation(force_tol=1e-7)

    frame = load_trajectory(traj_path)[-1]
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    # lattice.plot_dots(ax)
    frame.plot_bonds(ax)
    ax.set_zlim(-5, 5)
    fig.savefig(os.path.join(folder, 'sw-cone.pdf'))
    fig.savefig(os.path.join(folder, 'sw-cone.png'))
    plt.show()

    should_calc_curvature = False
    if should_calc_curvature:
        fig, ax = plt.subplots()
        Ks, g11, g12, g22 = calc_metric_curvature_triangular_lattice(
            snapshot.particles.position, nx, ny
        )
        plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    plt.show()


def plot_sphere_by_traceless_quadrupoles():
    folder = os.path.join(FIGURE_PATH, 'sphere-by-SW')
    lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(
        70, 70, padding=2, factor=0.0002, defects_x_jumps=4)

    lattice = lattice_gen.generate_lattice()
    fig: Figure = plt.figure(figsize=(7, 14))
    ax: Axes3D = fig.add_subplot(211, projection='3d', azim=-90, elev=90)
    print("Plotting bonds")
    lattice.plot_bonds(ax)
    print("After plotting bonds")

    ax.axis('off')
    box_size = 22
    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)

    should_relax_2D = False
    if should_relax_2D:
        print("Relaxing the lattice in 2D")
        lattice.do_relaxation()
        fig2D: Figure = plt.figure(figsize=(7, 7))
        ax: Axes3D = fig2D.add_subplot(111, projection='3d', azim=-90, elev=90)
        lattice.plot_bonds(ax)
        ax.axis('off')
        box_size = 22
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size, box_size)
        print("Saving fig 2D relaxed")
        fig2D.savefig(os.path.join(folder, "sphere-by-traceless-Q-relaxed-2D.svg"))

    lattice_gen.set_z_to_sphere(radius=120)
    lattice = lattice_gen.generate_lattice()
    print("Relaxing the lattice in 3D")
    lattice.log_trajectory(os.path.join(folder, 'trajectory.gsd'), 1000)
    lattice.do_relaxation(iteration_time=500)
    ax: Axes3D = fig.add_subplot(212, projection='3d', azim=-75)
    print("Plotting bonds relaxed")
    lattice.plot_bonds(ax)
    ax.set_zlim(-5, 5)
    plotutils.set_3D_labels(ax)
    fig.tight_layout()
    print("Saving fig")
    fig.savefig(os.path.join(folder, "sphere-by-SW.png"))
    plt.show()


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
    def func(x, a, b):
        return a / x + b * x

    params = fitter.FitParams(func, 1 + np.arange(len(dx_vs_x)), dx_vs_x)
    fit = fitter.FuncFit(params)
    axes[0].set_xlim(0)
    fit.plot_fit(axes[0])
    fit.print_results()

    plt.show()


def plot_single_SW():
    nx, ny = 8, 8
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=1.0)

    x = nx // 2 - 1
    y = ny // 2
    print(f"Putting SW defect at {x},{y}")
    lattice_gen.add_SW_defect(y, x)
    lattice = lattice_gen.generate_lattice()

    fig: Figure = plt.figure(figsize=(4, 10.6))
    ax1: Axes3D = fig.add_subplot(311, projection='3d', azim=-90, elev=90)
    ax2: Axes3D = fig.add_subplot(312, projection='3d', azim=-90, elev=90)
    ax3: Axes3D = fig.add_subplot(313, projection='3d', azim=123, elev=50)
    # ax1.axis('off')
    # ax2.axis('off')
    lattice.plot_bonds(ax1)
    lattice.plot_dots(ax1)

    snapshot = lattice.do_relaxation(force_tol=1e-8)

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
    fig.subplots_adjust(hspace=-0.3, top=1.05, bottom=0, left=-0.2, right=1)
    print(ax1.get_position())
    print(ax2.get_position())
    ax1.set_position([0, 0.66, 1, 0.4])

    mpl.rcParams["savefig.bbox"] = "standard"
    fig.savefig(os.path.join(FIGURE_PATH, 'single-sw.png'))

    plt.show()


def plot_single_SW2():
    folder = os.path.join(FIGURE_PATH, 'single-SW')
    initial_path = os.path.join(folder, 'initial.gsd')
    relaxed_2D_path = os.path.join(folder, '2D-relaxed.gsd')
    relaxed_3D_path = os.path.join(folder, '3D-relaxed.gsd')

    file_format = 'pdf'

    fig, ax = plot_frame(initial_path, 90, -90)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(folder, f'initial.{file_format}'),
                               0.8, 0.8, translate_y=0.1)

    fig, ax = plot_frame(relaxed_2D_path, 90, -90)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(folder, f'relaxed-2D.{file_format}'),
                               0.8, 0.8, translate_y=0.1)

    fig, ax = plot_frame(relaxed_3D_path, 35, -140)
    zlim = 0.5
    ax.set_zlim(-zlim, zlim)
    plotutils.set_zoom_by_limits(ax, 1.2)
    plotutils.save_fig_cropped(fig, os.path.join(folder, f'relaxed-3D.{file_format}'),
                               1.35, 1.1, translate_y=-0.6, translate_x=-0.3)
    # plt.show()

    return

    nx, ny = 8, 8
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=1.0)

    x = nx // 2 - 1
    y = ny // 2
    print(f"Putting SW defect at {x},{y}")
    lattice_gen.add_SW_defect(y, x)
    lattice = lattice_gen.generate_lattice()

    lattice.save_frame(initial_path)

    snapshot = lattice.do_relaxation(force_tol=1e-8)
    lattice.save_frame(relaxed_2D_path)

    relaxed = snapshot.particles.position
    if not np.all(np.isclose(relaxed[:, 2], 0)):
        print("The relaxed state is not in XY plane!!!")
    if not np.all(np.isclose(np.mean(relaxed[:, 0]), 0)):
        print("The relaxed state is not centered in X axis")
    if not np.all(np.isclose(np.mean(relaxed[:, 1]), 0)):
        print("The relaxed state is not centered in Y axis")

    lattice_gen.set_z_to_sphere(5)
    lattice = lattice_gen.generate_lattice()
    lattice.do_relaxation()
    lattice.save_frame(relaxed_3D_path)


def plot_single_inclusion2():
    folder = os.path.join(FIGURE_PATH, 'single-inclusion')
    initial_path = os.path.join(folder, 'initial.gsd')
    relaxed_2D_path = os.path.join(folder, '2D-relaxed.gsd')
    relaxed_3D_path = os.path.join(folder, '3D-relaxed.gsd')

    should_simulate = True
    if should_simulate:
        nx, ny = 7, 7
        lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=2.5, inclusion_d=1.2)

        x = nx // 2
        y = ny // 2
        print(f"Putting inclusion defect at {x},{y}")
        lattice_gen.add_inclusion_defect(y, x)
        lattice = lattice_gen.generate_lattice()
        lattice.save_frame(initial_path)

        snapshot = lattice.do_relaxation(force_tol=1e-8)
        lattice.save_frame(relaxed_2D_path)

        _verify_flatness(snapshot.particles.position)

        lattice_gen.set_z_to_sphere(5)
        lattice = lattice_gen.generate_lattice()
        lattice.do_relaxation()
        lattice.save_frame(relaxed_3D_path)

    file_format = 'pdf'

    fig, ax = plot_frame(initial_path, 90, -90)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(folder, f'initial.{file_format}'),
                               0.8, 0.8, translate_y=0.1)

    fig, ax = plot_frame(relaxed_2D_path, 90, -90)
    ax.set_axis_off()
    plotutils.save_fig_cropped(fig, os.path.join(folder, f'relaxed-2D.{file_format}'),
                               0.8, 0.8, translate_y=0.1)

    fig, ax = plot_frame(relaxed_3D_path, 35, -140)
    zlim = 0.5
    ax.set_zlim(-zlim, zlim)
    plotutils.set_zoom_by_limits(ax, 1.2)
    plotutils.save_fig_cropped(fig, os.path.join(folder, f'relaxed-3D.{file_format}'),
                               1.35, 1.1, translate_y=-0.6, translate_x=-0.3)
    # plt.show()

    return

    nx, ny = 8, 8
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=1.0)

    x = nx // 2 - 1
    y = ny // 2
    print(f"Putting SW defect at {x},{y}")
    lattice_gen.add_SW_defect(y, x)
    lattice = lattice_gen.generate_lattice()

    lattice.save_frame(initial_path)

    snapshot = lattice.do_relaxation(force_tol=1e-8)
    lattice.save_frame(relaxed_2D_path)

    _verify_flatness(snapshot.particles.position)

    lattice_gen.set_z_to_sphere(5)
    lattice = lattice_gen.generate_lattice()
    lattice.do_relaxation()
    lattice.save_frame(relaxed_3D_path)


def plot_frame(path, elev, azim):
    fig: Figure = plt.figure(layout='constrained')
    ax: Axes3D = fig.add_subplot(111, projection='3d', elev=elev, azim=azim)
    initial_config = load_trajectory(path)[0]
    initial_config.plot_bonds(ax)
    initial_config.plot_dots(ax)
    return fig, ax


def _verify_flatness(dots):
    if not np.all(np.isclose(dots[:, 2], 0)):
        print("The relaxed state is not in XY plane!!!")
    if not np.all(np.isclose(np.mean(dots[:, 0]), 0)):
        print("The relaxed state is not centered in X axis")
    if not np.all(np.isclose(np.mean(dots[:, 1]), 0)):
        print("The relaxed state is not centered in Y axis")


def plot_single_inclusion():
    nx, ny = 7, 7
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=2.5, inclusion_d=1.2)

    x = nx // 2
    y = ny // 2
    print(f"Putting inclusion defect at {x},{y}")
    lattice_gen.add_inclusion_defect(y, x)
    lattice = lattice_gen.generate_lattice()

    fig: Figure = plt.figure(figsize=(4, 10))
    ax1: Axes3D = fig.add_subplot(311, projection='3d', azim=-90, elev=90)
    ax2: Axes3D = fig.add_subplot(312, projection='3d', azim=-90, elev=90)
    ax3: Axes3D = fig.add_subplot(313, projection='3d', azim=-140, elev=35)
    ax1.axis('off')
    ax2.axis('off')
    lattice.plot_bonds(ax1)
    lattice.plot_dots(ax1)

    snapshot = lattice.do_relaxation(force_tol=1e-8)

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
    lattice.do_relaxation()
    lattice.plot_bonds(ax3)
    lattice.plot_dots(ax3)
    # plotutils.set_axis_scaled(ax)
    zlim = 0.5
    ax3.set_zlim(-zlim, zlim)
    # ax3.set(xticks=[], yticks=[], zticks=[])
    fig.tight_layout()
    fig.subplots_adjust(hspace=-0.2)

    fig.savefig(os.path.join(FIGURE_PATH, 'single-inclusion.png'))

    plt.show()


def main():
    # cone_by_traceless_quadrupoles()
    # sphere_by_traceless_quadrupoles()
    # plot_sphere_by_traceless_quadrupoles()
    # sphere_by_inclusions()
    # analyze_single_SW_displacement()
    # plot_single_inclusion()
    # plot_single_SW2()
    plot_single_inclusion2()


if __name__ == "__main__":
    main()
