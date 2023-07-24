import os.path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import hoomdlattice
import plotutils
from latticegenerator import calc_metric_curvature_triangular_lattice, TriangularLatticeGenerator
from simulations import create_lattice_for_sphere_by_traceless_quadrupoles, plot_dots, create_lattice_for_negative_K_SW

FIGURE_PATH = "../Figures/MD-simulations"


def plot_frames_from_trajectory(trajectory_file, output_folder):
    print("Converting frames to trajectory")
    frames = hoomdlattice.load_trajectory(trajectory_file)
    for frame in frames:
        fig, ax = plot_dots(frame, azim=-40, elev=12)
        ax.set_zlim(-1, 1)
        print(f"saving timestep={frame.timestep}")
        ax.set_title(f'$ E_s $={frame.harmonic_energy:.6f}, '
                     f'$ E_b $={frame.dihedrals_energy:.6f}, '
                     f'timestep={frame.timestep}\n'
                     f'total energy='
                     f'{frame.harmonic_energy + frame.dihedrals_energy:.6f}')
        fig.savefig(
            os.path.join(output_folder, f'ts={frame.timestep}.svg'))
        plt.clf()


def simulate_sphere_progress():
    folder = os.path.join(FIGURE_PATH, 'sphere-progress')
    trajectory_file = os.path.join(folder, f'trajectory.gsd')
    if os.path.exists(trajectory_file):
        plot_frames_from_trajectory(trajectory_file, os.path.join(FIGURE_PATH, 'sphere-progress'))
        return

    nx, ny = 48, 50
    lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(nx, ny, defects_x_jumps=6, factor=0.0001)
    # plot_flat(lattice_gen, os.path.join(folder, 'initial.svg'), 15)

    lattice_gen.set_dihedral_k(2)
    lattice_gen.set_z_to_noise()

    lattice = lattice_gen.generate_lattice()

    def pre_iteration_hook():
        return
        # fig, ax = plot_dots(lattice, azim=-26, elev=2)
        # ax.set_zlim(-2, 2)
        # plt.show()
        # print(lattice.sim.timestep)
        lattice.save_frame(
            os.path.join(FIGURE_PATH, 'sphere-progress', f'ts={lattice.sim.timestep}.gsd'))

    lattice.log_trajectory(trajectory_file, 1000)
    lattice.do_relaxation(dt=0.1, iteration_time=500, force_tol=1e-8, pre_iteration_hook=pre_iteration_hook)

    snapshot = lattice.sim.state.get_snapshot()
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
    plotutils.set_3D_labels(ax)

    fig, ax = plt.subplots()
    Ks, g11, g12, g22 = calc_metric_curvature_triangular_lattice(
        snapshot.particles.position, nx, ny
    )
    Ks[Ks < 0] = 0
    Ks[Ks > 0.001] = 0.0
    im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
    # im.set_clim(0, 0.01)

    plt.show()


def test_SW_line():
    folder = os.path.join(FIGURE_PATH, 'SW-line')

    nx, ny = 11, 11
    lattice_gen = TriangularLatticeGenerator(nx, ny)
    for i in range(0, nx - 1):
        lattice_gen.add_SW_defect(ny // 2, i)
    # lattice_gen.set_z_to_noise()

    lattice_gen.set_dihedral_k(30.0)
    lattice = lattice_gen.generate_lattice()
    lattice.log_trajectory(os.path.join(FIGURE_PATH, 'SW-line', 'high-b-trajectory.gsd'), 200)
    plot_frames_from_trajectory(os.path.join(folder, 'high-b-trajectory.gsd'), folder)
    lattice.do_relaxation()
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_bonds(ax)
    plotutils.set_3D_labels(ax)
    # plotutils.set_axis_scaled(ax)
    fig.savefig(os.path.join(FIGURE_PATH, 'SW-line', 'large-bending.svg'))

    plt.show()

    lattice_gen.set_z_to_sphere(radius=30)

    lattice_gen.set_dihedral_k(0.05)
    lattice = lattice_gen.generate_lattice()
    lattice.do_relaxation()
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=7, elev=21)
    lattice.plot_bonds(ax)
    plotutils.set_3D_labels(ax)
    # plotutils.set_axis_scaled(ax)
    ax.set_aspect('equal')
    fig.savefig(os.path.join(FIGURE_PATH, 'SW-line', 'low-bending.svg'))

    lattice_gen.set_dihedral_k(2.0)
    lattice = lattice_gen.generate_lattice()
    lattice.do_relaxation(force_tol=1e-9)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=33, elev=18)
    lattice.plot_bonds(ax)
    plotutils.set_3D_labels(ax)
    # plotutils.set_axis_scaled(ax)
    ax.set_aspect('equal')
    fig.savefig(os.path.join(FIGURE_PATH, 'SW-line', 'medium-bending.svg'))

    plt.show()


def plot_negative_curvature():
    folder = os.path.join(FIGURE_PATH, 'negative-k')
    trajectory_file = os.path.join(folder, f'trajectory.gsd')
    if os.path.exists(trajectory_file):
        plot_frames_from_trajectory(trajectory_file, folder)
        return

    nx, ny = 60, 70
    lattice_gen = create_lattice_for_negative_K_SW(
        nx, ny, defects_y_jumps=6, factor=0.0001)

    plot_flat(lattice_gen, os.path.join(folder, 'initial.svg'), 20)

    lattice_gen.set_z_to_noise()

    lattice = lattice_gen.generate_lattice()

    lattice.log_trajectory(trajectory_file, 1000)
    lattice.do_relaxation(dt=0.1, iteration_time=500, force_tol=1e-8)


def plot_flat(lattice_gen, filename, box_size):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    lattice = lattice_gen.generate_lattice()
    lattice.plot_bonds(ax)
    # lattice.plot_dots(ax)
    ax.set_xlim(-box_size, box_size)
    ax.set_ylim(-box_size, box_size)
    fig.savefig(filename)


def main():
    # simulate_sphere_progress()
    # test_SW_line()
    plot_negative_curvature()


if __name__ == '__main__':
    main()
