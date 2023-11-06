import os.path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from latticedefects import hoomdlattice, plots, trajectory
from latticedefects.geometry import calc_metric_curvature_triangular_lattice
from latticedefects.latticegenerator import TriangularLatticeGenerator
from latticedefects.plots.latticeplotutils import create_fig_and_plot_dots, plot_flat_and_save
from latticedefects.swdesign import create_lattice_for_sphere_by_traceless_quadrupoles, \
    create_lattice_for_negative_K_SW, create_cone_by_sw_symmetric
from latticedefects.trajectory import load_trajectory, plot_frames_from_trajectory
from latticedefects.utils import plotutils

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations")


def plot_sphere():
    folder = os.path.join(FIGURE_PATH, 'sphere-by-SW')

    initial_frame_path = os.path.join(folder, 'initial.gsd')

    should_calc = False
    if should_calc:
        lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(
            70, 70, padding=2, factor=0.0002, defects_x_jumps=4)

        initial = lattice_gen.generate_lattice()
        initial.save_frame(initial_frame_path)

        Rs = [250, 130, 50]
        for R in Rs:
            lattice_gen.set_z_to_sphere(radius=R)
            lattice = lattice_gen.generate_lattice()
            print("Relaxing the lattice in 3D")
            lattice.log_trajectory(os.path.join(folder, f'traj-R={R}.gsd'), 1000)
            lattice.do_relaxation(iteration_time=1000)

    should_plot = True
    if should_plot:
        first_frame = load_trajectory(initial_frame_path)[0]
        traj_path = os.path.join(folder, 'traj-R=250.gsd')
        traj = load_trajectory(traj_path)

        last_frame = traj[-1]

        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
        print("Plotting bonds")
        first_frame.plot_bonds(ax)
        print("After plotting bonds")

        ax.axis('off')
        box_size = 22
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size, box_size)

        fig.savefig(os.path.join(folder, "2D-lattice.pdf"))

        fig: Figure = plt.figure()

        ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-75)
        print("Plotting bonds relaxed")
        last_frame.plot_bonds(ax)
        ax.set_zlim(-5, 5)
        # plotutils.set_3D_labels(ax)
        fig.tight_layout()
        print("Saving fig")
        fig.savefig(os.path.join(folder, "sphere-by-SW.pdf"))
        # plt.show()


def plot_sphere_random_initial_configuration():
    folder = os.path.join(FIGURE_PATH, 'sphere-by-SW')
    lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(
        70, 70, padding=2, factor=0.0002, defects_x_jumps=4)

    print(lattice_gen.get_spring_constant(), lattice_gen.get_dihedral_k())

    curvature_map = np.array([[-1, 0], [-1, -2]])
    lattice_gen.set_z_by_curvature(curvature_map, 0.00005)
    # lattice_gen.set_z_to_noise(5)
    lattice = lattice_gen.generate_lattice()
    print("Relaxing the lattice in 3D")
    lattice.log_trajectory(os.path.join(folder, f'traj-noise.gsd'), 1000)
    lattice.do_relaxation(iteration_time=1000)


def plot_bad_boundary():
    folder = os.path.join(FIGURE_PATH, 'sphere-by-SW')

    initial_frame_path = os.path.join(folder, 'initial-bad-boundary.gsd')

    should_simulate = False
    if should_simulate:
        lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(
            49, 50, padding=3, factor=0.0005, defects_x_jumps=3)

        print(lattice_gen.get_spring_constant(), lattice_gen.get_dihedral_k())
        initial = lattice_gen.generate_lattice()
        initial.save_frame(initial_frame_path)

        lattice_gen.set_z_to_sphere(200)
        lattice = lattice_gen.generate_lattice()
        print("Relaxing the lattice in 3D")
        lattice.log_trajectory(os.path.join(folder, f'traj-bad-boundary.gsd'), 1000)
        lattice.do_relaxation(iteration_time=1000)

    should_plot = True
    if should_plot:
        first_frame = load_trajectory(initial_frame_path)[0]
        traj_path = os.path.join(folder, 'traj-bad-boundary.gsd')
        traj = load_trajectory(traj_path)
        last_frame = traj[-1]

        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
        print("Plotting bonds")
        first_frame.plot_bonds(ax)
        print("After plotting bonds")

        ax.axis('off')
        box_size = 16
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size, box_size)

        fig.savefig(os.path.join(folder, "2D-lattice-bad-boundary.pdf"))
        return

        fig: Figure = plt.figure()

        ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-75)
        print("Plotting bonds relaxed")
        last_frame.plot_bonds(ax)
        ax.set_zlim(-5, 5)
        fig.tight_layout()
        print("Saving fig")
        fig.savefig(os.path.join(folder, "bad-boundary-relaxed.pdf"))


def sphere_increasing_bending():
    folder = os.path.join(FIGURE_PATH, 'sphere-increasing-bending')
    initial_config_path = os.path.join(folder, 'initial-config.gsd')
    bs = [0.5, 0.8, 1, 2, 8, 14]

    should_simulate = False
    if should_simulate:
        nx, ny = 46, 50
        lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(nx, ny, defects_x_jumps=5, factor=0.00015)
        plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'), 15, with_axes=False)

        # lattice_gen.set_z_to_sphere()
        # set z to bumps
        lattice_gen.dots[:, 2] = -np.cos(2*np.pi*lattice_gen.dots[:, 0] / 20) \
            - np.cos(2*np.pi*lattice_gen.dots[:, 1] / 20)
        lattice = lattice_gen.generate_lattice()
        lattice.save_frame(initial_config_path)

        for b in bs:
            lattice_gen.set_dihedral_k(b)
            print(lattice_gen.get_spring_constant(), lattice_gen.get_dihedral_k())
            lattice = lattice_gen.generate_lattice()
            lattice.do_relaxation()
            lattice.save_frame(os.path.join(folder, f"b={b:.2f}.gsd"))

    def plot(lat):
        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-40, elev=12)
        lat.plot_dots(ax)
        ax.set_zlim(-2, 2)
        return fig, ax

    for b in bs:
        gsd_file = os.path.join(folder, f"b={b:.2f}.gsd")
        img_file1 = os.path.join(folder, f"b={b:.2f}.png")
        img_file2 = os.path.join(folder, f"b={b:.2f}.pdf")
        if os.path.exists(gsd_file):
            print(f"plotting b={b}")
            frame = trajectory.load_trajectory(gsd_file)[0]
            fig, ax = plot(frame)
            ax.set_title(r"$ \tilde{\kappa}=" + f"{b:.2f} $", y=1.0, pad=-20)
            # fig.subplots_adjust(top=0.9)
            # fig.suptitle(r"$ \tilde{\kappa}=" + f"{b:.2f} $")
            fig.savefig(img_file1)
            fig.savefig(img_file2)

    if os.path.exists(initial_config_path):
        frame = trajectory.load_trajectory(initial_config_path)[0]
        fig, ax = plot(frame)
        fig.savefig(os.path.join(folder, 'initial-config.pdf'))
        fig.savefig(os.path.join(folder, 'initial-config.png'))


def plot_SW_line():
    folder = os.path.join(FIGURE_PATH, 'SW-line')

    should_simulate = False
    if should_simulate:
        nx, ny = 11, 11
        lattice_gen = TriangularLatticeGenerator(nx, ny)
        for i in range(0, nx - 1):
            lattice_gen.add_SW_defect(ny // 2, i)
        # lattice_gen.set_z_to_noise()

        lattice_gen.set_dihedral_k(30.0)
        lattice = lattice_gen.generate_lattice()
        lattice.log_trajectory(os.path.join(folder, 'high-b-trajectory.gsd'), 200)
        
        lattice.do_relaxation()
        
        lattice_gen.set_z_to_sphere(radius=30)

        lattice_gen.set_dihedral_k(0.05)
        lattice = lattice_gen.generate_lattice()
        lattice.log_trajectory(os.path.join(folder, 'low-rigidity.gsd'), 200)
        lattice.do_relaxation(force_tol=1e-9)
        
        lattice_gen.set_dihedral_k(1.5)
        lattice = lattice_gen.generate_lattice()
        lattice.log_trajectory(os.path.join(folder, 'medium-rigidity.gsd'), 100)
        lattice.do_relaxation(force_tol=1e-9)

    # plot_frames_from_trajectory(os.path.join(folder, 'high-b-trajectory.gsd'), folder)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=7, elev=21)
    frame = load_trajectory(os.path.join(folder, 'low-rigidity.gsd'))[-1]
    frame.plot_bonds(ax)
    ax.set_aspect('equal')
    ax.tick_params('z', pad=10)
    ax.set_xlabel('X', labelpad=15)
    ax.set_ylabel('Y', labelpad=15)
    ax.set_zlabel('Z', labelpad=15)
    fig.savefig(os.path.join(FIGURE_PATH, 'SW-line', 'low-rigidity.pdf'), pad_inches=0.2)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=33, elev=18)
    frame = load_trajectory(os.path.join(folder, 'medium-rigidity.gsd'))[-1]
    frame.plot_bonds(ax)
    ax.set_aspect('equal')
    ax.set_zticks([-0.6, 0, 0.6],)
    ax.set_xlabel('X', labelpad=25)
    ax.set_ylabel('Y', labelpad=25)
    ax.set_zlabel('Z', labelpad=5)
    fig.savefig(os.path.join(FIGURE_PATH, 'SW-line', 'medium-rigidity.pdf'), pad_inches=0.3)

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

    plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'), 20)

    lattice_gen.set_z_to_noise()

    lattice = lattice_gen.generate_lattice()

    lattice.log_trajectory(trajectory_file, 1000)
    lattice.do_relaxation(dt=0.1, iteration_time=500, force_tol=1e-8)


def sphere_small_lattice():
    folder = os.path.join(FIGURE_PATH, 'sphere-small-lattice')
    trajectory_file = os.path.join(folder, f'trajectory.gsd')

    # nx, ny = 20, 30
    nx, ny = 50, 60
    dihedral_k = 1.2

    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=dihedral_k)
    # lattice_gen.add_SW_defect(15, 9)
    # lattice_gen.add_SW_defect(15, 10)
    lattice_gen.add_SW_defect(30, 24)
    lattice_gen.set_z_to_sphere(radius=1000)
    lattice = lattice_gen.generate_lattice()
    lattice.log_trajectory(os.path.join(folder, 'trajectory-single-SW.gsd'), 100)
    lattice.do_relaxation()

    lattice_gen = create_lattice_for_sphere_by_traceless_quadrupoles(
        nx, ny, padding=1, factor=0.0004, defects_x_jumps=4)
    lattice_gen.set_dihedral_k(dihedral_k)

    plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'))

    lattice_gen.set_z_to_sphere(radius=1000)
    lattice = lattice_gen.generate_lattice()
    lattice.log_trajectory(trajectory_file, 100)
    lattice.do_relaxation()


def cone_by_traceless_quadrupoles_symmetric():
    folder = os.path.join(FIGURE_PATH, 'cone-by-SW2')

    nx, ny = 50, 60
    defects_jumps = 4
    padding = 1
    x_shift = -3
    lattice_gen = create_cone_by_sw_symmetric(nx, ny, defects_jumps, padding, x_shift)
    # lattice_gen.set_sw_bonds(r0=0.3)

    lattice_gen.set_dihedral_k(2.0)

    plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'))
    print("flat saved")

    lattice_gen.set_z_to_sphere(radius=1000)

    lattice = lattice_gen.generate_lattice()
    lattice.log_trajectory(os.path.join(folder, 'trajectory.gsd'), 500)
    lattice.do_relaxation(force_tol=1e-7, iteration_time=500)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_dots(ax)
    plotutils.set_3D_labels(ax)
    # lattice.plot_bonds(ax)
    ax.set_zlim(-5, 5)

    plt.show()


def main():
    # simulate_sphere_progress()
    # plot_SW_line()
    # plot_negative_curvature()
    # sphere_increasing_bending()
    # sphere_small_lattice()
    # cone_by_traceless_quadrupoles_symmetric()
    # plot_sphere()
    # plot_sphere_random_initial_configuration()
    plot_bad_boundary()


if __name__ == '__main__':
    main()
