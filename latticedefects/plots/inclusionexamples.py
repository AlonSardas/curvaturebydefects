import os

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import plots
from latticedefects.geometry import calculate_curvatures_by_interpolation
from latticedefects.inclusiondesign import create_sphere_by_inclusion, create_cone_by_inclusion
from latticedefects.plots.latticeplotutils import plot_flat_and_save
from latticedefects.utils import plotutils

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations")


def sphere_by_inclusions():
    folder = os.path.join(FIGURE_PATH, 'inclusions-constant-k')

    nx, ny = 40, 50

    factor = 0.00007
    # factor = 0.0
    disk_width = 5
    padding = 0
    C0 = -0.0

    # TODO: fix the plot for this distribution, I think I didn't treat the radial
    # TODO: dependency correctly
    lattice_gen = create_sphere_by_inclusion(nx, ny, disk_width, factor, C0, padding)
    lattice_gen.set_dihedral_k(3.0)
    plot_flat_and_save(lattice_gen, os.path.join(folder, 'sphere-inclusions-initial'), 15, plot='dots')

    def simulate_for_inclusion_d(inclusion_d, name):
        lattice_gen.set_inclusion_d(inclusion_d)
        lattice_gen.set_z_to_sphere(1000)
        lattice = lattice_gen.generate_lattice()
        lattice.log_trajectory(os.path.join(folder, name + '.gsd'), 500)
        lattice.do_relaxation(iteration_time=1000)
        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d")
        lattice.plot_dots(ax)
        plotutils.set_axis_scaled(ax)
        ax.set_zlim(-5, 5)
        fig.savefig(os.path.join(folder, name + '-final.svg'))
        fig.savefig(os.path.join(folder, name + '-final.png'))

        fig, ax = plt.subplots()
        Ks, Hs = calculate_curvatures_by_interpolation(lattice.get_dots())
        plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    simulate_for_inclusion_d(1.2, 'growth')
    simulate_for_inclusion_d(0.8, 'shrink')
    plt.show()


def cone_by_inclusions():
    folder = os.path.join(FIGURE_PATH, 'inclusions-cone')

    nx, ny = 43, 50

    factor = 0.02
    disk_width = 4
    padding = 0
    r0 = 0.5

    lattice_gen = create_cone_by_inclusion(nx, ny, disk_width, factor, r0, padding)
    lattice_gen.set_inclusion_d(0.7)
    lattice_gen.set_dihedral_k(3.0)
    plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'), 15, plot='dots', with_axes=False)

    # lattice_gen.set_inclusion_d(inclusion_d)
    lattice_gen.set_z_to_sphere(1000)
    lattice = lattice_gen.generate_lattice()
    lattice.log_trajectory(os.path.join(folder, 'cone.gsd'), 500)
    lattice.do_relaxation(iteration_time=1000)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-120, elev=23)
    lattice.plot_dots(ax)
    plotutils.set_axis_scaled(ax)
    ax.set_zlim(-3, 3)
    fig.savefig(os.path.join(folder, 'cone-final.png'))
    fig.savefig(os.path.join(folder, 'cone-final.pdf'))

    fig, ax = plt.subplots()
    Ks, Hs = calculate_curvatures_by_interpolation(lattice.get_dots())
    plotutils.imshow_with_colorbar(fig, ax, Ks, "K")

    plt.show()


def main():
    # sphere_by_inclusions()
    cone_by_inclusions()


if __name__ == '__main__':
    main()
