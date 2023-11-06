import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import plots, geometry, inclusiondesign
from latticedefects.inclusiondesign import get_distribution_by_curvature
from latticedefects.utils import plotutils
from latticedefects.utils.plotutils import imshow_with_colorbar

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations", 'inclusionstorus')


def create_torus():
    folder = FIGURE_PATH
    traj_name = 'traj'

    nx, ny = 150, 170
    samples_x, samples_y = 80, 80

    r0 = 40.0
    Z0 = 0.2

    xs = np.arange(nx) - (nx - 1) / 2
    ys = (np.arange(ny) - (ny - 1) / 2) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt(xs ** 2 + ys ** 2)

    zs = Z0 * (rs / r0) ** 2 * np.exp(-(rs / r0) ** 2)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.plot(xs.flat,
            ys.flat,
            zs.flat, ".", color='C0', alpha=0.8)

    dots = np.array([xs.flat, ys.flat, zs.flat]).transpose()
    print("Calculating the curvature")
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(
        dots, samples_x, samples_y)

    print("Plotting the desired curvature")
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, "desired Ks")

    print("Calculating dist")
    dist = get_distribution_by_curvature(Ks)
    dist -= dist.min()

    print("plot dist")
    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    lattice_nx = 120
    # For some reason, smaller factor makes the simulations reach faster to equilibrium
    reduce_factor = 0.15
    # reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = inclusiondesign.create_defects_map_by_dist(
        dist, lattice_nx, reduce_factor, padding=(0, 1, 2, 2))
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects.png'))

    plt.show()

    print("Creating lattice")
    lattice_gen = inclusiondesign.create_lattice_by_defects_map(defects_map)
    lattice_gen.set_inclusion_d(0.8)
    lattice_gen.set_dihedral_k(4)
    lattice_gen.set_spring_constant(20)
    print("finished creating lattice")

    print(f"expected length: {lattice_gen.get_dihedral_k() / lattice_gen.get_spring_constant()}")

    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    r0 = r0 / nx * lattice_nx
    Z0 = 25.0
    lattice_gen.dots[:, 2] = Z0 * (rs / r0) ** 2 * np.exp(-(rs / r0) ** 2)
    lattice_gen.dots[:, 0] *= 1.2
    lattice_gen.dots[:, 1] *= 1.2

    with_hole = True
    if with_hole:
        lattice_gen.remove_dots_inside(5.0)
        traj_name = traj_name + '-hole'
    traj_name = traj_name + '.gsd'

    # lattice_gen.remove_dots_outside(35)

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)

    lattice.log_trajectory(os.path.join(folder, traj_name), 5000)
    lattice.do_relaxation(dt=0.1, iteration_time=5000)


def test_cone():
    folder = FIGURE_PATH
    traj_name = 'cone-test'

    Ks = np.zeros((150, 150))
    Ks[50, 75] = 1

    print("Plotting the desired curvature")
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, "desired Ks")

    print("Calculating dist")
    dist = get_distribution_by_curvature(Ks)
    dist -= dist.min()
    dist -= dist.mean() / 2
    dist[dist < 0] = 0

    print("plot dist")
    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    lattice_nx = 70
    reduce_factor = 0.05
    print("Calculating defects map")
    defects_map, _ = inclusiondesign.create_defects_map_by_dist(
        dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')

    plt.show()

    print("Creating lattice")
    lattice_gen = inclusiondesign.create_lattice_by_defects_map(defects_map)
    # lattice_gen.set_inclusion_d(0.7)
    lattice_gen.set_dihedral_k(27)
    lattice_gen.set_spring_constant(1)
    print("finished creating lattice")

    print(f"expected length: {lattice_gen.get_dihedral_k() / lattice_gen.get_spring_constant()}")
    lattice_gen.set_z_to_sphere(100000)

    traj_name = traj_name + '.gsd'

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)

    lattice.log_trajectory(os.path.join(folder, traj_name), 5000)
    lattice.do_relaxation(dt=0.01, iteration_time=5000)
    print("Done!!!")


def test_green():
    Ks = np.zeros((100, 100))

    Ks[50, 50] = 1
    # Ks[20:40, 10:30] = 1
    # Ks[70:150, 130:260] = 0.2

    dist = get_distribution_by_curvature(Ks)
    dist -= dist.min()
    fig, ax = plt.subplots()
    imshow_with_colorbar(fig, ax, dist, 'dist')
    plt.show()


def test_constant():
    folder = FIGURE_PATH
    traj_name = 'constant-test'

    Ks = np.ones((150, 150))

    print("Plotting the desired curvature")
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, "desired Ks")

    print("Calculating dist")
    dist = get_distribution_by_curvature(Ks)
    dist -= dist.min()
    # dist -= dist.mean() / 2
    # dist[dist < 0] = 0

    print("plot dist")
    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    lattice_nx = 70
    reduce_factor = 0.1
    print("Calculating defects map")
    defects_map, _ = inclusiondesign.create_defects_map_by_dist(
        dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')

    # plt.show()

    print("Creating lattice")
    lattice_gen = inclusiondesign.create_lattice_by_defects_map(defects_map)
    lattice_gen.set_inclusion_d(1.2)
    lattice_gen.set_dihedral_k(3)
    lattice_gen.set_spring_constant(20)
    print("finished creating lattice")

    print(f"expected length: {lattice_gen.get_dihedral_k() / lattice_gen.get_spring_constant()}")
    # lattice_gen.set_z_to_sphere(100000)
    lattice_gen.set_z_to_noise()

    traj_name = traj_name + '.gsd'

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)

    lattice.log_trajectory(os.path.join(folder, traj_name), 500)
    lattice.do_relaxation(dt=0.03, iteration_time=5000)
    print("Done!!!")


def main():
    create_torus()
    # test_green()
    # test_cone()
    # test_constant()


if __name__ == '__main__':
    main()
