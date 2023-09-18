import os

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import plots, swdesign
from latticedefects.utils import plotutils

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations", 'ellipsoid')


def create_ellipsoid():
    Ks = get_ellipsoid_Ks(140, 35, 30)
    Ks = Ks[:,40:-40]
    folder = FIGURE_PATH

    lattice_nx = 60

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks - ellipsoid')

    plt.show()

    print("Calculating dist")
    dist = swdesign.get_distribution_by_curvature(Ks)
    dist -= dist.min()

    fig, axes = plt.subplots(1, 2)
    print("plot dist")
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = swdesign.create_defects_map_by_dist(dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects.png'))

    plt.show()
    print("Creating lattice")
    lattice_gen = swdesign.create_lattice_by_defects_map(defects_map)
    print("finished creating lattice")

    lattice_gen.set_z_by_curvature(Ks, 0.1)
    lattice_gen.set_dihedral_k(15)

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)

    lattice.log_trajectory(os.path.join(folder, 'traj.gsd'), 1000)
    lattice.do_relaxation()


def get_ellipsoid_Ks(nx, ny, c_factor) -> np.ndarray:
    # See https://www.johndcook.com/blog/2019/10/07/curvature-of-an-ellipsoid/
    a = nx
    b = ny
    c = c_factor

    xs = np.arange(nx) - nx / 2
    ys = np.arange(ny) - ny / 2
    x, y = np.meshgrid(xs, ys)

    Ks = 1 / (a * b * c) ** 2 * \
         (x ** 2 / a ** 4 + y ** 2 / b ** 4 + 1 / c ** 2 * (1 - x ** 2 / a ** 2 - y ** 2 / b ** 2)) ** (-2)
    return Ks


def main():
    create_ellipsoid()


if __name__ == '__main__':
    main()
