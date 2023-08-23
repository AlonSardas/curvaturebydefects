"""
Here we want to see that indeed having several defects contributes linearly
to the Gaussian curvature.

It seems that we have a tradeoff: with small bending the defects are very local
and the surfaces are not smooth.
With large bending, we cannot create shapes with many details
"""
import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import plots, geometry, swdesign
from latticedefects.latticegenerator import TriangularLatticeGenerator
from latticedefects.plots import faceplot
from latticedefects.utils import plotutils

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations", "Playground")

dihedral_k = 1.05
folder = os.path.join(FIGURE_PATH, 'superposition')


def plot_single():
    nx, ny = 50, 60
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=dihedral_k)

    lattice_gen.add_SW_defect(ny // 2, nx // 2)
    lattice_gen.set_z_to_sphere(radius=1000)

    lattice = lattice_gen.generate_lattice()
    lattice.do_relaxation(force_tol=1e-6, iteration_time=500)
    lattice.save_frame(os.path.join(folder, 'single-sw.gsd'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_dots(ax)
    plotutils.set_3D_labels(ax)

    _plot_K_and_save(lattice, 'single-sw.svg')

    plt.show()


def plot_scattered():
    nx, ny = 50, 60
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=dihedral_k)

    lattice_gen.add_SW_defect(10, 15)
    lattice_gen.add_SW_defect(39, 22)
    lattice_gen.add_SW_defect(22, 40)
    lattice_gen.set_z_to_sphere(radius=1000)

    lattice = lattice_gen.generate_lattice()
    lattice.do_relaxation(force_tol=1e-6, iteration_time=500)
    lattice.save_frame(os.path.join(folder, 'scattered-sw.gsd'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_dots(ax)
    plotutils.set_3D_labels(ax)

    _plot_K_and_save(lattice, 'scattered-sw.svg')

    plt.show()


def plot_cone():
    nx, ny = 80, 100
    desired_Ks = np.zeros((int(ny * 0.86), nx))
    desired_Ks[30, 30] = 1
    desired_Ks[70, 50] = 1
    dist = swdesign.get_distribution_by_curvature(desired_Ks)
    dist -= dist.min()
    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    reduce_factor = 0.5
    reduce_factor = 1.0
    lattice_gen, defects_map, interp_vals = \
        faceplot.create_lattice_by_dist(dist, 60, reduce_factor=reduce_factor)

    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')

    # plt.draw()
    # plt.pause(0.001)
    # plt.show(block=False)

    # plt.ion()

    print(lattice_gen.nx, lattice_gen.ny)

    # lattice_gen.set_dihedral_k(dihedral_k+0.3)
    lattice_gen.set_dihedral_k(dihedral_k * 20)
    lattice_gen.set_z_by_curvature(desired_Ks, 0.1)
    lattice = lattice_gen.generate_lattice()

    lattice.save_frame(os.path.join(folder, 'cone-sw-initial.gsd'))
    lattice.log_trajectory(os.path.join(folder, 'cone-sw-traj.gsd'), 1000)
    lattice.do_relaxation(dt=0.01, force_tol=1e-3, iteration_time=500)
    lattice.save_frame(os.path.join(folder, 'cone-sw.gsd'))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_dots(ax)
    plotutils.set_3D_labels(ax)

    _plot_K_and_save(lattice, 'cone-sw.svg')

    plt.show()


def _plot_K_and_save(lattice, name):
    dots = lattice.get_dots()
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots, x_samples=240, y_samples=240)
    fig_Ks, ax_Ks = plt.subplots()
    plotutils.imshow_with_colorbar(fig_Ks, ax_Ks, Ks, "K")
    fig_Ks.savefig(os.path.join(folder, name))


def main():
    # plot_single()
    # plot_scattered()
    plot_cone()


if __name__ == '__main__':
    main()
