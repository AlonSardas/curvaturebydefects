import numpy as np
import os
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import plots, geometry
from latticedefects.geometry import get_interpolation
from latticedefects.plots.ellipsoid import get_ellipsoid_Ks
from latticedefects.trajectory import load_trajectory
from latticedefects.utils import plotutils


def test_curvature_by_interpolation():
    # Load the data
    FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations")
    folder = os.path.join(FIGURE_PATH, 'sphere-small-lattice')
    trajectory_file = os.path.join(folder, f'trajectory-single-SW.gsd')

    frames = load_trajectory(trajectory_file)
    frame = frames[-1]
    dots = frame.get_dots()

    nx, ny = 50, 60
    if False:
        Ks, _, _, _ = geometry.calc_metric_curvature_triangular_lattice(dots, nx, ny)
        fig, ax = plt.subplots()
        im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
        im.set_clim(-0.0001, 0.0001)
        plt.show()

    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots)
    fig, ax = plt.subplots()
    im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
    # im.set_clim(-0.0001, 0.0001)
    print(Ks[2:-2, 2:-2].sum())
    plt.show()


def test_inverse_negative():
    nx, ny = 40, 40
    desired_Ks = np.zeros((int(ny * 0.86), nx))
    desired_Ks[20, 20] = -1

    nx, ny = 40, 80
    xs = np.arange(nx)
    ys = np.arange(ny) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    zs = geometry.get_zs_based_on_Ks(nx, ny, desired_Ks, factor=10)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.plot(xs.flat,
            ys.flat,
            zs.flat, ".", color='C0', alpha=0.8)
    # ax.set_aspect('equal')

    dots = np.array([xs.flat, ys.flat, zs.flat]).transpose()
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots)

    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], desired_Ks, "desired Ks")
    plotutils.imshow_with_colorbar(fig, axes[1], Ks, "Ks")

    plt.show()


def test_inverse_ellipsoid():
    nx, ny = 60, 60
    desired_Ks = get_ellipsoid_Ks(nx, ny, 60)

    xs = np.arange(nx)
    ys = np.arange(ny) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    zs = geometry.get_zs_based_on_Ks(nx, ny, desired_Ks, factor=10)
    # zs = 60*np.sqrt(1 - (xs - nx / 2) ** 2 / nx ** 2 - (ys - ny / 2) ** 2 / ny ** 2)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.plot(xs.flat,
            ys.flat,
            zs.flat, ".", color='C0', alpha=0.8)
    # ax.set_aspect('equal')

    dots = np.array([xs.flat, ys.flat, zs.flat]).transpose()
    interp, quad_dots = get_interpolation(dots, 25, 25)
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.plot(quad_dots[0, :],
            quad_dots[1, :],
            quad_dots[2, :], ".", color='C1', alpha=0.8)

    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots, 25, 25)

    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], desired_Ks, "desired Ks")
    plotutils.imshow_with_colorbar(fig, axes[1], Ks, "Ks")

    plt.show()


def main():
    # test_curvature_by_interpolation()
    test_inverse_negative()
    # test_inverse_ellipsoid()


if __name__ == '__main__':
    main()
