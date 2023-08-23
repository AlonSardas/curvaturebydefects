from typing import Tuple

import math

import numpy as np
import os
import scipy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from scipy.interpolate.interpnd import LinearNDInterpolator

from latticedefects import geometry, swdesign, plots
from latticedefects.latticegenerator import TriangularLatticeGenerator
from latticedefects.plots.latticeplotutils import plot_flat_and_save
from latticedefects.trajectory import load_trajectory
from latticedefects.utils import plotutils

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations", 'test-dist-sw')


def load_raw_data():
    import latticedefects
    base_path = os.path.abspath(os.path.join(latticedefects.__file__, '..', '..', '..'))
    return np.loadtxt(os.path.join(base_path, 'Beethoven-head-data.csv'), delimiter=',')


def load_data():
    data = load_raw_data()

    angle = -0.3
    rot_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]])

    data = np.transpose(rot_matrix @ data.transpose())

    # remove the extra regions such as the hair
    data = data[data[:, 2] > -0.5]
    data = data[data[:, 1] < -0.5]
    data = data[data[:, 2] < 4]
    data = data[data[:, 0] < 1.3]
    data = data[data[:, 0] > -1.3]

    angle = -np.pi / 2
    rot_matrix = np.array([
        [1, 0, 0],
        [0, np.cos(angle), -np.sin(angle)],
        [0, np.sin(angle), np.cos(angle)]])

    data = np.transpose(rot_matrix @ data.transpose())

    data = data[data[:, 1] > 0.3]

    data[:, 2] -= 0.6

    return data


def plot_face_model():
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    dots = load_data()
    # dots = load_raw_data()
    ax.plot(dots[:, 0],
            dots[:, 1],
            dots[:, 2], ".", color='C0', alpha=0.8)
    ax.set_aspect('equal')

    plt.show()


def plot_face_curvature():
    # plot_face_model()

    dots = load_data()

    # dots = dots[np.abs(dots[:, 0]) < 0.5]
    # dots = dots[np.logical_and(dots[:, 1] > 1, dots[:, 1] < 1.5)]

    samples = 20

    _, interp_dots = geometry.get_interpolation(dots, samples, samples, interp_type='linear')
    np.nan_to_num(interp_dots, copy=False)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(121, projection="3d")
    interp_dots_t = interp_dots.transpose()
    ax.plot(interp_dots_t[:, 0],
            interp_dots_t[:, 1],
            interp_dots_t[:, 2], ".", color='C0', alpha=0.8)
    ax.set_aspect('equal')

    samples = 50
    _, interp_dots2 = geometry.get_interpolation(interp_dots.transpose(), samples, samples)
    interp_dots2 = interp_dots2.transpose()
    ax: Axes3D = fig.add_subplot(122, projection="3d")
    interp_dots_t = interp_dots.transpose()
    ax.plot(interp_dots2[:, 0],
            interp_dots2[:, 1],
            interp_dots2[:, 2], ".", color='C0', alpha=0.5)
    ax.set_aspect('equal')

    Ks, Hs = geometry.calculate_curvatures_by_interpolation(interp_dots.transpose(), samples, samples)

    plt.show()
    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Hs, "H")

    np.nan_to_num(Ks, copy=False)
    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
    # im.set_clim(-10, 10)

    Ks_avg = ndimage.gaussian_filter(Ks, sigma=1.5, mode='constant')
    print(Ks_avg)
    ax = axes[1]
    im = plotutils.imshow_with_colorbar(fig, ax, Ks_avg, "K averaged")

    plt.show()


def get_face_curvature() -> np.ndarray:
    dots = load_data()
    print(dots.shape)

    samples = 20

    _, interp_dots = geometry.get_interpolation(dots, samples, samples, interp_type='linear')
    np.nan_to_num(interp_dots, copy=False)

    samples = 50
    _, interp_dots2 = geometry.get_interpolation(interp_dots.transpose(), samples, int(samples * 1.5))
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(interp_dots.transpose(), samples, int(samples * 1.5))

    Ks_avg = ndimage.gaussian_filter(Ks, sigma=1.5, mode='constant')
    return Ks_avg


def get_distribution():
    Ks_avg = get_face_curvature()

    fig, axes = plt.subplots(1, 2)
    ax = axes[0]
    plotutils.imshow_with_colorbar(fig, ax, Ks_avg, "K averaged")

    dist = swdesign.get_distribution_by_curvature(Ks_avg)
    dist -= dist.min()

    ax = axes[1]
    plotutils.imshow_with_colorbar(fig, ax, dist, 'dist')
    plt.show()


def test_convolve():
    a = np.zeros((7, 7))
    a[2, 4] = 1

    nx, ny = 51, 51
    b = np.zeros((ny, nx))
    xs = np.arange(nx)
    ys = np.arange(ny)

    xs, ys = np.meshgrid(xs, ys)
    mask = np.abs(xs - (nx - 1) / 2) >= np.abs(ys - (ny - 1) / 2)
    b[mask] = 1

    c = scipy.signal.convolve2d(a, b, mode='same')

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, c, 'convolved')
    plt.show()


def test_dist():
    def Ks_func(x, y):
        g1 = np.exp(-((x - 15) ** 2 + (y / 2 - 10) ** 2) / 15)
        g2 = 0.6 * np.exp(-(((x - 27) / 2) ** 2 + (y - 20) ** 2) / 10)
        return g1 + g2

    def Ks_func(x, y):
        return np.logical_and(x==20, y==25)

    nx = 40
    ny = 36
    xs = np.arange(nx)
    ys = np.arange(ny)

    Xs, Ys = np.meshgrid(xs, ys)
    Ks = Ks_func(Xs, Ys)

    fig, axes = plt.subplots(1, 4)
    fig: Figure = fig
    ax = axes[0]
    plotutils.imshow_with_colorbar(fig, ax, Ks, "Ks")

    dist = swdesign.get_distribution_by_curvature(Ks)
    dist -= dist.min()
    ax = axes[1]
    plotutils.imshow_with_colorbar(fig, ax, dist, 'dist')

    lattice_gen, defects_map, interp_vals = \
        create_lattice_by_dist(dist, 60, reduce_factor=0.3)
    ax = axes[2]
    plotutils.imshow_with_colorbar(fig, ax, interp_vals, 'interpolated')

    ax = axes[3]
    plotutils.imshow_with_colorbar(fig, ax, defects_map, 'defects map')
    fig.savefig(os.path.join(FIGURE_PATH, 'dist.svg'))

    lattice_gen.set_dihedral_k(3.5)

    print("saving bonds")
    plot_flat_and_save(lattice_gen, os.path.join(FIGURE_PATH, 'initial'))

    lattice_gen.set_z_to_sphere(radius=1000)
    trajectory_file = os.path.join(FIGURE_PATH, f'trajectory.gsd')
    lattice = lattice_gen.generate_lattice()
    lattice.log_trajectory(trajectory_file, 500)
    lattice.do_relaxation(iteration_time=1000)

    # plt.show()


def plot_test_dist_result():
    trajectory_file = os.path.join(FIGURE_PATH, f'trajectory.gsd')
    frames = load_trajectory(trajectory_file)
    frame = frames[-1]
    dots = frame.frame.particles.position
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots)
    fig_Ks, ax_Ks = plt.subplots()
    Ks = Ks[30:-30, 30:-30]
    plotutils.imshow_with_colorbar(fig_Ks, ax_Ks, Ks, "K")
    plt.show()


def create_lattice_by_dist(dist: np.ndarray, nx: int, reduce_factor: float = 0.9) -> \
        Tuple[TriangularLatticeGenerator, np.ndarray, np.ndarray]:
    if np.any(dist < 0):
        raise ValueError("The given distribution has negative value!")

    dist: np.ndarray = dist.copy()
    dist /= dist.max()
    dist *= reduce_factor  # to have maximum value smaller than 1
    dist_ny, dist_nx = dist.shape
    y_x_factor = np.sqrt(3) / 2
    factor = nx / dist.shape[1]
    ny = math.floor(dist_ny / dist_nx * nx / y_x_factor)
    print(ny)

    dist_xs = np.arange(dist_nx)
    dist_ys = np.arange(dist_ny)

    lattice_gen = TriangularLatticeGenerator(nx, ny)
    dots = lattice_gen.dots
    min_x = np.min(dots[:, 0])
    min_y = np.min(dots[:, 1])
    print(factor)

    defects_map = np.zeros((ny, nx))
    interp_vals = np.zeros((ny, nx))
    x_jumps = 2
    y_jumps = 2
    for j in range(0, nx - 1, x_jumps):
        for i in range(1, ny - 1, y_jumps):
            dot = dots[lattice_gen.indices[i, j], :]
            x, y, z = dot
            x -= min_x
            y -= min_y
            x /= factor
            y /= factor
            val = scipy.interpolate.interpn((dist_ys, dist_xs), dist, (y, x))[0]
            r = np.random.random()
            interp_vals[i, j] = val
            # print(x, y, val)
            if r < val:
                # print(f"Putting SW at {j}, {i}")
                defects_map[i, j] = 1
                lattice_gen.add_SW_defect(i, j)
    return lattice_gen, defects_map, interp_vals


def test_face_design():
    Ks_avg = get_face_curvature()
    print(Ks_avg.shape)

    dist = swdesign.get_distribution_by_curvature(Ks_avg)
    dist -= dist.min()
    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    reduce_factor = 1.0
    print("generating the lattice")
    lattice_gen, defects_map, interp_vals = \
        create_lattice_by_dist(dist, 60, reduce_factor=reduce_factor)
    print("done generating the lattice")

    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')

    lattice_gen.set_dihedral_k(15)
    lattice_gen.set_z_by_curvature(Ks_avg, 0.00005)
    lattice = lattice_gen.generate_lattice()

    folder = FIGURE_PATH
    lattice.save_frame(os.path.join(folder, 'face-sw-initial.gsd'))
    lattice.log_trajectory(os.path.join(folder, 'face-sw-traj.gsd'), 1000)
    lattice.do_relaxation(dt=0.01, force_tol=1e-3, energy_tol=1e-6, iteration_time=1000)
    lattice.save_frame(os.path.join(folder, 'face-sw.gsd'))


def test_inverse():
    Ks_face = get_face_curvature()
    nx, ny = 40, 80
    xs = np.arange(nx)
    ys = np.arange(ny) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    zs = geometry.get_zs_based_on_Ks(nx, ny, Ks_face, factor=0.001)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.plot(xs.flat,
            ys.flat,
            zs.flat, ".", color='C0', alpha=0.8)
    # ax.set_aspect('equal')

    dots = np.array([xs.flat, ys.flat, zs.flat]).transpose()
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots)

    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], Ks_face, "K face")
    plotutils.imshow_with_colorbar(fig, axes[1], Ks, "K")

    plt.show()


def main():
    # plot_face_model()
    # plot_face_curvature()
    # test_convolve()
    # get_distribution()
    # test_dist()
    # plot_test_dist_result()
    # test_face_design()
    test_inverse()

if __name__ == '__main__':
    main()
