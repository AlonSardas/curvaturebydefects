import numpy as np
import os
import scipy
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import swdesign, plots, geometry
from latticedefects.swdesign import get_distribution_by_curvature
from latticedefects.trajectory import load_trajectory
from latticedefects.utils import plotutils
from latticedefects.utils.plotutils import imshow_with_colorbar

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations", 'torus')


# It seems that since we need negative curvature inside and positive outside
# then we must also include areas around them to support this change in curvature


def create_torus():
    folder = FIGURE_PATH

    # lattice_gen = swdesign.create_lattice_for_sphere_by_traceless_quadrupoles(120, 120)
    # lattice_gen.remove_dots_inside(25)
    # lattice_gen.remove_dots_outside(50)
    # Ks = get_torus_Ks(100, 100, 20, 50)
    # Ks = get_torus_Ks(100, 100, 20, 35)
    # Ks = get_torus_Ks(100, 100, 14, 20)
    Ks = get_torus_Ks(100, 100, 0, 72)

    lattice_nx = 100

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks - torus')

    # plt.show()

    # dist = swdesign.get_distribution_by_curvature(Ks)
    print("Calculating dist")
    dist = get_torus_dist_by_Ks(Ks, (190, 190))
    dist -= dist.min()

    fig, axes = plt.subplots(1, 2)
    print("plot dist")
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')
    # plt.show()

    reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = swdesign.create_defects_map_by_dist(dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects.png'))

    # plt.show()
    print("Creating lattice")
    lattice_gen = swdesign.create_lattice_by_defects_map(defects_map)
    print("finished creating lattice")

    lattice_gen.set_z_to_sphere()
    # lattice_gen.remove_dots_inside(7)
    # lattice_gen.remove_dots_outside(35)
    lattice_gen.set_dihedral_k(10)
    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    print(rs)
    lattice_gen.dots[:, 2] = -0.005 * (rs - lattice_nx / 2) ** 2
    print(lattice_gen.dots[:, 2])

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)
    # lattice.plot_bonds(ax)
    # lattice.plot_indexes_text(ax)

    # plt.show()

    lattice.log_trajectory(os.path.join(folder, 'traj.gsd'), 1000)
    lattice.do_relaxation()


def create_positive_K_on_annulus():
    folder = FIGURE_PATH

    Ks = get_torus_Ks(100, 100, -400, 72)

    lattice_nx = 70

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks - torus')

    plt.show()

    # dist = swdesign.get_distribution_by_curvature(Ks)
    print("Calculating dist")
    dist = get_torus_dist_by_Ks(Ks, (190, 190))
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

    lattice_gen.set_z_to_sphere()
    lattice_gen.remove_dots_inside(7)
    # lattice_gen.remove_dots_outside(35)
    lattice_gen.set_dihedral_k(15)
    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    print(rs)
    lattice_gen.dots[:, 2] = -0.005 * (rs - lattice_nx / 4) ** 2
    print(lattice_gen.dots[:, 2])

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)
    # lattice.plot_bonds(ax)
    # lattice.plot_indexes_text(ax)

    # plt.show()

    lattice.log_trajectory(os.path.join(folder, 'traj-pos-only.gsd'), 1000)
    lattice.do_relaxation()


def create_torus2():
    folder = FIGURE_PATH

    # Ks = get_torus_Ks_with_jump(100, 100, 10, 30)
    Ks = get_torus_Ks_with_jump(100, 100, 0, 54)
    Ks[Ks == 0] = 1

    lattice_nx = 80

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks - torus')
    plt.show()

    print("Calculating dist")
    dist = get_torus_dist_by_Ks(Ks, (100, 100))
    # dist = get_torus_dist_by_Ks(Ks, None)
    dist -= dist.min()

    fig, axes = plt.subplots(1, 2)
    print("plot dist")
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = swdesign.create_defects_map_by_dist(dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects-v2.png'))

    plt.show()
    print("Creating lattice")
    lattice_gen = swdesign.create_lattice_by_defects_map(defects_map)
    print("finished creating lattice")

    lattice_gen.set_z_to_sphere()
    # lattice_gen.remove_dots_inside(10)
    lattice_gen.remove_dots_outside(65)
    lattice_gen.set_dihedral_k(80)
    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    lattice_gen.dots[:, 2] = -0.010 * (rs - lattice_nx / 3) ** 2

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)
    # lattice.plot_bonds(ax)
    # lattice.plot_indexes_text(ax)

    # plt.show()

    lattice.log_trajectory(os.path.join(folder, 'traj2.gsd'), 1000)
    lattice.do_relaxation()


def create_torus3():
    folder = FIGURE_PATH

    Ks = np.ones((100, 100))
    set_values_in(Ks, 25, -5)
    set_values_in(Ks, 10, 0)
    set_values_out(Ks, 35, 0)
    lattice_nx = 100

    fig, ax = plt.subplots()
    plotutils.imshow_with_colorbar(fig, ax, Ks, 'Ks - torus')
    # plt.show()

    print("Calculating dist")
    # dist = get_torus_dist_by_Ks(Ks, (50, 50))
    dist = get_torus_dist_by_Ks(Ks, None)
    dist -= dist.min()
    increase_contrast(dist, 5)

    fig, axes = plt.subplots(1, 2)
    print("plot dist")
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = swdesign.create_defects_map_by_dist(dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[1], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects-v3.png'))

    plt.show()
    print("Creating lattice")
    lattice_gen = swdesign.create_lattice_by_defects_map(defects_map)
    print("finished creating lattice")

    print(f"expected length: {lattice_gen.get_dihedral_k() / lattice_gen.get_spring_constant()}")

    r0 = 10
    lattice_gen.set_z_to_sphere()
    lattice_gen.remove_dots_inside(r0)
    # lattice_gen.remove_dots_outside(55)
    lattice_gen.set_dihedral_k(7)
    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    lattice_gen.dots[:, 2] = -0.010 * (rs - r0 - lattice_nx / 4) ** 2

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)
    # lattice.plot_bonds(ax)
    # lattice.plot_indexes_text(ax)

    # plt.show()

    lattice.log_trajectory(os.path.join(folder, 'traj3.gsd'), 1000)
    lattice.do_relaxation()


def create_torus4():
    folder = FIGURE_PATH
    traj_name = 'traj4'

    nx, ny = 150, 170
    samples_x, samples_y = 80, 80

    r0 = 40.0
    sigma = 18
    Z0 = 20.0

    xs = np.arange(nx)
    ys = np.arange(ny) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt((xs - nx / 2) ** 2 + (ys - ny / 2 * np.sqrt(3) / 2) ** 2)

    # zs = Z0 * np.exp(-(rs - r0) ** 2 / (2 * sigma ** 2))
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

    # plt.show()

    print("Calculating dist")
    dist = get_torus_dist_by_Ks(Ks, (30, 30))
    # dist = get_torus_dist_by_Ks(Ks, None)
    dist -= dist.min()

    print("plot dist")
    fig, axes = plt.subplots(1, 3)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    threshold = 0.04
    dist[dist >= threshold] = 1
    dist[dist < threshold] = 0
    should_inverse_defects = False
    if should_inverse_defects:
        dist = 1 - dist
        traj_name = traj_name + '-inv'

    # increase_contrast(dist, 50)

    plotutils.imshow_with_colorbar(fig, axes[1], dist, 'dist - high contrast')

    lattice_nx = 70
    # For some reason, smaller factor makes the simulations reach faster to equilibrium
    # reduce_factor = 0.8
    reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = swdesign.create_defects_map_by_dist(dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[2], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects-v4.png'))

    plt.show()
    print("Creating lattice")
    lattice_gen = swdesign.create_lattice_by_defects_map(defects_map)
    lattice_gen.set_dihedral_k(8)
    lattice_gen.set_spring_constant(1)
    print("finished creating lattice")

    print(f"expected length: {lattice_gen.get_dihedral_k() / lattice_gen.get_spring_constant()}")

    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    r0 = r0 / nx * lattice_nx
    Z0 = 2
    lattice_gen.dots[:, 2] = Z0 * (rs / r0) ** 2 * np.exp(-(rs / r0) ** 2)
    lattice_gen.dots[:, 0] *= 0.9
    lattice_gen.dots[:, 1] *= 0.9

    # lattice_gen.dots[:, 2] = zs.flatten()
    with_hole = False
    if with_hole:
        lattice_gen.remove_dots_inside(10)
        traj_name = traj_name + '-hole'
    traj_name = traj_name + '.gsd'

    lattice_gen.remove_dots_outside(35)
    # rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    # lattice_gen.dots[:, 2] = -0.010 * (rs - r0 - lattice_nx / 4) ** 2

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)
    # lattice.plot_bonds(ax)
    # lattice.plot_indexes_text(ax)

    # plt.show()

    # Choose the Q according to length

    lattice.log_trajectory(os.path.join(folder, traj_name), 15000)
    lattice.do_relaxation(dt=0.09, iteration_time=15000)

    # This required fine-tuning of the bending energy and the reduce factor of the defects
    # The reduce factor was helpful to make the bump in the middle less sharp
    # so the surface will look more similar to a torus

    # This is also a bit tricky, since to have this shape we must have the points in the middle
    # the defects appear only there...


def create_torus5():
    folder = FIGURE_PATH
    traj_name = 'traj5c'

    nx, ny = 150, 170
    samples_x, samples_y = 80, 80

    r0 = 40.0
    Z0 = 20.0

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
    dist = get_torus_dist_by_Ks(Ks, None)
    dist -= dist.min()

    print("plot dist")
    fig, axes = plt.subplots(1, 3)
    plotutils.imshow_with_colorbar(fig, axes[0], dist, 'dist')

    threshold = 0.048
    dist[dist >= threshold] = 1
    dist[dist < threshold] = 0

    plotutils.imshow_with_colorbar(fig, axes[1], dist, 'dist - high contrast')

    lattice_nx = 70
    # For some reason, smaller factor makes the simulations reach faster to equilibrium
    # reduce_factor = 0.6
    reduce_factor = 1.0
    print("Calculating defects map")
    defects_map, _ = swdesign.create_defects_map_by_dist(dist, lattice_nx, reduce_factor)
    plotutils.imshow_with_colorbar(fig, axes[2], defects_map, 'defects')
    fig.savefig(os.path.join(folder, 'dist-and-defects-v5.png'))

    # plt.show()
    print("Creating lattice")
    lattice_gen = swdesign.create_lattice_by_defects_map(defects_map)
    lattice_gen.set_dihedral_k(0.3)
    lattice_gen.set_spring_constant(1)
    print("finished creating lattice")

    print(f"expected length: {lattice_gen.get_dihedral_k() / lattice_gen.get_spring_constant()}")

    rs = np.sqrt(lattice_gen.dots[:, 0] ** 2 + lattice_gen.dots[:, 1] ** 2)
    r0 = r0 / nx * lattice_nx
    Z0 = 18.0
    lattice_gen.dots[:, 2] = Z0 * (rs / r0) ** 2 * np.exp(-(rs / r0) ** 2)
    lattice_gen.dots[:, 0] *= 0.9
    lattice_gen.dots[:, 1] *= 0.9

    with_hole = True
    if with_hole:
        lattice_gen.remove_dots_inside(5.0)
        traj_name = traj_name + '-hole'
    traj_name = traj_name + '.gsd'

    lattice_gen.remove_dots_outside(35)

    lattice = lattice_gen.generate_lattice()
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    lattice.plot_dots(ax)

    lattice.log_trajectory(os.path.join(folder, traj_name), 5000)
    lattice.do_relaxation(dt=0.1, iteration_time=5000, energy_tol=1e-8, force_tol=1e-3)
    print("finished 1st relaxation")
    lattice.do_relaxation(dt=0.05, iteration_time=5000, energy_tol=1e-9)
    # fire_args={'finc_dt': 1.10, 'min_steps_adapt': 5}
    print("Done!!!")

    # When dt is large ~ 0.15, it converges quite rapidly to a minimum - but this is not
    # the minimum configuration that we want.
    # When dt is smaller, the time to converge becomes very large,
    # and sometimes there are fluctuations, that increase the total energy


def get_torus_dist_by_Ks(Ks, green_func_box=None):
    if green_func_box is None:
        return get_distribution_by_curvature(Ks)
    # Here we make the green function only to some small area
    nx, ny = green_func_box
    b = np.zeros((ny, nx))
    xs = np.arange(nx)
    ys = np.arange(ny)
    xs, ys = np.meshgrid(xs, ys)
    mask = np.abs(xs - (nx - 1) / 2) >= np.abs(ys - (ny - 1) / 2)
    b[mask] = 1
    return scipy.signal.convolve2d(Ks, b, mode='same')


def increase_contrast(dist: np.ndarray, factor):
    k_max = dist.max()
    dist /= k_max
    dist[:, :] = np.exp(factor * dist) / np.exp(factor)
    dist *= k_max
    return dist


def plot_torus_Ks():
    Ks = get_torus_Ks(100, 100, 20, 50)
    fig, ax = plt.subplots()
    imshow_with_colorbar(fig, ax, Ks, 'Ks - torus')
    plt.show()


def get_torus_Ks(nx, ny, r0, r1, factor=1.0):
    """
    This is not really accurate, just something similar...

    See
    https://trecs.se/torus.php
    for exact, yet in bad coordinates
    """
    # Ks = np.zeros((ny, nx))
    xs = np.arange(nx) - nx / 2
    ys = np.arange(ny) - ny / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt(xs ** 2 + ys ** 2)

    Ks = factor * (rs - (r0 + r1) / 2)
    Ks[rs > r1] = 0
    Ks[rs < r0] = 0
    return Ks


def get_torus_Ks_with_jump(nx, ny, r0, r1, factor=1.0):
    """
    A simple curvature map with negative inside, and positive outside
    """
    Ks = np.zeros((ny, nx))
    xs = np.arange(nx) - nx / 2
    ys = np.arange(ny) - ny / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt(xs ** 2 + ys ** 2)

    Ks[rs <= (r0 + r1) / 2] = -1
    Ks[rs > (r0 + r1) / 2] = 1

    Ks[rs > r1] = 0
    Ks[rs < r0] = 0
    Ks *= factor
    return Ks


def set_values_in(Ks: np.ndarray, r0, value):
    ny, nx = Ks.shape
    xs = np.arange(nx) - nx / 2
    ys = np.arange(ny) - ny / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt(xs ** 2 + ys ** 2)

    Ks[rs <= r0] = value
    return Ks


def set_values_out(Ks: np.ndarray, r0, value):
    ny, nx = Ks.shape
    xs = np.arange(nx) - nx / 2
    ys = np.arange(ny) - ny / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt(xs ** 2 + ys ** 2)

    Ks[rs > r0] = value
    return Ks


def test_inverse_design():
    # Ks = np.zeros((100, 100))
    Ks = np.ones((300, 300))
    Ks[:, :] = 0
    # set_values_in(Ks, 25, -5)
    # set_values_in(Ks, 90, 2)
    # set_values_out(Ks, 40, 0)
    # set_values_in(Ks, 10, 0)
    # set_values_out(Ks, 35, 0)

    Ks[20:40, 10:30] = 1
    Ks[70:150, 130:260] = 0.2

    nx, ny = 90, 110
    xs = np.arange(nx)
    ys = np.arange(ny) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    zs = geometry.get_zs_based_on_Ks(nx, ny, Ks, factor=0.000000000000001)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    ax.plot(xs.flat,
            ys.flat,
            zs.flat, ".", color='C0', alpha=0.8)

    dots = np.array([xs.flat, ys.flat, zs.flat]).transpose()
    actual_Ks, Hs = geometry.calculate_curvatures_by_interpolation(
        dots, 50, 50)

    fig, axes = plt.subplots(1, 2)
    plotutils.imshow_with_colorbar(fig, axes[0], Ks, "desired Ks")
    plotutils.imshow_with_colorbar(fig, axes[1], actual_Ks, "Ks")

    plt.show()


def main():
    # create_torus()
    # create_torus2()
    # plot_torus_Ks()
    # create_torus3()
    # create_positive_K_on_annulus()
    # test_inverse_design()
    # create_torus4()
    create_torus5()


def continue_relaxation():
    folder = FIGURE_PATH
    file_path = os.path.join(folder, 'torus5-hole.gsd')

    traj = load_trajectory(file_path)
    # traj[0].
    # Lattice



if __name__ == '__main__':
    main()
