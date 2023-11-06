import math

from typing import Tuple

import numpy as np
import scipy

from latticedefects.latticegenerator import TriangularLatticeGenerator


def create_lattice_for_sphere_by_traceless_quadrupoles(nx, ny,
                                                       defects_x_jumps=4,
                                                       padding=1,
                                                       factor=0.0001):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    nx -= 1  # For nx dots we have nx-1 bonds
    middle_x = nx // 2
    max_x = nx - padding + 1
    xs_right = np.arange(middle_x, max_x, defects_x_jumps)
    print(xs_right)
    xs_left = -xs_right[1:] + 2 * middle_x
    xs = np.append(xs_left[::-1], xs_right)
    print(f"defects at {xs}")
    for x in xs:
        y_defects = round((ny - 2 * padding) * factor * (x - nx / 2) ** 2)
        x_index = round(x)
        if y_defects > ny:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{x_index}, ny={ny}, defects={y_defects}"
            )
        print(f"putting in x={x_index}, {y_defects} defects")
        for yy in np.linspace(padding, ny - padding, int(y_defects) + 2)[1:-1]:
            y = round(yy)
            print(f"Putting SW at {x_index},{y}")
            lattice_gen.add_SW_defect(y, x_index)
    return lattice_gen


def create_lattice_for_negative_K_SW(nx, ny,
                                     defects_y_jumps=4,
                                     padding=1,
                                     factor=0.0001):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    middle_y = ny // 2
    max_y = ny - padding + 1
    ys_right = np.arange(middle_y, max_y, defects_y_jumps)
    ys_left = -ys_right[1:] + 2 * middle_y
    ys = np.append(ys_left[::-1], ys_right) - 1
    print(f"defects at {ys}")
    for y in ys:
        x_defects = round((nx - 2 * padding) * factor * (y - ny / 2) ** 2)
        y_index = round(y)
        if x_defects > nx:
            raise RuntimeError(
                "Got more defects than lattice site at column "
                f"{y_index}, nx={nx}, defects={x_defects}"
            )
        print(f"putting in y={y_index}, {x_defects} defects")
        for xx in np.linspace(padding, nx - padding, int(x_defects) + 2)[1:-1]:
            x = round(xx)
            print(f"Putting SW at {y_index},{x}")
            lattice_gen.add_SW_defect(y_index, x)
    return lattice_gen


def create_cone_by_sw(nx, ny, defects_jumps, padding):
    lattice_gen = TriangularLatticeGenerator(nx, ny)
    for x in range(nx // 2, nx - padding, defects_jumps):
        for y in range(padding, ny - padding, defects_jumps):
            # if (x - nx // 2) >= abs(y - ny // 2):
            if (x - nx // 2) >= abs(y - ny // 2):
                print(f"Putting SW at {x},{y}")
                try:
                    lattice_gen.add_SW_defect(y, x)
                except RuntimeError as e:
                    print("warning: ", e.args)

    return lattice_gen


def create_cone_by_sw_symmetric(nx, ny, defects_jumps, padding, x_shift=0):
    """
    This is another way to get a cone, by using the homogeneous solutions.
    see
    https://www.desmos.com/calculator/osbqsnqug6
    """
    lattice_gen = TriangularLatticeGenerator(nx, ny)
    xs_r = np.arange(nx // 2 + x_shift, nx - padding, defects_jumps)
    xs_l = np.arange(nx // 2 + x_shift, padding, -defects_jumps)
    xs = np.append(xs_l[::-1][:-1], xs_r)
    ys_r = np.arange(ny // 2, ny - padding, defects_jumps / (np.sqrt(3) / 2))
    ys_l = np.arange(ny // 2, padding, -defects_jumps / (np.sqrt(3) / 2))
    ys = np.append(ys_l[::-1][:-1], ys_r)

    print(xs)
    print(ys)

    for x in xs:
        for y in ys:
            if abs(x - nx / 2) >= (np.sqrt(3) / 2) * abs(y - ny / 2):
                xi, yi = round(x), round(y)
                print(f"Putting SW at {xi},{yi}")
                lattice_gen.add_SW_defect(yi, xi)

    return lattice_gen


def get_distribution_by_curvature(Ks: np.ndarray) -> np.ndarray:
    """
    Use the green function to create the distribution of SW according
    the Gaussian curvature given in Ks

    :param Ks: 2D array with the desired Gaussian curvature
    :return: 2D array with same shape
    """
    nx, ny = Ks.shape
    nx, ny = 2 * nx + 1, 2 * ny + 1
    b = np.zeros((ny, nx))
    xs = np.arange(nx)
    ys = np.arange(ny)
    xs, ys = np.meshgrid(xs, ys)
    mask = np.abs(xs - (nx - 1) / 2) >= np.abs(ys - (ny - 1) / 2)
    b[mask] = 1
    return scipy.signal.convolve2d(Ks, b, mode='same')


def create_defects_map_by_dist(
        dist: np.ndarray, nx: int, reduce_factor: float = 0.9,
        x_jumps=2, y_jumps=2, padding=(0, 1, 1, 1)) -> \
        Tuple[np.ndarray, np.ndarray]:
    if np.any(dist < 0):
        raise ValueError("The given distribution has negative value!")

    dist: np.ndarray = dist.copy()
    dist /= dist.max()
    dist *= reduce_factor  # to have maximum value smaller than 1
    dist_ny, dist_nx = dist.shape
    y_x_factor = np.sqrt(3) / 2
    factor = nx / dist_nx
    ny = math.floor(dist_ny * factor / y_x_factor)


    dist_xs = np.arange(dist_nx)
    dist_ys = np.arange(dist_ny)

    defects_map = np.zeros((ny, nx))
    interp_vals = np.zeros((ny, nx))
    left_pad, right_pad, bottom_pad, top_pad = padding

    print(nx, ny, dist.shape, factor, (ny-top_pad-1)*y_x_factor/factor)

    for j in range(left_pad, nx - right_pad, x_jumps):
        for i in range(bottom_pad, ny - top_pad, y_jumps):
            x = j / factor
            y = i * y_x_factor / factor
            # print(x,y)
            val = scipy.interpolate.interpn((dist_ys, dist_xs), dist, (y, x))[0]
            r = np.random.random()
            interp_vals[i, j] = val
            if r < val:
                defects_map[i, j] = 1

    return defects_map, interp_vals


def create_lattice_by_defects_map(defects_map: np.ndarray) -> TriangularLatticeGenerator:
    ny, nx = defects_map.shape
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    for j in range(0, nx):
        for i in range(1, ny):
            if defects_map[i, j] == 1:
                lattice_gen.add_SW_defect(i, j)
    return lattice_gen
