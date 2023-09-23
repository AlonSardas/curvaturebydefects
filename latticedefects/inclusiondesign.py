import numpy as np
import scipy
from typing import Tuple

from latticedefects import swdesign
from latticedefects.latticegenerator import TriangularLatticeGenerator


def create_sphere_by_inclusion(nx, ny, disk_width, factor, C0, padding,
                               inclusion_d=0.8) -> TriangularLatticeGenerator:
    # TODO: fix the plot for this distribution, I think I didn't treat the radial
    # TODO: dependency correctly

    lattice_gen = TriangularLatticeGenerator(nx, ny, inclusion_d=inclusion_d)
    angle = 0
    for r in range(disk_width, nx, disk_width):
        disk_area = np.pi * r ** 2 - np.pi * (r - disk_width) ** 2
        defects_in_disk = round((factor * (r - disk_width / 2) ** 2 + C0) * disk_area)
        print(f"r={r}, disk area: {disk_area}, defect in disk: {defects_in_disk}")
        if defects_in_disk <= 1:
            continue
        for d in range(defects_in_disk):
            x = round(nx / 2 + np.cos(angle) * (r - disk_width / 2))
            y = round(ny / 2 + np.sin(angle) * (r - disk_width / 2))
            angle += 2 * np.pi / defects_in_disk
            if x < padding or y < padding or x >= nx - padding or y >= ny - padding:
                continue
            print(f"Putting inclusion at {x},{y}, for angle={angle / np.pi * 180 % 360:.3f}")
            try:
                lattice_gen.add_inclusion_defect(y, x)
            except RuntimeError as e:
                print("warning: ", e.args)
        angle += 2 * np.pi / defects_in_disk / 2

    return lattice_gen


def create_cone_by_inclusion(nx, ny, disk_width, factor, r0, padding, inclusion_d=0.8) -> TriangularLatticeGenerator:
    lattice_gen = TriangularLatticeGenerator(nx, ny, inclusion_d=inclusion_d)
    angle = 0
    for r in range(disk_width, nx, disk_width):
        disk_area = np.pi * r ** 2 - np.pi * (r - disk_width) ** 2
        defects_in_disk = round(factor * (np.log(((r - disk_width / 2) / r0))) * disk_area)
        print(f"r={r}, disk area: {disk_area}, defect in disk: {defects_in_disk}")
        if defects_in_disk <= 1:
            continue
        for d in range(defects_in_disk):
            x = round(nx / 2 + np.sqrt(3) / 2 * np.cos(angle) * (r - disk_width / 2))
            y = round(ny / 2 + np.sin(angle) * (r - disk_width / 2))
            angle += 2 * np.pi / defects_in_disk
            if x < padding or y < padding or x >= nx - padding or y >= ny - padding:
                continue
            print(f"Putting inclusion at {x},{y}, for angle={angle / np.pi * 180 % 360:.3f}")
            try:
                lattice_gen.add_inclusion_defect(y, x)
            except RuntimeError as e:
                print("warning: ", e.args)
        angle += 2 * np.pi / defects_in_disk / 2

    return lattice_gen


def get_distribution_by_curvature(Ks: np.ndarray) -> np.ndarray:
    """
    Use the green function to create the distribution of inclusion according
    the Gaussian curvature given in Ks

    :param Ks: 2D array with the desired Gaussian curvature
    :return: 2D array with same shape
    """
    nx, ny = Ks.shape
    nx, ny = 2 * nx + 1, 2 * ny + 1
    xs = np.arange(nx) - (nx - 1) / 2
    ys = (np.arange(ny) - (ny - 1) / 2) * np.sqrt(3) / 2
    xs, ys = np.meshgrid(xs, ys)
    rs = np.sqrt(xs ** 2 + ys ** 2)
    eps = 0.01
    green = np.log(rs + eps)
    return scipy.signal.convolve2d(Ks, green, mode='same')


def create_lattice_by_defects_map(defects_map: np.ndarray) -> TriangularLatticeGenerator:
    ny, nx = defects_map.shape
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    for j in range(0, nx):
        for i in range(1, ny):
            if defects_map[i, j] == 1:
                lattice_gen.add_inclusion_defect(i, j)
    return lattice_gen


def create_defects_map_by_dist(
        dist: np.ndarray, nx: int, reduce_factor: float = 1.0,
        x_jumps=1, y_jumps=1, padding=(0, 0, 0, 0)) -> \
        Tuple[np.ndarray, np.ndarray]:
    return swdesign.create_defects_map_by_dist(
        dist, nx, reduce_factor,
        x_jumps=x_jumps, y_jumps=y_jumps,
        padding=padding)
