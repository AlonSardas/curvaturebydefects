import os.path

from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects import hoomdlattice, plots, trajectory, geometry
from latticedefects.trajectory import plot_frames_from_trajectory
from latticedefects.utils import plotutils
from latticedefects.latticegenerator import TriangularLatticeGenerator
from latticedefects.geometry import calc_metric_curvature_triangular_lattice
from latticedefects.plots.latticeplotutils import create_fig_and_plot_dots, plot_flat_and_save
from latticedefects.swdesign import create_lattice_for_sphere_by_traceless_quadrupoles, create_lattice_for_negative_K_SW

FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations", 'flat-examples')


def create_lattice_constant_distribution(nx, ny, defects_jump_x, defects_jump_y):
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    for i in range(3, ny - 1, defects_jump_y):
        for j in range(1, nx - 2, defects_jump_x):
            lattice_gen.add_SW_defect(i, j)

    return lattice_gen


def constant_distribution():
    folder = os.path.join(FIGURE_PATH, 'constant-distribution')

    dihedrals = [0.01, 0.1, 1.0, 2.0, 5.0, 10.0]

    lattice_gen = create_lattice_constant_distribution(32, 36, 4, 6)
    plot_flat_and_save(lattice_gen, os.path.join(folder, 'initial'))

    lattice_gen.set_z_to_sphere(radius=1000)

    for k in dihedrals:
        print(f"simulating k={k}")
        trajectory_file = os.path.join(folder, f'trajectory-k{k:.2f}.gsd')
        lattice_gen.set_dihedral_k(k)
        lattice = lattice_gen.generate_lattice()
        lattice.log_trajectory(trajectory_file, 500)
        lattice.do_relaxation()


def main():
    constant_distribution()


if __name__ == '__main__':
    main()
