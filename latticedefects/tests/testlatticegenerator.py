from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects.latticegenerator import TriangularLatticeGenerator

from latticedefects.utils import plotutils


def test_dihedrals_simple():
    nx, ny = 2, 3
    lattice_gen = TriangularLatticeGenerator(nx, ny, dihedral_k=200)

    print(lattice_gen.frame.dihedrals.group)
    assert lattice_gen.frame.dihedrals.group == [(0, 2, 1, 3), (4, 2, 5, 3), (5, 2, 3, 1)]
    # lattice_gen.dots[1, 2] = lattice_gen.dots[2, 2] = 0.1
    # lattice_gen.dots[1, :] = lattice_gen.dots[2, :]

    lattice = lattice_gen.generate_lattice()

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_bonds(ax)
    plotutils.set_3D_labels(ax)
    plotutils.set_axis_scaled(ax)

    lattice.do_relaxation()

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_bonds(ax)
    plotutils.set_3D_labels(ax)
    plotutils.set_axis_scaled(ax)
    plt.show()


def test_dihedrals_with_SW():
    nx, ny = 3, 3
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    lattice_gen.add_SW_defect(1, 0)
    print(lattice_gen.frame.dihedrals.group)
    lattice = lattice_gen.generate_lattice()
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_bonds(ax)
    plt.show()


def test_plot_dots():
    nx, ny = 10, 10
    lattice_gen = TriangularLatticeGenerator(nx, ny)

    lattice_gen.add_inclusion_defect(4, 4)
    lattice = lattice_gen.generate_lattice()
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    lattice.plot_dots(ax)
    plt.show()


def main():
    # test_dihedrals_with_SW()
    # test_dihedrals_simple()
    test_plot_dots()


if __name__ == '__main__':
    main()
