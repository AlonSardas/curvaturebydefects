import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import FIREalgorithm
import springlattice


def create_simple_test_lattice():
    xs, ys = np.meshgrid(np.arange(3), np.arange(4))
    xs = xs.flatten()
    ys = ys.flatten()
    zs = np.zeros(len(xs))
    dots = np.array([xs, ys, zs]).transpose()

    springs = np.zeros((len(xs), 2), int)

    return springlattice.SpringLattice(dots, springs, 1, 1)


def create_chain_lattice(n=5) -> springlattice.SpringLattice:
    dots = np.zeros((n, 3))
    dots[:, 0] = np.arange(n)

    dots += 0.2 * (np.random.random(dots.shape) - 0.5)

    springs = np.zeros((n, 1), dtype=int)
    springs[:, 0] = np.arange(n) + 1
    springs[-1, 0] = 0

    return springlattice.SpringLattice(dots, springs, 1, 1)


def test_relaxation():
    lattice = create_chain_lattice(n=10)
    # print("forces", lattice.calculate_forces())

    FIREalgorithm.FIRE(lattice)

    print(lattice.dots)
    print(lattice.calculate_forces())

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.plot(lattice.dots[:, 0], lattice.dots[:, 1], lattice.dots[:, 2])
    set_axis_scaled(ax)

    plt.show()


def test_simple_triangles():
    dots = np.zeros((4, 3))
    dots[0, :2] = [-0.5, 0]
    dots[1, :2] = [0, 0.7]
    dots[2, :2] = [0.5, 0]
    dots[3, :2] = [0, -0.7]

    dots[:, 2] += 0.05 * (np.random.random(4) - 0.5)

    springs = np.zeros((4, 3), dtype=int)
    springs[0, :] = [1, 2, 3]
    springs[1, :] = [0, 2, 3]
    springs[2, :] = [0, 1, 3]
    springs[3, :] = [0, 1, 2]

    springs_constants = 20 * np.ones(springs.shape)
    # springs_constants[1, 2] = 0
    # springs_constants[3, 1] = 0
    # springs_constants[0, 1] = 0
    # springs_constants[2, 0] = 0

    lattice = springlattice.SpringLattice(dots, springs, springs_constants, 1)

    FIREalgorithm.FIRE(lattice)

    print(lattice.calculate_forces())
    print(lattice.dots)

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.plot(lattice.dots[:, 0], lattice.dots[:, 1], lattice.dots[:, 2])
    set_axis_scaled(ax)

    plt.show()


def test_triangular_lattice():
    lattice = springlattice.create_triangular_lattice(5, 7)

    print(lattice.dots[:, :2])

    print(lattice.calculate_forces())
    FIREalgorithm.FIRE(lattice)
    print(lattice.calculate_forces())

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d')
    ax.plot(lattice.dots[:, 0], lattice.dots[:, 1], lattice.dots[:, 2], '.')
    set_axis_scaled(ax)
    plt.show()


def set_axis_scaled(ax: Axes3D):
    max_lim = max(ax.get_xlim()[1], ax.get_ylim()[1], ax.get_zlim()[1])
    min_lim = min(ax.get_xlim()[0], ax.get_ylim()[0], ax.get_zlim()[0])
    ax.set_xlim(min_lim, max_lim)
    ax.set_ylim(min_lim, max_lim)
    ax.set_zlim(min_lim, max_lim)


def main():
    # lattice = create_simple_test_lattice()
    # lattice.calculate_forces()
    # test_relaxation()
    # test_simple_triangles()
    test_triangular_lattice()


if __name__ == '__main__':
    main()
