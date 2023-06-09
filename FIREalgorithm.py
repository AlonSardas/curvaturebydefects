"""
Implements the FIRE (Fast Inertial Relaxation Engine) algorithm.

Here are some resources about it:
https://www.math.uni-bielefeld.de/~gaehler/papers/fire.pdf
http://users.jyu.fi/~pekkosk/resources/pdf/FIRE.pdf
https://doi.org/10.1016/j.commatsci.2020.109584
https://nanosurf.fzu.cz/wiki/doku.php?id=fire_minimization
"""
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import springlattice


# noinspection PyPep8Naming
def FIRE(lattice: springlattice.SpringLattice, dt=0.01, dt_max=0.1, convergence_force_threshold: float = 0.01):
    N_min = 5
    f_inc = 1.1
    f_dec = 0.5
    alpha_start = 0.1
    f_alpha = 0.99

    alpha = alpha_start

    mass = lattice.mass
    print(mass)
    v = np.zeros((lattice.N, 3))
    N_max = 1000
    cut = 0
    for i in range(N_max):
        F = lattice.calculate_forces()
        F_norm = np.sqrt(np.sum(F ** 2))
        if F_norm < convergence_force_threshold * lattice.N:
            print(f"relaxed after {i} iterations")
            return

        v += F / mass * dt
        lattice.dots += v * dt
        lattice.dots -= np.mean(lattice.dots, axis=0)[np.newaxis, :]

        v_norm = np.sqrt(np.sum(v ** 2))

        # Note: P is not a vector, it's a number
        P = np.sum(F * v)

        # print(dt, alpha, P, F_norm)

        v = (1 - alpha) * v + alpha * F / F_norm * v_norm

        if P > 0 and i - cut > N_min:
            dt = min(dt * f_inc, dt_max)
            alpha = alpha * f_alpha

        if P <= 0:
            cut = i
            dt = dt * f_dec
            v *= 0
            alpha = alpha_start

        # print(np.sum(F * v, axis=1) < 0)
        # Note this just make sense, I didn't find it written explicitly in the relaxation
        # algorithm, but it seems to be good
        v[np.sum(F * v, axis=1) < 0, :] = 0

        should_plot = True
        if should_plot and i % 10 == 0:
            print(lattice.calculate_forces()[0,:])
            fig: Figure = plt.figure()
            ax: Axes3D = fig.add_subplot(111, projection='3d')
            ax.scatter3D(lattice.dots[:, 0], lattice.dots[:, 1], lattice.dots[:, 2])
            ax.set_xlim(-5, 5)
            ax.set_ylim(-5, 5)
            ax.set_zlim(-5, 5)
            plt.show()

    else:
        print(f"Warning: No convergence after {N_max} iterations!")


# noinspection PyPep8Naming
def FIRE_bad(lattice: springlattice.SpringLattice, dt=0.01, convergence_force_threshold: float = 0.01):
    N_min = 5
    f_inc = 1.1
    f_dec = 0.5
    alpha_start = 0.1
    f_alpha = 0.99
    dt_max = 0.2

    alpha = alpha_start

    mass = lattice.mass
    print(mass)
    v = np.zeros((lattice.N, 3))
    N_max = 1000
    cut = 0
    for i in range(N_max):
        F = lattice.calculate_forces()
        F_norm = np.sqrt(np.sum(F ** 2, axis=1))
        if np.all(F_norm < convergence_force_threshold):
            print(f"relaxed after {i} iterations")
            return

        v += F / mass * dt
        lattice.dots += v * dt
        lattice.dots -= np.mean(lattice.dots, axis=0)[np.newaxis, :]

        v_norm = np.sqrt(np.sum(v ** 2, axis=1))

        # Note: P is not a vector, it's a number
        P = np.sum(F * v)

        print(dt, alpha, P, F_norm)

        v = (1 - alpha) * v + alpha * F / F_norm[:, np.newaxis] * v_norm[:, np.newaxis]

        if P > 0 and i - cut > N_min:
            dt = min(dt * f_inc, dt_max)
            alpha = alpha * f_alpha

        if P <= 0:
            cut = i
            dt = dt * f_dec
            v *= 0
            alpha = alpha_start

        # fig: Figure = plt.figure()
        # ax: Axes3D = fig.add_subplot(111, projection='3d')
        # ax.scatter3D(lattice.dots[:, 0], lattice.dots[:, 1], lattice.dots[:, 2])
        # ax.set_xlim(-5, 5)
        # ax.set_ylim(-5, 5)
        # ax.set_zlim(-5, 5)
        # plt.show()

    else:
        print(f"Warning: No convergence after {N_max} iterations!")
