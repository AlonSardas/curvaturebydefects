"""
Use hoomd package to simulate springs and masses

See doc at:
https://hoomd-blue.readthedocs.io/en/latest/module-md-minimize.html
https://hoomd-blue.readthedocs.io/en/latest/howto/molecular.html
"""
import gsd.hoomd
import hoomd
import numpy as np
from hoomd import md
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import plotutils
from latticegenerator import TriangularLatticeGenerator, calc_metric_curvature_triangular_lattice


def harmonic_example():
    frame = gsd.hoomd.Frame()

    # Place a polymer in the box.
    frame.particles.N = 5
    frame.particles.position = [[-2, 0, 0], [-1, 0, 0], [0, 0, 0], [1, 0, 0], [2, 0, 0]]
    frame.particles.types = ["A"]
    frame.particles.typeid = [0] * 5
    frame.configuration.box = [20, 20, 20, 0, 0, 0]

    # Connect particles with bonds.
    frame.bonds.N = 4
    frame.bonds.types = ["A-A"]
    frame.bonds.typeid = [0] * 4
    frame.bonds.group = [[0, 1], [1, 2], [2, 3], [3, 4]]

    with gsd.hoomd.open(name="molecular.gsd", mode="xb") as f:
        f.append(frame)

    # Apply the harmonic potential on the bonds.
    harmonic = hoomd.md.bond.Harmonic()
    harmonic.params["A-A"] = dict(k=100, r0=1.0)

    # Perform the MD simulation.
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_gsd(filename="molecular.gsd")
    langevin = hoomd.md.methods.Langevin(filter=hoomd.filter.All(), kT=1.0)
    integrator = hoomd.md.Integrator(dt=0.005, methods=[langevin], forces=[harmonic])
    gsd_writer = hoomd.write.GSD(
        filename="molecular_trajectory.gsd",
        trigger=hoomd.trigger.Periodic(1000),
        mode="xb",
    )
    sim.operations.integrator = integrator
    sim.operations.writers.append(gsd_writer)
    sim.run(10e3)


def test_simple():
    frame = gsd.hoomd.Frame()
    N = 4
    frame.particles.N = N

    dots = np.zeros((4, 3))
    dots[0, :2] = [-0.5, 0]
    dots[1, :2] = [0, 0.7]
    dots[2, :2] = [0.5, 0]
    dots[3, :2] = [0, -0.7]
    dots[:, 2] += 0.1 * (np.random.random(N) - 0.5)

    frame.particles.position = dots
    frame.particles.types = ["A"]
    frame.particles.typeid = [0] * N
    frame.configuration.box = [20, 20, 20, 0, 0, 0]

    frame.bonds.group = [[0, 1], [1, 2], [2, 3], [3, 0], [1, 3], [0, 2]]
    N_bonds = len(frame.bonds.group)
    frame.bonds.N = N_bonds
    frame.bonds.types = ["A-A"]
    frame.bonds.typeid = [0] * N_bonds
    harmonic = md.bond.Harmonic()
    harmonic.params["A-A"] = dict(k=20, r0=1.0)

    dihedral_periodic = md.dihedral.Periodic()
    dihedral_periodic.params["A-A-A-A"] = dict(k=0.5, d=1, n=1, phi0=0)
    frame.dihedrals.N = 1
    frame.dihedrals.types = ["A-A-A-A"]
    frame.dihedrals.typeid = [0] * frame.dihedrals.N
    frame.dihedrals.group = [[0, 1, 3, 2]]

    with gsd.hoomd.open(name="molecular.gsd", mode="w") as f:
        f.append(frame)

    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_gsd(filename="molecular.gsd")

    # Use FIRE as integrator
    fire = md.minimize.FIRE(dt=0.05, force_tol=1e-5, angmom_tol=1e-2, energy_tol=1e-10)
    fire.methods.append(md.methods.NVE(hoomd.filter.All()))
    fire.forces.append(harmonic)
    fire.forces.append(dihedral_periodic)
    sim.operations.integrator = fire

    while not (fire.converged):
        print("here")
        sim.run(100)

    snapshot = sim.state.get_snapshot()

    dots = snapshot.particles.position

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    ax.plot(dots[:, 0], dots[:, 1], dots[:, 2])
    plotutils.set_axis_scaled(ax)
    plt.show()


def generate_triangular_lattice(
        nx: int, ny: int, d=1.0, spring_constant=20.0, dihedral_k=2.0
):
    N = nx * ny
    dots = np.zeros((N, 3))
    indices = np.arange(nx * ny).reshape(ny, nx)
    dots[indices, 1] = (np.arange(ny) * d * np.sqrt(3) / 2)[:, np.newaxis]
    dots[indices, 0] = (np.arange(nx) * d)[np.newaxis, :]
    dots[indices[1::2, :], 0] -= d / 2

    # We add noise in the z axis, to allow out of plane perturbations
    dots[:, 2] += d * 0.05 * (np.random.random(N) - 0.5)

    frame = gsd.hoomd.Frame()

    # Place a polymer in the box.
    frame.particles.N = N
    frame.particles.position = dots
    frame.particles.types = ["A"]
    frame.particles.typeid = [0] * N
    frame.configuration.box = [3 * d * nx, 3 * d * ny, 3 * d * nx, 0, 0, 0]

    harmonic = add_bonds_to_frame(nx, ny, frame, spring_constant, d)
    dihedrals = add_dihedrals_to_frame(nx, ny, frame, dihedral_k)

    return frame, harmonic, dihedrals


def add_bonds_to_frame(nx, ny, frame, spring_constant, d):
    indices = np.arange(nx * ny).reshape(ny, nx)

    frame.bonds.types = ["A-A"]

    bonds = set()

    def add_bond(i1, j1, i2, j2):
        pos = np.array([i1, j1, i2, j2])
        if any(pos < 0):
            return
        if i1 >= ny or i2 >= ny or j1 >= nx or j2 >= nx:
            return
        p1 = indices[i1, j1]
        p2 = indices[i2, j2]
        bonds.add((p1, p2))

    for i in range(ny):
        j_shift = (i - 1) % 2
        for j in range(nx):
            add_bond(i, j, i, j + 1)
            add_bond(i, j, i + 1, j - 1 + j_shift)
            add_bond(i, j, i + 1, j + j_shift)

    frame.bonds.group = list(bonds)
    N_bonds = len(frame.bonds.group)
    frame.bonds.N = N_bonds
    frame.bonds.typeid = [0] * N_bonds

    harmonic = md.bond.Harmonic()
    harmonic.params["A-A"] = dict(k=spring_constant, r0=d * 1.5)
    return harmonic


def add_dihedrals_to_frame(nx, ny, frame, k):
    indices = np.arange(nx * ny).reshape(ny, nx)

    # There are 3 'types' of dihedrals for each triangle. Each triangle has 3
    # vertices with other triangles and we add the dihedral forces to each one
    dihedrals = []

    for i in range(nx * ny - nx):
        # Even row
        if (i // nx) % 2 == 0 and i % nx != nx - 1:
            dihedrals.append((i, i + nx, i + 1, i + nx + 1))
        # Odd row
        if (i // nx) % 2 == 1 and i % nx != nx - 1 and i % nx != nx - 2:
            dihedrals.append((i, i + nx + 1, i + 1, i + nx + 2))

    for i in range(nx, nx * ny):
        # Even row
        if (i // nx) % 2 == 0 and i % nx != nx - 1:
            dihedrals.append((i, i - nx, i + 1, i - nx + 1))
        # Odd row
        if (i // nx) % 2 == 1 and i % nx != nx - 1 and i % nx != nx - 2:
            dihedrals.append((i, i - nx + 1, i + 1, i - nx + 2))

    for i in range(nx * 2, nx * ny):
        # Even row
        if (i // nx) % 2 == 0 and i % nx != 0:
            dihedrals.append((i, i - nx - 1, i - nx, i - 2 * nx))
        # Odd row
        if (i // nx) % 2 == 1 and i % nx != nx - 1:
            dihedrals.append((i, i - nx, i - nx + 1, i - 2 * nx))

    dihedral_periodic = md.dihedral.Periodic()
    dihedral_periodic.params["A-A-A-A"] = dict(k=k, d=1, n=1, phi0=0)
    frame.dihedrals.N = len(dihedrals)
    frame.dihedrals.types = ["A-A-A-A"]
    frame.dihedrals.typeid = [0] * frame.dihedrals.N
    frame.dihedrals.group = dihedrals

    print(len(dihedrals))
    print(dihedrals)

    return dihedral_periodic


def add_SW_defect(frame, nx, ny, i, j):
    indices = np.arange(nx * ny).reshape(ny, nx)
    j_shift = (i + 1) % 2
    print(frame.bonds.group)
    frame.bonds.group.remove((indices[i, j], indices[i, j + 1]))
    frame.bonds.group.append((indices[i - 1, j + j_shift], indices[i + 1, j + j_shift]))
    print(frame.bonds.group)


def test_triangular_lattice():
    nx, ny = 14, 14
    L0 = 0.5
    lattice = TriangularLatticeGenerator(nx, ny, d=L0)
    R = 7.0
    lattice.set_z_to_sphere(R)
    print(1/R**2)
    # lattice.add_SW_defect(2, 1)
    # lattice.add_inclusion_defect(2, 2)
    # lattice.add_SW_defect(10, 5)
    # lattice.add_SW_defect(1, 2)

    frame, harmonic, dihedrals = lattice.frame, lattice.harmonic, lattice.dihedrals
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(frame)

    # Use FIRE as integrator
    fire = md.minimize.FIRE(dt=0.05, force_tol=1e-3, angmom_tol=1e-2, energy_tol=1e-10)
    fire.methods.append(md.methods.NVE(hoomd.filter.All()))
    # fire.methods.append(md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = fire
    fire.forces.append(harmonic)
    fire.forces.append(dihedrals)

    snapshot = sim.state.get_snapshot()

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    plot_dots(ax, snapshot)
    plot_bonds(ax, snapshot)
    plotutils.set_axis_scaled(ax)
    plotutils.set_3D_labels(ax)
    # ax.set_zlim(-0.5, 0.5)
    # plt.show()

    fig, axes = plt.subplots(2, 2)
    Ks, g11, g12, g22 = calc_metric_curvature_triangular_lattice(snapshot.particles.position, nx, ny, L0)
    print(Ks.shape)
    plotutils.imshow_with_colorbar(fig, axes[0, 0], g11, "g11")
    plotutils.imshow_with_colorbar(fig, axes[0, 1], g22, "g22")
    plotutils.imshow_with_colorbar(fig, axes[1, 0], g12, "g12")
    plotutils.imshow_with_colorbar(fig, axes[1, 1], Ks[3:-3, 3:-3], "K")

    plt.show()

    while not fire.converged:
        print("here")
        sim.run(100)

    snapshot = sim.state.get_snapshot()

    dots = snapshot.particles.position
    print(dots[0, :], dots[1, :])
    print(np.sqrt(np.sum((dots[0, :] - dots[1, :]) ** 2)))

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")
    plot_dots(ax, snapshot)
    plot_bonds(ax, snapshot)
    plotutils.set_axis_scaled(ax)
    # ax.set_zlim(-10, 10)
    plt.show()


def plot_dots(ax: Axes3D, snapshot: hoomd.snapshot.Snapshot):
    dots = snapshot.particles.position
    ax.plot(dots[:, 0], dots[:, 1], dots[:, 2], ".")


def plot_bonds(ax: Axes3D, snapshot: hoomd.snapshot.Snapshot):
    dots = snapshot.particles.position
    bonds = snapshot.bonds.group
    for bond in bonds:
        p1, p2 = bond
        ax.plot(
            [dots[p1, 0], dots[p2, 0]],
            [dots[p1, 1], dots[p2, 1]],
            [dots[p1, 2], dots[p2, 2]],
            "-r",
        )


def main():
    test_triangular_lattice()


if __name__ == "__main__":
    main()
