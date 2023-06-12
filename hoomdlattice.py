"""
Use HOOMD-blue package to simulate springs and masses.
The relaxation is done using FIRE algorithm.

See doc at:
https://hoomd-blue.readthedocs.io/en/latest/module-md-minimize.html
https://hoomd-blue.readthedocs.io/en/latest/howto/molecular.html
"""
import hoomd
from hoomd import md
from mpl_toolkits.mplot3d import Axes3D


def do_relaxation(frame, harmonic, dihedrals, dt=0.05, force_tol=1e-3, angmom_tol=1e-2,
                  energy_tol=1e-10) -> hoomd.Snapshot:
    sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
    sim.create_state_from_snapshot(frame)

    # Use FIRE as integrator
    fire = md.minimize.FIRE(dt, force_tol, angmom_tol, energy_tol)
    fire.methods.append(md.methods.ConstantVolume(hoomd.filter.All()))
    sim.operations.integrator = fire
    fire.forces.append(harmonic)
    fire.forces.append(dihedrals)

    while not fire.converged:
        print("Fire iteration")
        sim.run(100)

    return sim.state.get_snapshot()


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
