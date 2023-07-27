"""
Use HOOMD-blue package to simulate springs and masses.
The relaxation is done using FIRE algorithm.

See doc at:
https://hoomd-blue.readthedocs.io/en/latest/module-md-minimize.html
https://hoomd-blue.readthedocs.io/en/latest/howto/molecular.html
"""
import gsd.hoomd
import hoomd
import numpy as np
from hoomd import md
from mpl_toolkits.mplot3d import Axes3D
from typing import Callable, Optional, Union


class Lattice(object):
    def __init__(self, frame, harmonic, dihedrals):
        frame.validate()
        self.frame = frame
        self.harmonic = harmonic
        self.dihedrals = dihedrals

        sim = hoomd.Simulation(device=hoomd.device.CPU(), seed=1)
        sim.create_state_from_snapshot(frame)
        self.sim = sim

    def log_trajectory(self, filename, dt_steps):
        logger = hoomd.logging.Logger()
        logger += self.harmonic
        logger += self.dihedrals
        trigger = hoomd.trigger.Or(
            [hoomd.trigger.Periodic(dt_steps)]
        )
        gsd_writer = hoomd.write.GSD(filename=filename,
                                     trigger=trigger,
                                     mode='wb', logger=logger)
        self.sim.operations.writers.append(gsd_writer)

    def do_relaxation(self, dt=0.05, force_tol=1e-6, angmom_tol=1e-2,
                      energy_tol=1e-10, iteration_time=1000,
                      pre_iteration_hook: Optional[Callable] = None) -> hoomd.Snapshot:
        sim = self.sim
        harmonic = self.harmonic
        dihedrals = self.dihedrals

        fire = md.minimize.FIRE(dt, force_tol, angmom_tol, energy_tol)
        fire.methods.append(md.methods.ConstantVolume(hoomd.filter.All()))
        sim.operations.integrator = fire
        fire.forces.append(harmonic)
        fire.forces.append(dihedrals)

        sim.run(0, write_at_start=True)  # This is necessary for calculating the energy in the first step

        while not fire.converged:
            E_stretching = harmonic.energy
            E_bending = dihedrals.energy
            print("Fire iteration. "
                  f"E-stretch: {E_stretching:.6f}, E-bend: {E_bending:.6f}, "
                  f"E-total: {E_stretching+E_bending:.6f}")
            if pre_iteration_hook is not None:
                pre_iteration_hook()
            sim.run(iteration_time)

        return sim.state.get_snapshot()

    def save_frame(self, filepath):
        logger = hoomd.logging.Logger()
        logger += self.harmonic
        logger += self.dihedrals
        hoomd.write.GSD.write(self.sim.state, filepath, logger=logger)

    def get_dots(self)->np.ndarray:
        snapshot = self.sim.state.get_snapshot()
        return snapshot.particles.position

    def plot_dots(self, ax: Axes3D):
        snapshot = self.sim.state.get_snapshot()
        dots = snapshot.particles.position
        ax.plot(dots[:, 0], dots[:, 1], dots[:, 2], ".", color='C0', alpha=0.8)

    def plot_bonds(self, ax: Axes3D):
        plot_bonds(ax, self.sim.state.get_snapshot())


def do_relaxation(frame, harmonic, dihedrals, dt=0.05, force_tol=1e-5, angmom_tol=1e-2,
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


def plot_dots(ax: Axes3D,
              snapshot: Union[hoomd.Snapshot, gsd.hoomd.Frame]):
    dots = snapshot.particles.position
    return ax.plot(dots[:, 0], dots[:, 1], dots[:, 2], ".")[0]


def plot_bonds(ax: Axes3D,
               snapshot: Union[hoomd.Snapshot, gsd.hoomd.Frame]):
    dots = snapshot.particles.position
    bonds = snapshot.bonds.group

    for i, bond in enumerate(bonds):
        p1, p2 = bond

        color = 'C9'
        if snapshot.bonds.typeid[i] == 2:
            color = 'r'

        ax.plot(
            [dots[p1, 0], dots[p2, 0]],
            [dots[p1, 1], dots[p2, 1]],
            [dots[p1, 2], dots[p2, 2]],
            "-", color=color, alpha=0.8
        )
