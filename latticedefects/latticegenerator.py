import gsd.hoomd
import numpy as np
from hoomd import md

from latticedefects.hoomdlattice import Lattice


# noinspection PyPep8Naming
class TriangularLatticeGenerator(object):
    def __init__(self, nx: int, ny: int, d=1.0, spring_constant=20.0, dihedral_k=2.0, inclusion_d=1.5):
        N = nx * ny
        frame = gsd.hoomd.Frame()
        self.N, self.nx, self.ny = N, nx, ny
        self.frame = frame
        self.d = d

        dots = np.zeros((N, 3))
        indices = np.arange(nx * ny).reshape(ny, nx)
        self.indices = indices
        dots[indices, 1] = (np.arange(ny) * d * np.sqrt(3) / 2)[:, np.newaxis]
        dots[indices, 0] = (np.arange(nx) * d)[np.newaxis, :]
        dots[indices[1::2, :], 0] += d / 2
        self.dots = dots
        self._center_XY_plane()

        frame.particles.N = N
        frame.particles.position = dots
        frame.particles.types = ["A", "SW", "I"]
        frame.particles.typeid = [0] * N
        frame.configuration.box = [3 * d * nx, 3 * d * ny, 3 * d * nx, 0, 0, 0]
        self.harmonic = self._add_bonds_to_frame(spring_constant, d, inclusion_d)
        self.dihedrals = self._add_dihedrals_to_frame(dihedral_k)

    def _duplicate_frame(self):
        new = gsd.hoomd.Frame()
        old = self.frame
        new.particles = old.particles
        new.particles.position = self.dots.copy()
        new.bonds = old.bonds
        new.dihedrals = old.dihedrals
        new.configuration.box = old.configuration.box
        return new

    def generate_lattice(self) -> Lattice:
        return Lattice(self._duplicate_frame(), self.harmonic, self.dihedrals)

    def _add_bonds_to_frame(self, spring_constant, d, inclusion_d):
        frame = self.frame
        indices = self.indices
        nx, ny = self.nx, self.ny

        frame.bonds.types = ["basic", "inclusion", "SW"]
        harmonic = md.bond.Harmonic()
        harmonic.params["basic"] = dict(k=spring_constant, r0=d)
        harmonic.params["SW"] = dict(k=spring_constant, r0=d)
        harmonic.params["inclusion"] = dict(k=spring_constant, r0=inclusion_d)

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
            j_shift = i % 2
            for j in range(nx):
                add_bond(i, j, i, j + 1)
                add_bond(i, j, i + 1, j - 1 + j_shift)
                add_bond(i, j, i + 1, j + j_shift)

        frame.bonds.group = list(bonds)
        N_bonds = len(frame.bonds.group)
        frame.bonds.N = N_bonds
        frame.bonds.typeid = [0] * N_bonds

        return harmonic

    def _add_dihedrals_to_frame(self, k):
        frame = self.frame
        nx, ny = self.nx, self.ny
        indices = self.indices

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
        dihedral_periodic.params["dihedral-basic"] = dict(k=k, d=1, n=1, phi0=0)
        frame.dihedrals.N = len(dihedrals)
        frame.dihedrals.types = ["dihedral-basic"]
        frame.dihedrals.typeid = [0] * frame.dihedrals.N
        frame.dihedrals.group = dihedrals

        return dihedral_periodic

    def set_dihedral_k(self, dihedral_k: float):
        self.dihedrals.params["dihedral-basic"]["k"] = dihedral_k

    def set_inclusion_d(self, inclusion_d: float):
        self.harmonic.params["inclusion"]["r0"] = inclusion_d

    def set_z_to_noise(self, magnitude=0.05):
        # We add noise in the z axis, to allow out of plane perturbations
        self.dots[:, 2] += self.d * magnitude * (np.random.random(self.N) - 0.5)

    def set_z_to_sphere(self, radius=None):
        if radius is None:
            radius = self.nx * 3
        mean_x = np.mean(self.dots[:, 0])
        mean_y = np.mean(self.dots[:, 1])
        self.dots[:, 2] = np.sqrt(radius ** 2 - (self.dots[:, 0] - mean_x) ** 2 - (self.dots[:, 1] - mean_y) ** 2)
        if np.any(np.isnan(self.dots)):
            raise RuntimeError("Got some invalid values for the position, "
                               "this may happen if the radius of the sphere is too small")
        self.dots[:, 2] -= np.mean(self.dots[:, 2])

    def _center_XY_plane(self):
        self.dots[:, 0] -= np.mean(self.dots[:, 0])
        self.dots[:, 1] -= np.mean(self.dots[:, 1])

    def add_SW_defect(self, i, j, should_fix_dihedral=True):
        frame, nx, ny = self.frame, self.nx, self.ny
        indices = self.indices
        j_shift = i % 2
        old_index_left = indices[i, j]
        old_index_right = indices[i, j + 1]
        new_index_down = indices[i - 1, j + j_shift]
        new_index_up = indices[i + 1, j + j_shift]
        bond_index = frame.bonds.group.index((old_index_left, old_index_right))
        frame.bonds.group[bond_index] = (new_index_down, new_index_up)
        frame.bonds.typeid[bond_index] = 2

        frame.particles.typeid[old_index_left] = 1
        frame.particles.typeid[old_index_right] = 1
        frame.particles.typeid[new_index_up] = 1
        frame.particles.typeid[new_index_down] = 1

        if should_fix_dihedral:
            # We have 5 dihedrals to fix - one to each of the edges involved
            for d, group in enumerate(frame.dihedrals.group):
                if {group[1], group[2]} == {indices[i, j], indices[i, j + 1]}:
                    frame.dihedrals.group[d] = [group[1], group[0], group[3], group[2]]

                elif {group[1], group[2], group[3]} == {old_index_left, new_index_down, old_index_right}:
                    frame.dihedrals.group[d] = [group[0], group[1], group[2], new_index_up]

                elif {group[1], group[2], group[3]} == {old_index_left, new_index_up, old_index_right}:
                    frame.dihedrals.group[d] = [group[0], group[1], group[2], new_index_down]

                elif {group[0], group[1], group[2]} == {old_index_left, new_index_up, old_index_right}:
                    frame.dihedrals.group[d] = [new_index_down, group[1], group[2], group[3]]

                elif {group[0], group[1], group[2]} == {old_index_left, new_index_down, old_index_right}:
                    frame.dihedrals.group[d] = [new_index_up, group[1], group[2], group[3]]

    def add_inclusion_defect(self, i, j):
        frame, nx, ny = self.frame, self.nx, self.ny
        indices = self.indices
        frame.particles.typeid[indices[i, j]] = 2
        for b, bond in enumerate(frame.bonds.group):
            if indices[i, j] in bond:
                frame.bonds.typeid[b] = 1
