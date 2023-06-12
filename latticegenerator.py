import gsd.hoomd
import numpy as np
from hoomd import md


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
        dots[indices[1::2, :], 0] -= d / 2
        self.dots = dots

        frame.particles.N = N
        frame.particles.position = dots
        frame.particles.types = ["A", "SW", "I"]
        frame.particles.typeid = [0] * N
        frame.configuration.box = [3 * d * nx, 3 * d * ny, 3 * d * nx, 0, 0, 0]
        self.harmonic = self._add_bonds_to_frame(spring_constant, d, inclusion_d)
        self.dihedrals = self._add_dihedrals_to_frame(dihedral_k)

    def _add_bonds_to_frame(self, spring_constant, d, inclusion_d):
        frame = self.frame
        indices = self.indices
        nx, ny = self.nx, self.ny

        frame.bonds.types = ["basic", "inclusion"]
        harmonic = md.bond.Harmonic()
        harmonic.params["basic"] = dict(k=spring_constant, r0=d)
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
            j_shift = (i - 1) % 2
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

    def set_z_to_noise(self, magnitude=0.05):
        # We add noise in the z axis, to allow out of plane perturbations
        self.dots[:, 2] += self.d * magnitude * (np.random.random(self.N) - 0.5)

    def set_z_to_sphere(self, radius=10.0):
        mean_x = np.mean(self.dots[:, 0])
        mean_y = np.mean(self.dots[:, 1])
        self.dots[:, 2] = np.sqrt(radius ** 2 - (self.dots[:, 0] - mean_x) ** 2 - (self.dots[:, 1] - mean_y) ** 2)
        if np.any(np.isnan(self.dots)):
            raise RuntimeError("Got some invalid values for the position, "
                               "this may happen if the radius of the sphere is too small")
        self.dots[:, 2] -= np.mean(self.dots[:, 2])

    def add_SW_defect(self, i, j, should_remove_dihedral=True):
        frame, nx, ny = self.frame, self.nx, self.ny
        indices = self.indices
        j_shift = (i + 1) % 2
        frame.bonds.group.remove((indices[i, j], indices[i, j + 1]))
        frame.bonds.group.append((indices[i - 1, j + j_shift], indices[i + 1, j + j_shift]))

        frame.particles.typeid[indices[i, j]] = 1
        frame.particles.typeid[indices[i, j + 1]] = 1
        frame.particles.typeid[indices[i - 1, j + j_shift]] = 1
        frame.particles.typeid[indices[i + 1, j + j_shift]] = 1

        if should_remove_dihedral:
            # We remove the dihedral that corresponds to this panel
            for d, group in enumerate(frame.dihedrals.group):
                if {group[1], group[2]} == {indices[i, j], indices[i, j + 1]}:
                    frame.dihedrals.group.remove(group)
                    frame.dihedrals.typeid.pop()
                    frame.dihedrals.N -= 1

                    break
            else:
                raise RuntimeError("Could not find the dihedral for bond "
                                   f"{indices[i, j]}, {indices[i, j + 1]}")

    def add_inclusion_defect(self, i, j):
        frame, nx, ny = self.frame, self.nx, self.ny
        indices = self.indices
        frame.particles.typeid[indices[i, j]] = 2
        for b, bond in enumerate(frame.bonds.group):
            if indices[i, j] in bond:
                frame.bonds.typeid[b] = 1


# noinspection PyPep8Naming
def calc_metric_curvature_triangular_lattice(dots, nx, ny):
    indices = np.arange(nx * ny).reshape(ny, nx)

    tx = 2 * (nx - 1)
    ty = ny - 1
    g11 = np.zeros((ty, tx))
    g12 = np.zeros((ty, tx))
    g22 = np.zeros((ty, tx))

    for i in range(ty):
        j_shift = i % 2
        for j in range(tx):
            if (i + j) % 2 == 0:
                p1 = indices[i + 1, j // 2 + 1]
                p2 = indices[i + 1, j // 2]
                p3 = indices[i, j // 2 + j_shift]
            else:
                p1 = indices[i, j // 2]
                p2 = indices[i, j // 2 + 1]
                p3 = indices[i + 1, j // 2 + 1 - j_shift]
            # print(i, j, dots[p1, :2], dots[p2, :2], dots[p3, :2])
            l1 = np.linalg.norm(dots[p1, :] - dots[p2, :])
            l2 = np.linalg.norm(dots[p2, :] - dots[p3, :])
            l3 = np.linalg.norm(dots[p3, :] - dots[p1, :])

            g11[i, j] = l1 ** 2
            g12[i, j] = -np.sqrt(3) / 3 * (l2 ** 2 - l3 ** 2)
            g22[i, j] = 1 / 3 * (2 * l2 ** 2 + 2 * l3 ** 2 - l1 ** 2)

    # g11 /= L0 ** 2
    # g12 /= L0 ** 2
    # g22 /= L0 ** 2

    g_dets = g11 * g22 - g12 ** 2
    g_inv11 = 1 / g_dets * g22
    g_inv22 = 1 / g_dets * g11
    g_inv12 = -1 / g_dets * g12

    should_assert = True
    if should_assert:
        I11 = g_inv11 * g11 + g_inv12 * g12
        I12 = g_inv11 * g12 + g_inv12 * g22
        I22 = g_inv12 * g12 + g_inv22 * g22
        assert np.all(np.isclose(I11, 1)), "There's an error with the inverse of the metric"
        assert np.all(np.isclose(I12, 0)), "There's an error with the inverse of the metric"
        assert np.all(np.isclose(I22, 1)), "There's an error with the inverse of the metric"

    # Note: I used here np.gradient which approximates the derivative using 2 adjacent values
    def g_inv(i, j):
        if i == j == 1:
            return g_inv11
        elif i == j == 2:
            return g_inv22
        else:
            return g_inv12

    def g_metric(i, j):
        if i == j == 1:
            return g11
        elif i == j == 2:
            return g22
        else:
            return g12

    g = lambda i, j, k: np.gradient(g_metric(i, j), axis=2 - k)

    gamma = lambda i, k, l: sum((1 / 2 * g_inv(i, m) * (g(m, k, l) + g(m, l, k) - g(k, l, m)))
                                for m in [1, 2])

    Ks = -1 / g11 * (
            np.gradient(gamma(2, 1, 2), axis=1, edge_order=2) - np.gradient(gamma(2, 1, 1), axis=0, edge_order=2) +
            gamma(1, 1, 2) * gamma(2, 1, 1) - gamma(1, 1, 1) * gamma(2, 1, 2) +
            gamma(2, 1, 2) * gamma(2, 1, 2) - gamma(2, 1, 1) * gamma(2, 2, 2))

    # We multiply the Gaussian curvature since 2 triangles correspond to 1
    # unit cell in the coordinates domain
    return Ks * 2, g11, g12, g22
