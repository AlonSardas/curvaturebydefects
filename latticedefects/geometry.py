from typing import Tuple

import numpy as np
from scipy.interpolate.interpnd import LinearNDInterpolator, CloughTocher2DInterpolator


def calc_metric_curvature_triangular_lattice(dots, nx, ny):
    """
    Calculate the Gaussian curvature assuming the lattice is a regular
    triangular lattice - this means that we assume there are no defects
    """
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


def calculate_curvatures_by_interpolation(
        dots, x_samples: int = 120, y_samples: int = 120) -> Tuple[np.ndarray, np.ndarray]:
    # interp = LinearNDInterpolator(dots[:, :2], dots[:, 2])
    interp = CloughTocher2DInterpolator(dots[:, :2], dots[:, 2])
    xs = np.linspace(np.min(dots[:, 0]), np.max(dots[:, 0]), x_samples)
    ys = np.linspace(np.min(dots[:, 1]), np.max(dots[:, 1]), y_samples)
    xs += (xs[1] - xs[0]) / 2
    ys += (ys[1] - ys[0]) / 2
    Xs, Ys = np.meshgrid(xs, ys)
    Zs = interp(Xs, Ys)

    quad_dots = np.array([Xs.flatten(), Ys.flatten(), Zs.flatten()])
    import origami.quadranglearray
    from origami.origamimetric import OrigamiGeometry
    quads = origami.quadranglearray.QuadrangleArray(quad_dots, y_samples, x_samples)
    geom = OrigamiGeometry(quads)
    Ks, Hs = geom.get_curvatures_by_shape_operator()
    return Ks, Hs
