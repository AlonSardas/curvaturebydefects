import os.path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure

FIGURES_DIR = '../Figures'


def sphere_by_isotropic_inclusions():
    area_ratio = 2
    C2 = 1
    Kr = 0.01

    # density_func = lambda x, y: 1 - (x ** 2 + y ** 2)
    density_func = lambda r: 1 / (1 - area_ratio) * (Kr * r ** 2) + C2

    fig, ax = plt.subplots()
    # density_plot(ax, x_lim, y_lim, density_func)
    concentration_plot_with_radial_symmetry(ax, [0.4, 11], density_func)
    ax: Axes = ax
    ax.set_xlabel('x [a.u.]')
    ax.set_ylabel('y [a.u.]')
    fig: Figure = fig
    fig.savefig(os.path.join(FIGURES_DIR, 'SphericalIsotropicDensity.png'))


def cone_by_isotropic_inclusions():
    area_ratio = 2
    r0 = 30
    q = 0.1

    density_func = lambda r: 1 / (1 - area_ratio) * (2 * q / np.pi * np.log(r / r0))

    fig, ax = _create_fig_ax()
    # density_plot(ax, x_lim, y_lim, density_func)
    concentration_plot_with_radial_symmetry(ax, [0.7, r0 + 0.3], density_func)
    ax.set_aspect('equal')
    _set_arbitrary_units(ax)
    fig.savefig(os.path.join(FIGURES_DIR, 'ConeIsotropicDensity.png'))


def sphere_by_traceless_radial():
    a = 1 / 4
    K = 0.01
    sa = np.sqrt(a)

    q = lambda r: 1 / 8 * K * r ** 2
    func = lambda r: -2 * sa * q(r) / ((sa - 1) * (sa + 1 - 2 * q(r)))

    fig, ax = _create_fig_ax()
    concentration_plot_with_radial_symmetry(ax, [5, 16], func)
    ax.set_aspect('equal')
    _set_arbitrary_units(ax)
    fig.savefig(os.path.join(FIGURES_DIR, 'SphereTracelessRadial.png'))


def sphere_by_dislocations():
    fig, ax = _create_fig_ax()

    K = 1
    density_func = lambda r: 1 / 2 * K * r

    rs = np.linspace(1, 7, 9)
    _plot_radial_dislocation_density(ax, rs, density_func)

    ax.set_aspect('equal')
    _set_arbitrary_units(ax)

    fig.savefig(os.path.join(FIGURES_DIR, 'SphereDislocations.png'))


def cone_by_dislocations():
    fig, ax = _create_fig_ax()

    q = 1
    density_func = lambda r: q / (2 * np.pi * r)

    rs = np.linspace(3, 17, 8) * 10
    _plot_radial_dislocation_density(ax, rs, density_func, )

    ax.set_aspect('equal')
    _set_arbitrary_units(ax)

    fig.savefig(os.path.join(FIGURES_DIR, 'ConeDislocations.png'))


def _plot_radial_dislocation_density(ax: Axes, rs, density_func: Callable):
    dislocation_line_length = rs.max() * 0.02
    print(dislocation_line_length)
    for i in range(len(rs) - 1):
        r = rs[i]

        density = density_func(rs[i])
        if density < 0:
            raise RuntimeError(f"Found invalid density {density} for r={r}")

        area = np.pi * rs[i + 1] ** 2 - np.pi * r ** 2
        dots = int(np.round(density * area))
        print(density, area, dots)
        angles = np.pi / 4 * i + np.linspace(0, 2 * np.pi, dots)
        for angle in angles:
            x = rs[i] * np.cos(angle)
            y = rs[i] * np.sin(angle)
            _plot_dislocation(ax, x, y, angle, dislocation_line_length)


def sphere_by_traceless_horizontal():
    fig, ax = _create_fig_ax()
    a = 1.2
    sa = np.sqrt(a)
    K = 0.1

    # This function is without approximation. We used the approximated function below
    # func = lambda x, y: sa * K * y ** 2 / ((sa - 1) * (sa + 1 + K * y ** 2))
    func = lambda x, y: sa * K * y ** 2 / (a - 1)
    size = 1
    concentration_plot(ax, [-size, size + 0.01], [-size, size + 0.01], 0.1, 4, func)
    _set_arbitrary_units(ax)
    fig.savefig(os.path.join(FIGURES_DIR, 'SphereTracelessHorizontal.png'))


def concentration_plot(ax: Axes, x_lim, y_lim, box_edge, defects_per_box_edge, concentration_func):
    xs = np.arange(x_lim[0], x_lim[1], box_edge)
    ys = np.arange(y_lim[0], y_lim[1], box_edge)
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            x = (xs[j + 1] + xs[j]) / 2
            y = (ys[i + 1] + ys[i]) / 2

            concentration = concentration_func(x, y)
            if concentration < 0 or concentration > 1:
                raise RuntimeError(
                    f"Found invalid concentration {concentration} at point ({x}, {y})")

            defects = int(np.round(concentration * defects_per_box_edge ** 2))
            print(f"Found {defects} defects")
            for d in range(defects):
                defect_x = np.random.choice(np.linspace(xs[j], xs[j + 1], defects_per_box_edge))
                defect_y = np.random.choice(np.linspace(ys[i], ys[i + 1], defects_per_box_edge))

                ax.plot(defect_x, defect_y, '.b', markersize=9)


def concentration_plot_with_radial_symmetry(ax: Axes, r_lim, concentration_func):
    rs = np.linspace(r_lim[0], r_lim[1], 10)
    for i in range(len(rs) - 1):
        r = rs[i]

        concentration = concentration_func(rs[i])
        if concentration < 0 or concentration > 1:
            raise RuntimeError(
                f"Found invalid concentration {concentration} for r={r}")

        area = np.pi * rs[i + 1] ** 2 - np.pi * r ** 2
        dots = int(np.round(concentration * area))
        print(concentration, area, dots)
        angles = np.pi / 4 * i + np.linspace(0, 2 * np.pi, dots)
        xs = rs[i] * np.cos(angles)
        ys = rs[i] * np.sin(angles)
        ax.plot(xs, ys, '.b', markersize=9)


def density_plot_by_random_scatter(ax, x_lim, y_lim, density_func: Callable, dots: int = 1000):
    for i in range(dots):
        x = np.random.uniform(x_lim[0], x_lim[1])
        y = np.random.uniform(y_lim[0], y_lim[1])

        density = density_func(x, y)
        if density < 0 or density > 1:
            raise RuntimeError(f"Found invalid density {density}")
        should_plot = np.random.random() < density
        if should_plot:
            ax.plot(x, y, '.b')


def _plot_dislocation(ax: Axes, x, y, angle, line_length: float = 0.2):
    """
    Plot a sign for the dislocation with direction
    """
    # line_length = 0.2
    line_factor = 0.75
    x0 = x - line_length * line_factor * np.cos(angle - np.pi / 2)
    y0 = y - line_length * line_factor * np.sin(angle - np.pi / 2)
    x1 = x + line_length * line_factor * np.cos(angle - np.pi / 2)
    y1 = y + line_length * line_factor * np.sin(angle - np.pi / 2)
    ax.plot([x0, x1], [y0, y1], '-b')

    x0 = x
    y0 = y
    x1 = x + line_length * np.cos(angle)
    y1 = y + line_length * np.sin(angle)
    ax.plot([x0, x1], [y0, y1], '-b')


def _set_arbitrary_units(ax: Axes):
    ax.set_xlabel('x [a.u.]')
    ax.set_ylabel('y [a.u.]')


def _create_fig_ax() -> (Figure, Axes):
    fig: Figure = plt.figure()
    ax: Axes = fig.subplots()
    return fig, ax


def main():
    # sphere_by_isotropic_inclusions()
    # cone_by_isotropic_inclusions()
    # sphere_by_dislocations()
    sphere_by_traceless_horizontal()
    # cone_by_dislocations()
    # sphere_by_traceless_radial()
    plt.show()


if __name__ == '__main__':
    main()
