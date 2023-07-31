import os
from matplotlib import pyplot as plt

from latticedefects import plots, geometry
from latticedefects.trajectory import load_trajectory
from latticedefects.utils import plotutils


def test_curvature_by_interpolation():
    # Load the data
    FIGURE_PATH = os.path.join(plots.BASE_PATH, "MD-simulations")
    folder = os.path.join(FIGURE_PATH, 'sphere-small-lattice')
    trajectory_file = os.path.join(folder, f'trajectory-single-SW.gsd')

    frames = load_trajectory(trajectory_file)
    frame = frames[-1]
    dots = frame.get_dots()

    nx, ny = 50, 60
    if False:
        Ks, _, _, _ = geometry.calc_metric_curvature_triangular_lattice(dots, nx, ny)
        fig, ax = plt.subplots()
        im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
        im.set_clim(-0.0001, 0.0001)
        plt.show()
    
    Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots)
    fig, ax = plt.subplots()
    im = plotutils.imshow_with_colorbar(fig, ax, Ks, "K")
    # im.set_clim(-0.0001, 0.0001)
    print(Ks[2:-2, 2:-2].sum())
    plt.show()
    

def main():
    test_curvature_by_interpolation()


if __name__ == '__main__':
    main()
