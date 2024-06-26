from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from typing import Optional

from latticedefects.hoomdlattice import Lattice
from latticedefects.utils import plotutils


def create_fig_and_plot_dots(lattice: Lattice, azim, elev):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d", azim=azim, elev=elev)
    lattice.plot_dots(ax)
    plotutils.set_3D_labels(ax)
    # plotutils.set_axis_scaled(ax)
    return fig, ax


def plot_flat_and_save(lattice_gen, filename,
                       box_size: Optional[float] = None, plot='bonds', with_axes=False):
    fig = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection='3d', azim=-90, elev=90)
    lattice = lattice_gen.generate_lattice()
    if plot == 'bonds' or plot == 'both':
        lattice.plot_bonds(ax)
    if plot == 'dots' or plot == 'both':
        lattice.plot_dots(ax)
    if box_size:
        ax.set_xlim(-box_size, box_size)
        ax.set_ylim(-box_size, box_size)
    if not with_axes:
        ax.set_axis_off()
    fig.savefig(filename + '.svg')
    fig.savefig(filename + '.pdf')
    fig.savefig(filename + '.png')
    lattice.save_frame(filename + '.gsd')
