import os.path

import argparse
import gsd.hoomd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

from latticedefects.hoomdlattice import load_trajectory
from latticedefects.plots import simulations
from latticedefects.utils import plotutils


def list_func(args):
    filename = args.file
    if not os.path.exists(filename):
        raise RuntimeError(f"Couldn't find the file: {filename}")
    log = gsd.hoomd.read_log(filename)
    steps = log['configuration/step']
    print("The available timesteps are:")
    print(steps)


def plot_func(args):
    filename = args.file
    if not os.path.exists(filename):
        raise RuntimeError(f"Couldn't find the file: {filename}")
    traj = load_trajectory(filename)
    frame = traj[args.i]

    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    if args.bonds:
        frame.plot_bonds(ax)
    else:
        frame.plot_dots(ax)
    plotutils.set_3D_labels(ax)

    if args.axis == 'equal':
        ax.set_aspect('equal')

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Plot the frames from a GSD file")

    parser.add_argument('--file', help='path to GSD file', required=True)

    subparsers = parser.add_subparsers(required=True)

    list_parser = subparsers.add_parser('list', help='list the available frames in the trajectory')
    list_parser.set_defaults(func=list_func)

    plot_parser = subparsers.add_parser('plot', help='plot the frame')
    plot_parser.add_argument('-i', help='The index of the frame. Defaults to last frame', default=-1, type=int)
    plot_parser.add_argument('--axis', default='equal', choices=['equal', 'auto'])
    plot_parser.add_argument('--bonds', action='store_true')
    plot_parser.set_defaults(func=plot_func)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
