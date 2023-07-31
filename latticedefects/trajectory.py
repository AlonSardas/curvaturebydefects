import gsd
import matplotlib
import os
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d import Axes3D
from typing import Sequence

import latticedefects
from latticedefects import geometry
from latticedefects.hoomdlattice import plot_bonds, plot_dots
from latticedefects.plots.latticeplotutils import create_fig_and_plot_dots
from latticedefects.utils import plotutils


class Frame(object):
    def __init__(self, frame, step, harmonic_energy, dihedrals_energy):
        self.frame = frame
        self.timestep = step
        self.harmonic_energy = harmonic_energy
        self.dihedrals_energy = dihedrals_energy

    def get_dots(self):
        return self.frame.particles.position

    def plot_dots(self, ax: Axes3D):
        dots = self.frame.particles.position
        ax.plot(dots[:, 0], dots[:, 1], dots[:, 2], ".", color='C0', alpha=0.8)

    def plot_bonds(self, ax: Axes3D):
        plot_bonds(ax, self.frame)


def load_trajectory(filepath: str) -> Sequence[Frame]:
    log = gsd.hoomd.read_log(filepath)
    steps = log['configuration/step']
    if 'log/md/bond/Harmonic/energy' in log:
        harmonic_energy = log['log/md/bond/Harmonic/energy']
    else:
        harmonic_energy = [None]
    if 'log/md/dihedral/Periodic/energy' in log:
        dihedrals_energy = log['log/md/dihedral/Periodic/energy']
    else:
        dihedrals_energy = [None]

    frames = []
    with gsd.hoomd.open(filepath) as f:
        for i, frame in enumerate(f):
            frames.append(Frame(
                frame, steps[i], harmonic_energy[i], dihedrals_energy[i]))
        return frames


def plot_all_frames(frames: Sequence[Frame]):
    fig: Figure = plt.figure()
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    dots_plot = plot_dots(ax, frames[0].frame)

    # Make a horizontal slider
    frames_slider_ax = fig.add_axes([0.2, 0.05, 0.5, 0.03])
    curvature_button_ax = fig.add_axes([0.75, 0.05, 0.15, 0.08])

    frames_slider = matplotlib.widgets.Slider(
        ax=frames_slider_ax,
        label='frame index',
        valmin=0,
        valmax=len(frames) - 1,
        valstep=1,
        valinit=0,
    )

    def update_frame(frame_index):
        frame = frames[frame_index]
        dots = frame.frame.particles.position
        dots_plot.set_data_3d(dots[:, 0], dots[:, 1], dots[:, 2])
        ax.auto_scale_xyz(dots[:, 0], dots[:, 1], dots[:, 2])
        fig.canvas.draw_idle()

    frames_slider.on_changed(update_frame)

    button = Button(curvature_button_ax, 'Curvature', hovercolor='0.975')

    def plot_curvature_event(event):
        frame_index = frames_slider.val
        frame = frames[frame_index]
        dots = frame.frame.particles.position
        Ks, Hs = geometry.calculate_curvatures_by_interpolation(dots)
        fig_Ks, ax_Ks = plt.subplots()
        plotutils.imshow_with_colorbar(fig_Ks, ax_Ks, Ks, "K")
        fig_Ks.show()

    button.on_clicked(plot_curvature_event)

    try:
        plt.show()
    except KeyboardInterrupt:
        print("Bye!")


def plot_frames_from_trajectory(trajectory_file, output_folder):
    print("Converting trajectory to frame images")
    frames = latticedefects.trajectory.load_trajectory(trajectory_file)
    for frame in frames:
        fig: Figure = plt.figure()
        ax: Axes3D = fig.add_subplot(111, projection="3d", azim=-40, elev=12)
        create_fig_and_plot_dots(ax, frame)
        plotutils.set_3D_labels(ax)

        ax.set_zlim(-1, 1)
        print(f"saving timestep={frame.timestep}")
        ax.set_title(f'$ E_s $={frame.harmonic_energy:.6f}, '
                     f'$ E_b $={frame.dihedrals_energy:.6f}, '
                     f'timestep={frame.timestep}\n'
                     f'total energy='
                     f'{frame.harmonic_energy + frame.dihedrals_energy:.6f}')
        fig.savefig(
            os.path.join(output_folder, f'ts={frame.timestep}.svg'))
        plt.clf()
