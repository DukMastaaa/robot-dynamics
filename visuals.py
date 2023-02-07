import functools
import itertools
from typing import Optional
import numpy as np
import modern_robotics as mr

from tqdm import tqdm

from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import proj3d, Axes3D
import matplotlib.pyplot as plt
import matplotlib.axes as axes
import matplotlib.animation as animation
import matplotlib.figure

# Set ffmpeg path to local executable path
# https://www.youtube.com/watch?v=bNbN9yoEOdU
from matplotlib.animation import FFMpegWriter
plt.rcParams["animation.ffmpeg_path"] = r"D:\Programs\ffmpeg-5.1.2-essentials_build\bin\ffmpeg.exe"


class AnglePlot:
    def __init__(self,
                 theta_axis: axes.Axes, dtheta_axis: axes.Axes, tau_axis: axes.Axes,
                 theta_history: np.ndarray, dtheta_history: np.ndarray, tau_history: np.ndarray,
                 torque_limits: np.ndarray,
                 t_final: float, ticks_per_sec: int):
        self.theta_history = theta_history
        self.dtheta_history = dtheta_history
        self.tau_history = tau_history
        self.t_final = t_final
        self.ticks_per_sec = ticks_per_sec
        self.histories = (self.theta_history, self.dtheta_history, self.tau_history)
        
        self.joint_count = len(theta_history[:,0])
        
        self.time_values = np.linspace(0, self.t_final, len(theta_history[0,:]))
        self.end_time = self.time_values[-1]
        self.ylimits = list(map(lambda history: (np.min(history) * 1.1, np.max(history) * 1.1), self.histories))
        
        self.theta_axis = theta_axis
        self.dtheta_axis = dtheta_axis
        self.tau_axis = tau_axis
        self.axes = (self.theta_axis, self.dtheta_axis, self.tau_axis)

        axis_ylabels = ["angle [rad]", "velocity [rad/s]", "torque [Nm]"]
        for axis, ylabel in zip(self.axes, axis_ylabels):
            axis.set_ylabel(ylabel)

        self.tau_axis.set_xlabel("time [s]")
        unique_torque_limits = set(torque_limits)
        for limit in unique_torque_limits:
            self.tau_axis.plot([0, self.end_time], [-limit, -limit], color="gray", linestyle="dashed")
            self.tau_axis.plot([0, self.end_time], [limit, limit], color="gray", linestyle="dashed")

        mock_data = np.zeros((1, 5))
        self.theta_plots = self.theta_axis.plot([0], mock_data)
        self.dtheta_plots = self.dtheta_axis.plot([0], mock_data)
        self.tau_plots = self.tau_axis.plot([0], mock_data)
        
        self.plotlists = (self.theta_plots, self.dtheta_plots, self.tau_plots)

        # labels = [f"joint {i+1}" for i in range(self.joint_count)]
        # self.legend = self.legend(
        #     labels, loc='lower right', bbox_to_anchor=(1,-0.1), ncol=len(labels), bbox_transform=self.fig.transFigure
        # )

        # handles, labels = self.ax.get_legend_handles_labels()
        # self.legend = self.fig.legend(handles, labels, loc='upper center')
        
        self.artists = tuple(self.theta_plots + self.dtheta_plots + self.tau_plots) + self.axes \
            # + (self.legend,)

    def update(self, index):
        if index == 0:
            for axis, (lower, upper) in zip(self.axes, self.ylimits):
                axis.set_xbound(0, self.end_time)
                axis.set_ybound(lower, upper)
        time = self.time_values[:index+1]
        for plots, history in zip(self.plotlists, self.histories):
            for i, plot in enumerate(plots):
                plot.set_data(time, history[i,:index+1])

        # self.theta_plot.set_data(time, (self.theta_history[:,:index+1]).T)
        # self.dtheta_plot.set_data(time, (self.dtheta_history[:,:index+1]).T)
        # self.tau_plot.set_data(time, (self.tau_history[:,:index+1]).T)


def set_plot3d_data(plot3d, x, y, z):
    plot3d.set_data(x, y)
    plot3d.set_3d_properties(z)

class FrameDraw:
    def __init__(self, ax: Axes3D, name: str = "", s: str = "rgb", L: float = 1.0):
        self.ax = ax
        self.L = L

        self.x_line, = self.ax.plot3D([0], [0], [0], s[0] + "-")
        self.y_line, = self.ax.plot3D([0], [0], [0], s[1] + "-")
        self.z_line, = self.ax.plot3D([0], [0], [0], s[2] + "-")

        self.name = name
        self.text = self.ax.text(0, 0, 0, s=self.name, fontsize=12)
        
        self.artists = (self.x_line, self.y_line, self.z_line, self.text)
    
    def update(self, T: np.ndarray):
        """Plots the frame defined by the transformation matrix T."""
        _, origin = mr.TransToRp(T)
        # get images of basis vectors in new frame
        x_image = (T @ [self.L,      0,      0, 1])[0:3]
        y_image = (T @ [     0, self.L,      0, 1])[0:3]
        z_image = (T @ [     0,      0, self.L, 1])[0:3]
        # combine each image and origin, unpack into plot()
        set_plot3d_data(self.x_line, *np.array([origin, x_image]).T)
        set_plot3d_data(self.y_line, *np.array([origin, y_image]).T)
        set_plot3d_data(self.z_line, *np.array([origin, z_image]).T)
        # draw text
        self.text.set_position_3d(origin)


def fk_frames(thetalist, Mlink_list, Mrotor_list, Slist) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Returns a list of transformation matrices for each of the joint frames,
    at the given configuration, for the internal frames and joint frames.
    Mlink_list = $M_{i_{L}, (i-1)_{L}}$
    Mrotor_list = $M_{i_{R}, (i-1)_{L}}$
    """
    
    n = len(thetalist)
    joint_frames = []
    internal_frames = []
    
    link_accumulator_M = np.eye(4)
    for i in range(n):
        joint_temp = link_accumulator_M @ mr.TransInv(Mrotor_list[i])
        new_link_acc = link_accumulator_M @ mr.TransInv(Mlink_list[i])
        link_frame = mr.FKinSpace(new_link_acc, Slist[:,0:i+1], thetalist[0:i+1])
        internal_frames.append(link_frame)

        joint_frame = mr.FKinSpace(joint_temp, Slist[:,0:i+1], thetalist[0:i+1])
        joint_frames.append(joint_frame)

        link_accumulator_M = new_link_acc

    return joint_frames, internal_frames


class ArmAnimation:
    def __init__(self, ax: Axes3D,
                 theta_history: np.ndarray,
                 t_final: float, ticks_per_sec: int,
                 Mlink_list: np.ndarray, Mrotor_list: np.ndarray, Slist: np.ndarray):
        self.theta_history = theta_history
        self.ticks_per_sec = ticks_per_sec
        self.total_ticks = int(t_final * ticks_per_sec)
        
        self.Mlink_list = Mlink_list
        self.Mrotor_list = Mrotor_list
        self.Slist = Slist
        
        self.ax: Axes3D = ax
        self.ax.autoscale(enable=False, axis='both')
        self.ax.set_box_aspect(aspect=(4, 4, 3), zoom=1.2)
        
        c_x, c_y, c_z = 0, 0, 0.5
        r_x, r_y, r_z = 1, 1, 1
        self.ax.set_xbound(c_x-r_x/2, c_x+r_x/2)
        self.ax.set_ybound(c_y-r_y/2, c_y+r_y/2)
        self.ax.set_zbound(c_z-r_z/2, c_z+r_z/2)
        self.ax.view_init(elev=12, azim=64)
        
        self.ax.set_xlabel("x [m]")
        self.ax.set_ylabel("y [m]")
        self.ax.set_zlabel("z [m]")

        self.jointcount = len(theta_history[:,0])
        L = 0.05

        self.origin_frame = FrameDraw(self.ax, name="O", s="rrr", L=0.1)
        self.joint_frames_to_draw = [
            FrameDraw(self.ax, name=f"J{i+1}", s="ckm", L=L) for i in range(self.jointcount)
        ]
        self.internal_frames_to_draw = [
            FrameDraw(self.ax, name=f"L{i+1}", s="rgb", L=L) for i in range(self.jointcount + 1)
        ]

        self.link_line, = self.ax.plot3D([0], [0], [0], "gray")
        self.artists = tuple(
            fd.artists
            for fd in itertools.chain(
                [self.origin_frame], self.joint_frames_to_draw, self.internal_frames_to_draw,
            )
        ) + (self.link_line,)

    def draw_leg(self, *, joint_frames, internal_frames):
        # update origin frame
        self.origin_frame.update(np.eye(4))
        # update joint frames
        for T, framedraw in zip(joint_frames, self.joint_frames_to_draw):
            framedraw.update(T)
        # update internal frames
        for T, framedraw in zip(internal_frames, self.internal_frames_to_draw):
            framedraw.update(T)
        # update link line
        origins = np.array(
            [(T @ [0, 0, 0, 1])[:3] for T in joint_frames] + \
                [(internal_frames[-1] @ [0, 0, 0, 1])[:3]]
        )
        set_plot3d_data(self.link_line, *origins.T)
    
    def update(self, frame_num: int):
        thetalist = self.theta_history[:,frame_num]
        
        # get transformation matrices of joint frames, which coincide with the rotor frames.
        # also get transformation matrices of internal link frames
        # by faking the zero configuration matrices Mseq
        # to be at the link frames instead of the joints
        joint_frames, internal_frames = fk_frames(thetalist, self.Mlink_list, self.Mrotor_list, self.Slist)
        
        self.draw_leg(joint_frames=joint_frames, internal_frames=internal_frames)


class FullAnimation:
    def __init__(self, fig: Optional[matplotlib.figure.Figure],
                 theta_history: np.ndarray, dtheta_history: np.ndarray, tau_history: np.ndarray,
                 torque_limits: np.ndarray,
                 t_final: float, ticks_per_sec: int,
                 Mlink_list: np.ndarray, Mrotor_list: np.ndarray, Slist: np.ndarray,
                 title: str):
        if fig is None:
            self.fig = plt.figure()
        else:
            self.fig = fig
        self.fig.set_figwidth(10)
        self.fig.set_figheight(6)
        
        self.t_final = t_final
        self.ticks_per_sec = ticks_per_sec
        self.total_ticks = int(t_final * ticks_per_sec)
        self.title = title
        
        mosaic = [
            ["theta", "arm"],
            ["dtheta", "arm"],
            ["tau", "arm"],
        ]
        gridspec_kw = {
            "width_ratios": [0.4, 0.6],
            "right": 0.94,
            "wspace": 0.24
        }
        self.axd: dict[str, plt.Axes] = self.fig.subplot_mosaic(
            mosaic,
            gridspec_kw=gridspec_kw,
        )
        
        # change arm axes to be 3d
        ss = self.axd["arm"].get_subplotspec()
        self.axd["arm"].remove()
        self.axd["arm"] = self.fig.add_subplot(ss, projection="3d")
        
        self.arm_anim = ArmAnimation(
            self.axd["arm"], theta_history, t_final, ticks_per_sec, Mlink_list, Mrotor_list, Slist 
        )
        self.angle_anim = AnglePlot(
            self.axd["theta"], self.axd["dtheta"], self.axd["tau"],
            theta_history, dtheta_history, tau_history, torque_limits, t_final, ticks_per_sec
        )
        
        labels = [f"joint {i+1}" for i in range(len(theta_history[:,0]))]
        self.legend = self.fig.legend(
            handles=functools.reduce(lambda x, y: x+y, self.angle_anim.plotlists),
            labels=labels,
            loc='upper center',
            # bbox_to_anchor=(1,-0.1),
            # bbox_transform=self.fig.transFigure,
            ncol=len(labels),
        )
        
        self.anim: Optional[animation.FuncAnimation] = None
        
        self.artists = self.arm_anim.artists + self.angle_anim.artists + (self.legend,)
        
        self.paused = False
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)

        self.pbar: Optional[tqdm] = None
    
    def key_press(self, event):
        if event.key == " " and self.anim is not None:
            if self.paused:
                self.anim.resume()
            else:
                self.anim.pause()
            self.paused = not self.paused

    def animation_init(self):
        return self.artists
    
    def animation_tick(self, frame_num: int):
        if self.pbar is not None:
            self.pbar.update(1)
            if frame_num + self.step >= self.total_ticks:
                self.pbar.close()
        self.arm_anim.update(frame_num)
        self.angle_anim.update(frame_num)
        return self.artists

    def animate(self, use_mp4: bool, filepath: str, fps: int):
        self.step = max(1, int(self.ticks_per_sec / fps))
        self.anim = animation.FuncAnimation(
            self.fig, self.animation_tick, frames=range(0, self.total_ticks, self.step), init_func=self.animation_init,
            blit=False
            # blit=True
        )
        self.pbar = tqdm(total=(self.total_ticks // self.step))
        if use_mp4:
            writer = animation.FFMpegWriter(fps=fps)
            self.anim.save(filepath, writer=writer, dpi=250)
        else:
            plt.show()

"""
        self.t_text = self.ax.text(
            x=0.9, y=0.85,
            s="t=0", fontsize=10, ha="left",
            transform=self.ax.transFigure
        )
        self.t_text.set_text(f"t={round(frame_num/self.ticks_per_sec, 2)}")
"""