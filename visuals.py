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

# Set ffmpeg path to local executable path
# https://www.youtube.com/watch?v=bNbN9yoEOdU
from matplotlib.animation import FFMpegWriter
plt.rcParams["animation.ffmpeg_path"] = r"D:\Programs\ffmpeg-5.1.2-essentials_build\bin\ffmpeg.exe"


def set_plot3d_data(plot3d, x, y, z):
    plot3d.set_data(x, y)
    plot3d.set_3d_properties(z)

class FrameDraw:
    def __init__(self, ax: Axes3D, name="", s="rgb", L=1):
        self.ax = ax
        self.L = L

        self.x_line, = self.ax.plot3D([0], [0], [0], s[0] + "-")
        self.y_line, = self.ax.plot3D([0], [0], [0], s[1] + "-")
        self.z_line, = self.ax.plot3D([0], [0], [0], s[2] + "-")

        self.name = name
        self.text = self.ax.text(0, 0, 0, ss=self.name, fontsize=12)
        
        self.artists = (self.x_line, self.y_line, self.z_line, self.text)
    
    def update(self, T):
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
    def __init__(self, theta_history, t_final, ticks_per_sec, Mlink_list, Mrotor_list, Slist, title):
        self.theta_history = theta_history
        self.ticks_per_sec = ticks_per_sec
        self.total_ticks = int(t_final * ticks_per_sec)
        
        self.Mlink_list = Mlink_list
        self.Mrotor_list = Mrotor_list
        self.Slist = Slist
        
        self.fig = plt.figure()
        self.ax: Axes3D = plt.axes(projection="3d")
        self.ax.autoscale(enable=False, axis='both')
        self.title = title
        
        c_x, c_y, c_z = 0, 0, 0.5
        r_x, r_y, r_z = 1, 1, 1
        self.ax.set_xbound(c_x-r_x/2, c_x+r_x/2)
        self.ax.set_ybound(c_y-r_y/2, c_y+r_y/2)
        self.ax.set_zbound(c_z-r_z/2, c_z+r_z/2)
        self.ax.view_init(elev=12, azim=64)
        
        self.ax.set_title(title)
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
        self.t_text = self.fig.text(
            x=0.45, y=0.85,
            s="t=0", fontsize=10, ha="left",
            transform=self.fig.transFigure
        )
        self.fig.text(x=0.8, y=0.85, s="rgb=internal\nckm=joint", fontsize=10, transform=self.fig.transFigure)
        
        self.anim: Optional[animation.FuncAnimation] = None
        self.artists = tuple(
            fd.artists
            for fd in itertools.chain(
                [self.origin_frame], self.joint_frames_to_draw, self.internal_frames_to_draw,
            )
        ) + (self.link_line, self.t_text)
        
        self.paused = False
        self.fig.canvas.mpl_connect('key_press_event', self.key_press)
        
        self.pbar = None
    
    def key_press(self, event):
        if event.key == " " and self.anim is not None:
            if self.paused:
                self.anim.resume()
            else:
                self.anim.pause()
            self.paused = not self.paused

    def clear_plot(self):
        """
        Clears plot without clearing axes.
        https://stackoverflow.com/a/43081720
        """
        for artist in plt.gca().lines + plt.gca().collections:
            artist.remove()
    
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
        origins = np.array([(T @ [0, 0, 0, 1])[:3] for T in joint_frames])
        set_plot3d_data(self.link_line, *origins.T)

    def animation_init(self):
        return self.artists
    
    def animation_tick(self, frame_num: int):
        if self.pbar is not None:
            self.pbar.update(1)
            if frame_num + self.step >= self.total_ticks:
                self.pbar.close()
        
        thetalist = self.theta_history[:,frame_num]
        
        # get transformation matrices of joint frames, which coincide with the rotor frames.
        # also get transformation matrices of internal link frames
        # by faking the zero configuration matrices Mseq
        # to be at the link frames instead of the joints
        joint_frames, internal_frames = fk_frames(thetalist, self.Mlink_list, self.Mrotor_list, self.Slist)
        
        self.draw_leg(joint_frames=joint_frames, internal_frames=internal_frames)
        
        self.t_text.set_text(f"t={round(frame_num/self.ticks_per_sec, 2)}")
        
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


def angle_plots(theta_history, dtheta_history, tau_history,
                t_final, ticks_per_sec, filename, do_tau: bool):
    """
    Plots theta and d_theta over time, given the video parameters.
    Setting do_tau also plots tau over time.
    """
    subplot = (3, 1) if do_tau else (2, 1)

    N = int(t_final * ticks_per_sec)
    times = np.array(range(N)) / ticks_per_sec
    jointcount = len(theta_history[:,0])

    plt.figure(1)
    plt.subplot(*subplot, 1)
    plt.plot(times, theta_history.T)
    plt.legend([f"theta_{i+1}" for i in range(jointcount)], loc="upper right")
    # plt.xlabel("time [s]")
    plt.ylabel("angle [rad]")
    
    
    plt.subplot(*subplot, 2)
    plt.plot(times, dtheta_history.T)
    plt.legend([f"dtheta_{i+1}" for i in range(jointcount)], loc="upper right")
    plt.ylabel("velocity [rad/s]")
    if not do_tau:
        plt.xlabel("time [s]")

    if do_tau:
        plt.subplot(*subplot, 3)
        plt.plot(times, tau_history.T)
        plt.legend([f"tau_{i+1}" for i in range(jointcount)], loc="upper right")
        plt.ylabel("torque [Nm]")
        plt.xlabel("time [s]")

    plt.suptitle(f"Plots for {filename}")
    plt.show()
