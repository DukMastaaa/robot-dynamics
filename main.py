import numpy as np
import modern_robotics as mr

from simulation import *
from visuals import *


def generic_movement_test(jointcount: int, time_per_joint: float, 
                          ticks_per_sec: int) -> tuple[float, float, np.ndarray]:
    """Returns a theta_history where each joint value sweeps from 0 to 2pi."""
    N = time_per_joint * ticks_per_sec
    assert N.is_integer()
    N = int(N)
    theta_history = np.zeros((jointcount, N*jointcount))
    for i in range(jointcount):
        theta_history[i,(i*N):((i+1)*N)] = np.linspace(0, 2*np.pi, N)
    t_final = N * jointcount
    return t_final, ticks_per_sec, theta_history


def run_sim(geared_params, thetalist0, dthetalist0, tau_function,
            t_final, ticks_per_sec) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    N = int(t_final * ticks_per_sec)
    return forward_dynamics_simulation(
        thetalist0, dthetalist0, tau_function,
        geared_inverse_dynamics, geared_params,  # ✨dependency injection✨
        t_final, N
    )


def export_data(dump_filename: str, t_final: float, ticks_per_sec: float, 
                theta_history: np.ndarray, dtheta_history: np.ndarray, tau_history: np.ndarray) -> None:
    with open(dump_filename, "wb") as file:
        np.save(file, np.array([t_final, ticks_per_sec]))
        np.save(file, theta_history)
        np.save(file, dtheta_history)
        np.save(file, tau_history)


def import_data(dump_filename: str) -> tuple[float, int, np.ndarray, np.ndarray, np.ndarray]:
    with open(dump_filename, "rb") as file:
        t_final, ticks_per_sec = np.load(file)
        theta_history = np.load(file)
        dtheta_history = np.load(file)
        tau_history = np.load(file)
    return t_final, ticks_per_sec, theta_history, dtheta_history, tau_history


def mytraj(t):
    # quintic time scaling
    # adapted from code of mr.JointTrajectory
    start = np.zeros(5)
    finish = np.array([np.pi/2, 0, 0, 0, 0])
    Tf = 2
    if t > Tf:
        return finish
    elif t < 0:
        return start
    else:
        normalised_t = t / Tf
        s = mr.QuinticTimeScaling(Tf, t)
        return s * finish + (1-s) * start


def main():
    import atlas_definition as atlas
    geared_params = atlas.geared_params.copy()
    atlas.set_end_effector_mass(geared_params, 0.5, atlas.mm_to_m(100))
    atlas.set_gear_ratios(geared_params, [10, 25, 25, 5, 15])
    
    # thetalist0 = np.zeros(5)
    # dthetalist0 = np.zeros(5)
    
    # traj_function = lambda t: np.zeros(5)
    
    # kp = np.array([1, 1, 1, 1, 1]) * 75
    # ki = np.array([1, 1, 1, 1, 1]) * 0
    # kd = np.array([1, 1, 1, 1, 1]) * 75
    # damping_coefficient = 2
    
    # tau_function = tau_function_from_pid(kp, ki, kd, damping_coefficient, geared_params["torque_limits"], traj_function)
    
    # t_final = 20
    # ticks_per_sec = 30

    # theta_history, dtheta_history, tau_history = run_sim(
    #     geared_params, thetalist0, dthetalist0, tau_function, t_final, ticks_per_sec
    # )

    filename = "backup.npy"
    # export_data(filename, t_final, ticks_per_sec, theta_history, dtheta_history, tau_history)
    t_final, ticks_per_sec, theta_history, dtheta_history, tau_history = import_data(filename)

    fps = 24
    title = "uhhhhhh"
    use_mp4 = False
    # use_mp4 = True
    mp4_filename = "animtest.mp4"
    
    anim = FullAnimation(
        None,
        theta_history, dtheta_history, tau_history,
        t_final, ticks_per_sec,
        geared_params["Mlink_list"], geared_params["Mrotor_list"], geared_params["Slist"],
        "beans"
    )
    anim.animate(use_mp4, mp4_filename, fps)



if __name__ == "__main__":
    main()
