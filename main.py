import numpy as np
import modern_robotics as mr

from simulation import *
from visuals import *


def run_sim(geared_params, thetalist0, dthetalist0, tau_function, t_final, ticks_per_sec):
    N = int(t_final * ticks_per_sec)
    return forward_dynamics_simulation(
        thetalist0, dthetalist0, tau_function,
        geared_inverse_dynamics, geared_params,  # ✨dependency injection✨
        t_final, N
    )


def export_data(geared_params, dump_filename, thetalist0, dthetalist0, tau_function, t_final, ticks_per_sec):
    theta_history, dtheta_history, tau_history = run_sim(
        geared_params, thetalist0, dthetalist0, tau_function, t_final, ticks_per_sec
    )
    with open(dump_filename, "wb") as file:
        np.save(file, np.array([t_final, ticks_per_sec]))
        np.save(file, theta_history)
        np.save(file, dtheta_history)
        np.save(file, tau_history)


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


def uhhh(geared_params):
    thetalist0 = np.zeros(5)
    dthetalist0 = np.zeros(5)
    damping_coefficient = 2
    
    traj_function = lambda t: np.zeros(5)
    
    tau_function = tau_function_from_pid(10, 0.1, 0, damping_coefficient, [1, 1, 1, 1, 1], traj_function)
    
    t_final = 20
    ticks_per_sec = 20

    theta_history, dtheta_history, tau_history = run_sim(
        geared_params, thetalist0, dthetalist0, tau_function, t_final, ticks_per_sec
    )
    
    angle_plots(theta_history, dtheta_history, tau_history, t_final, ticks_per_sec, "", True)

    fps = 24
    title = "uhhhhhh"
    use_mp4 = False
    # use_mp4 = True
    mp4_filename = "traj.mp4"
    
    anim = ArmAnimation(
        theta_history,
        t_final, ticks_per_sec,
        geared_params["Mlink_list"], geared_params["Mrotor_list"], Slist,
        title
    )
    
    anim.animate(use_mp4, mp4_filename, fps)



if __name__ == "__main__":
    from atlas_definition import *
    set_end_effector_mass(geared_params, 0.5, mm_to_m(100))
    set_gear_ratios(geared_params, [5, 10, 10, 1, 1])
    
    uhhh(geared_params)
    
    """
    # do_export = True
    do_export = False
    
    if do_export:
        dump_filename = "traj.npy"

        # thetalist0 = np.array([0, np.pi, -np.pi/2, 0, 0])
        thetalist0 = np.zeros(5)
        dthetalist0 = np.zeros(5)
        damping_coefficient = 2
        
        traj_function = lambda t: np.zeros(5)
        
        tau_function = tau_function_from_pid(6, 0.4, 0.5, damping_coefficient, [80, 80, 80, 80, 80], traj_function)
        
        # tau_function = lambda t, delta_t, theta, dtheta: -damping_coefficient * dtheta

        t_final = 10
        ticks_per_sec = 50

        export_data(geared_params, dump_filename, thetalist0, dthetalist0, tau_function, t_final, ticks_per_sec)

    else:
        dump_filename = "traj.npy"
        with open(dump_filename, "rb") as file:
            t_final, ticks_per_sec = np.load(file)
            theta_history = np.load(file)
            dtheta_history = np.load(file)
            tau_history = np.load(file)
        
        angle_plots(theta_history, dtheta_history, tau_history, t_final, ticks_per_sec, "", True)
        
        
        # t_final = 1
        # ticks_per_sec = 50
        
        # N = t_final * ticks_per_sec
        # jointcount = 5
        # theta_history = np.zeros((jointcount, N*jointcount))
        # for i in range(jointcount):
        #     theta_history[i,(i*N):((i+1)*N)] = np.linspace(0, 2*np.pi, N)

        # t_final *= jointcount
        
        
        
        fps = 24
        title = "uhhhhhh"
        
        use_mp4 = False
        # use_mp4 = True
        mp4_filename = "traj.mp4"
        # mp4_filename = "test.mp4"
        
        anim = ArmAnimation(
            theta_history,
            t_final, ticks_per_sec,
            Mlink_list, Mrotor_list, Slist,
            title
        )
        
        anim.animate(use_mp4, mp4_filename, fps)
    """




"""if __name__ == "__main__":
    # from general_params import *
    # from geared_params import *
    from gearedptest import *
    
    # ungeared_params = {
    #     "g": np.array([0, 0, -9.81]),
    #     "Mlist": Mlist,
    #     "Slist": Slist,
    #     "Glist": Glist
    # }
    
    # # verify implementation is correct
    # thetalist0 = np.array([np.pi/2, -np.pi/2, -np.pi/8])
    # dthetalist0 = 0.1 * np.ones(3)
    # q2b_verify(thetalist0, dthetalist0, ungeared_params)
    
    # # run ungeared simulations
    # anim1(ungeared_params)
    # anim2(ungeared_params)
    
    # # run controller simulations
    # tau_max_h = 80
    # tau_max_k = 80
    # tau_max_a = 5
    # tau_abs_max = np.array([tau_max_h, tau_max_k, tau_max_a])
    # anim3(ungeared_params, tau_abs_max)
    # anim4(ungeared_params, tau_abs_max)
    
    # run geared simulation
    geared_params = {
        # "g": np.array([0, 0, -9.81]),
        "g": np.array([0, -9.81, 0]),
        "Mlink_list": Mlink_list,
        "Mrotor_list": Mrotor_list,
        "Glink": Glink_list,
        "Grotor": Grotor_list,
        "Alist": Alist,
        "Rlist": Rlist,
        "gear_ratios": gear_ratio_list
    }
    # anim5(geared_params)
    anim6(geared_params)
"""