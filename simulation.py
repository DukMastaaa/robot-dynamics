from typing import Callable
import numpy as np
import modern_robotics as mr

from tqdm import tqdm

### MATH HELPER FUNCTIONS

def wrap_2pi(angles):
    """
    Wraps the given angles to within [-pi, pi).
    https://stackoverflow.com/a/15927914
    """
    return (angles + np.pi) % (2 * np.pi) - np.pi

def clip(vec: np.ndarray, abs_max: np.ndarray):
    """
    Limits the magnitude of each element in vec to its corresponding
    element in abs_max.
    """
    result = np.zeros(len(vec))
    for i, val in enumerate(vec):
        if abs(val) <= abs_max[i]:
            result[i] = val
        else:
            result[i] = abs_max[i] * np.sign(val)
    return result


### SIMULATION FUNCTIONS

def mass_matrix(thetalist: np.ndarray, inverse_dynamics_function: Callable, params):
    """
    Calculates the mass matrix at the given theta by building
    up the results from successive calls to the given
    inverse_dynamics_function, with signature:
        inverse_dynamics_function(theta, dtheta, ddtheta, **params)
    """
    n = len(thetalist)
    M = np.zeros((n, n))
    params = params.copy()
    params["g"]  = np.zeros(3)
    params["Ftip"] = np.zeros(6)
    dthetalist = np.zeros(n)
    for i in range(n):
        ddthetalist = np.zeros(n)
        ddthetalist[i] = 1
        M[:,i] = inverse_dynamics_function(
            thetalist, dthetalist, ddthetalist, **params
        )
    return M

def h_term(thetalist: np.ndarray, dthetalist: np.ndarray, inverse_dynamics_function: Callable, params):
    """
    Calculates the term h(theta, dtheta) = c(theta, dtheta) + g(theta),
    by calling the given inverse_dynamics_function with signature:
        inverse_dynamics_function(theta, dtheta, ddtheta, **params)
    """
    n = len(thetalist)
    ddthetalist = np.zeros(n)
    params = params.copy()
    params["Ftip"] = np.zeros(6)
    return inverse_dynamics_function(
        thetalist, dthetalist, ddthetalist, **params
    )

def forward_dynamics_simulation(thetalist0: np.ndarray, dthetalist0: np.ndarray,
                                tau_function: Callable, inverse_dynamics_function: Callable,
                                params,
                                t_final: float, N: int):
    """
    Computes the forward dynamics from t in [0, t_final] for N integration steps,
    given initial conditions and zero Ftip.
    Joint torques at each time step are calculated by tau_function with signature:
        tau_function(t, delta_t, theta, dtheta)
    This function uses the given inverse_dynamics_function with signature:
        inverse_dynamics_function(theta, dtheta, ddtheta, **params)
    Returns the joint angle vectors, velocities and torques for each timestep
    as N by len(thetalist0) matrices.
    """
    num_joints = len(thetalist0)
    delta_t = t_final / N
    
    params = params.copy()
    params["Ftip"] = np.zeros(6)

    # initialise histories
    theta_history = np.zeros((num_joints, N))
    theta_history[:,0] = thetalist0
    dtheta_history = np.zeros((num_joints, N))
    dtheta_history[:,0] = dthetalist0
    tau_history = np.zeros((num_joints, N))

    for i in tqdm(range(N-1)):
        M = mass_matrix(theta_history[:,i], inverse_dynamics_function, params)
        h = h_term(theta_history[:,i], dtheta_history[:,i], inverse_dynamics_function, params)

        # call tau_function with current theta and dtheta
        taulist = tau_function(i * delta_t, delta_t, theta_history[:,i], dtheta_history[:,i])
        # save to history
        tau_history[:,i] = taulist
        ddthetalist = np.linalg.solve(M, taulist - h)

        # integrate
        theta_history[:,i+1] = theta_history[:,i] + dtheta_history[:,i] * delta_t
        dtheta_history[:,i+1] = dtheta_history[:,i] + ddthetalist * delta_t
    
    tau_history[:,N-1] = tau_history[:,N-2]

    return theta_history, dtheta_history, tau_history


### CONTROLLER CODE

def tau_function_from_pid(kp: np.ndarray, ki: np.ndarray, kd: np.ndarray,
                          damping_coefficient: float, tau_abs_max: np.ndarray,
                          theta_ref_function: Callable):
    """
    Returns a function tau(t, delta_t, theta, dtheta) suitable for use with
    forward_dynamics_simulation, applying joint torques to enforce the
    joint trajectory defined by theta_ref_function(t).
    Viscous friction in the joints is modelled with the given damping_coefficient.
    tau_abs_max is a vector specifying the torque limits for each joint.
    """
    joint_count = len(tau_abs_max)
    prev_error = np.zeros(joint_count)
    integral = np.zeros(joint_count)

    # we define a ✨lexical closure✨ to pass to the simulation 
    def tau_function(t, delta_t, theta, dtheta):
        nonlocal prev_error, integral
        friction = -damping_coefficient * dtheta
        error = theta_ref_function(t) - theta
        # we don't use the dtheta passed in, as we'd need to
        # differentiate the trajectory to get d(error)/dt.
        # instead, find numerical derivative with 1st-order approx.
        derror = (error - prev_error) / delta_t
        # again, numerical approximation using an accumulator
        integral = integral + theta * delta_t
        # total torques from friction and PID. elementwise multiply
        tau = friction + np.multiply(kp, error) + np.multiply(ki, integral) + np.multiply(kd, derror)
        # enforce torque limits
        tau = clip(tau, tau_abs_max)
        prev_error = error
        return tau

    return tau_function


### GEARED DYNAMICS

def empty(n):
    return [None for _ in range(n)]

def geared_inverse_dynamics(thetalist: np.ndarray, dthetalist: np.ndarray, ddthetalist: np.ndarray,
                            g: np.ndarray, Ftip: np.ndarray,
                            Mlink_list: np.ndarray, Mrotor_list: np.ndarray,
                            Glink: np.ndarray, Grotor: np.ndarray,
                            Alist: np.ndarray, Rlist: np.ndarray,
                            gear_ratios: np.ndarray, **_):
    """
    Calculates the inverse dynamics for a geared system.
    Mlink_list is T_{i_{L}, (i-1)_{L}} in the textbook.
    Mrotor_list is T_{i_{R}, (i-1)_{L}} in the textbook.
    Glink and Grotor are \\mathcal{G}_{i_{L}} and
    \\mathcal{G}_{i_{R}} in the textbook.
    The rest of the parameters are defined as in
    mr.InverseDynamics.
    """
    n = len(thetalist)
        
    # T_{i_{L}, (i-1)_{L}} in the textbook
    Tlink = empty(n+1)
    Tlink[n] = Mlink_list[n]
    # T_{i_{R}, (i-1)_{L}} in the textbook
    Trotor = empty(n+1)
    # the textbook doesn't define this because it's nonsense
    # (the hand has no rotor), but it's used in the first formula
    # for backwards iteration. G is zero though, so the whole
    # term goes to zero. it still needs to be defined here for the
    # code to work, however.
    Trotor[n] = np.eye(4)

    # twists
    Vrotor = empty(n+1)
    Vrotor[n] = np.zeros(6)
    Vlink = empty(n+1)
    Vlink[-1] = np.zeros(6)
    # accelerations
    Vdot_rotor = empty(n+1)
    Vdot_rotor[n] = np.zeros(6)
    Vdot_link = empty(n+1)
    Vdot_link[-1] = np.concatenate([np.zeros(3), -g])

    # wrenches
    Flist = empty(n+1)
    Flist[n] = Ftip

    # gearhead and motor torques
    tau_gear = empty(n)
    tau_motor = empty(n)
    
    # set G_{n_{R}}, in textbook, to zero (note 0-based indexing)
    Grotor = np.insert(Grotor, n, 0, axis=0)
    mr.InverseDynamics
    # forward iteration
    # we index from 0, and abuse that index -1 is end of list
    for i in range(0, n):
        Trotor[i] = \
            mr.MatrixExp6(-mr.VecTose3(Rlist[i]) * gear_ratios[i] * thetalist[i]) \
            @ Mrotor_list[i]
        Tlink[i] = \
            mr.MatrixExp6(-mr.VecTose3(Alist[i]) * thetalist[i]) \
            @ Mlink_list[i]
        Vrotor[i] = mr.Adjoint(Trotor[i]) @ Vlink[i-1] \
            + Rlist[i] * gear_ratios[i] * dthetalist[i]
        Vlink[i] = mr.Adjoint(Tlink[i]) @ Vlink[i-1] \
            + Alist[i] * dthetalist[i]
        Vdot_rotor[i] = mr.Adjoint(Trotor[i]) @ Vdot_link[i-1] \
            + (mr.ad(Vrotor[i]) @ Rlist[i]) * gear_ratios[i] * dthetalist[i] \
            + Rlist[i] * gear_ratios[i] * ddthetalist[i]
        Vdot_link[i] = mr.Adjoint(Tlink[i])  @ Vdot_link[i-1] \
            + mr.ad(Vlink[i]) @ Alist[i] \
            + Alist[i] * ddthetalist[i]
        
    # backward iteration
    for i in reversed(range(0, n)):
        Flist[i] = \
            mr.Adjoint(Tlink[i+1]).T @ Flist[i+1] \
            + Glink[i] @ Vdot_link[i] \
            - mr.ad(Vlink[i]).T @ (
                Glink[i] @ Vlink[i]
            ) \
            + mr.Adjoint(Trotor[i+1]).T @ (
                Grotor[i+1] @ Vdot_rotor[i+1]
                - mr.ad(Vrotor[i+1]).T @ (
                    Grotor[i+1] @ Vrotor[i+1]
                )
            )
        tau_gear[i] = Alist[i].T @ Flist[i]
        tau_motor[i] = tau_gear[i]/gear_ratios[i] \
            + Rlist[i].T @ (
                Grotor[i] @ Vdot_rotor[i]
                - mr.ad(Vrotor[i]).T @ (
                    Grotor[i] @ Vrotor[i]
                )
            )
    return tau_motor
