import numpy as np
import modern_robotics as mr
from enum import Enum

### Unit Conversion

def mm_to_m(x):
    return x/1000

def gcm2_to_kgm2(x):
    return x * 1e-7

def gfcm2_to_kgm2(x):
    return x/100/1000*9.81

def diameter_to_radius(x):
    return x / 2


### Robot Geometry
# since the end-effector mass is yet to be defined, it's important that
# we calculate the original T of the last link which will then be modified.

def get_Alist(Mlink_list, Slist):
    """
    Returns a list of screw axes of joint i expressed in the link frame {i}.
    """
    n = np.shape(Slist)[1]
    Alist = [None for _ in range(n)]
    running_product = np.eye(4)
    for i in range(n):
        running_product = running_product @ Mlink_list[i]
        adj = mr.Adjoint(np.linalg.inv(running_product))
        Alist[i] = adj @ Slist[:,i]
    return Alist

def pToTrans(p):
    return mr.RpToTrans(np.eye(3), p)

L1 = mm_to_m(120)
L2 = mm_to_m(500.3)
L3 = mm_to_m(139.5)
L4 = mm_to_m(367.8)
L5_original = mm_to_m(119.2)

M_1R_0L = np.eye(4)
M_1L_0L = pToTrans([0, 0, -L1/2])

M_2R_1L = pToTrans([0, 0, -L1/2])
M_2L_1L = pToTrans([0, 0, -L1/2 - L2/2])

M_3R_2L = pToTrans([0, 0, -L2/2])
M_3L_2L = pToTrans([0, 0, -L2/2 - L3/2])

M_4R_3L = pToTrans([0, 0, -L3/2])
M_4L_3L = pToTrans([-L4/2, 0, -L3/2])

M_5R_4L = pToTrans([-L4/2, 0, 0])
M_5L_4L_original = pToTrans([-L4/2 - L5_original/2, 0, 0])

M_6R_5L = None
M_6L_5L_original = pToTrans([-L5_original/2, 0, 0])

# discrepancy with transpose here and in get_Alist
Slist = np.array([
    [0, 0, 1,           0,        0,  0],
    [0, 1, 0,         -L1,        0,  0],
    [0, 1, 0,      -L1-L2,        0,  0],
    [1, 0, 0,           0, L1+L2+L3,  0],
    [0, 1, 0,   -L1-L2-L3,        0, L4],
]).T

Rlist = np.array([
    [0, 0, 1, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0],
])


### Masses and Inertias

class Dir(Enum):
    X = 0
    Y = 1
    Z = 2

def direction_to_vec(dir: Dir):
    if dir == Dir.X:
        return np.array([1, 0, 0])
    elif dir == Dir.Y:
        return np.array([0, 1, 0])
    else:
        return np.array([0, 0, 1])

# assumes all motor weight is part of the rotor, not the stator,
# and that the links are homogeneous so com is geometric centre

def rotor_inertia(m, I, dir: Dir):
    if dir == Dir.X:
        moi = [I, 0, 0]
    elif dir == Dir.Y:
        moi = [0, I, 0]
    else:
        moi = [0, 0, I]
    return np.diag(np.concatenate([direction_to_vec(dir), [m, m, m]]))

def link_mass(R,r, rho, L):
    volume = np.pi * L * (R**2 - r**2)
    mass = rho * volume
    return mass

def link_moi(R, r, rho, L, axial_dir: Dir):
    axial_moi = np.pi*rho*L * (R**4 - r**4) / 2
    radial_moi = np.pi*rho*L * (3*(R**4 - r**4) + (L**2)*(R**2 - r**2)) / 12
    axial_vec = direction_to_vec(axial_dir)
    moi = axial_vec * axial_moi + (1 - axial_vec) * radial_moi
    return moi

def moi_m_to_inertia(moi, mass):
    return np.diag(np.concatenate([moi, [mass, mass, mass]]))

## Rotor
NEMA17_MASS_TOTAL = 1.05
NEMA17_MOI = gfcm2_to_kgm2(110)
NEMA17_HOLDING_TORQUE = 0.65  # Nm
NEMA23_MASS_TOTAL = 2.24
NEMA23_MOI = gfcm2_to_kgm2(480)  # this is a guess
NEMA23_HOLDING_TORQUE = 3  # Nm

m1_rotor = NEMA23_MASS_TOTAL
moi1_rotor = NEMA23_MOI
torque_lim_1 = NEMA23_HOLDING_TORQUE

m2_rotor = NEMA23_MASS_TOTAL
moi2_rotor = NEMA23_MOI
torque_lim_2 = NEMA23_HOLDING_TORQUE

m3_rotor = NEMA17_MASS_TOTAL
moi3_rotor = NEMA17_MOI
torque_lim_3 = NEMA17_HOLDING_TORQUE
# m3_rotor = NEMA23_MASS_TOTAL
# moi3_rotor = NEMA23_MOI
# torque_lim_3 = NEMA23_HOLDING_TORQUE

m4_rotor = NEMA17_MASS_TOTAL
moi4_rotor = NEMA17_MOI
torque_lim_4 = NEMA17_HOLDING_TORQUE

m5_rotor = NEMA17_MASS_TOTAL
moi5_rotor = NEMA17_MOI
torque_lim_5 = NEMA17_HOLDING_TORQUE

G1_rotor = rotor_inertia(m1_rotor, moi1_rotor, Dir.Z)
G2_rotor = rotor_inertia(m2_rotor, moi2_rotor, Dir.Y)
G3_rotor = rotor_inertia(m3_rotor, moi3_rotor, Dir.Y)
G4_rotor = rotor_inertia(m4_rotor, moi4_rotor, Dir.X)
G5_rotor = rotor_inertia(m5_rotor, moi5_rotor, Dir.Y)
Grotor_list = np.array([G1_rotor, G2_rotor, G3_rotor, G4_rotor, G5_rotor])

torque_limits = np.array([torque_lim_1, torque_lim_2, torque_lim_3, torque_lim_4, torque_lim_5])

# Link
rho = 2710  # kg*m^3
outer_radius = diameter_to_radius(0.048)
inner_radius = diameter_to_radius(0.0448)

m1_link = link_mass(outer_radius, inner_radius, rho, L1)
m2_link = link_mass(outer_radius, inner_radius, rho, L2)
m3_link = link_mass(outer_radius, inner_radius, rho, L3)
m4_link = link_mass(outer_radius, inner_radius, rho, L4)
m5_link_original = link_mass(outer_radius, inner_radius, rho, L5_original)

moi1_link = link_moi(outer_radius, inner_radius, rho, L1, Dir.Z)
moi2_link = link_moi(outer_radius, inner_radius, rho, L2, Dir.Y)
moi3_link = link_moi(outer_radius, inner_radius, rho, L3, Dir.Y)
moi4_link = link_moi(outer_radius, inner_radius, rho, L4, Dir.X)
moi5_link_original = link_moi(outer_radius, inner_radius, rho, L5_original, Dir.Y)

G1_link = moi_m_to_inertia(moi1_link, m1_link)
G2_link = moi_m_to_inertia(moi2_link, m2_link)
G3_link = moi_m_to_inertia(moi3_link, m3_link)
G4_link = moi_m_to_inertia(moi4_link, m4_link)
G5_link_original = moi_m_to_inertia(moi5_link_original, m5_link_original)


## Define variables to be filled in by functions
geared_params = {
    "g": np.array([0, 0, -9.81]),
    "Mlink_list": None,
    "Mrotor_list": None,
    "Glink": None,
    "Grotor": Grotor_list,
    "Alist": None,
    "Rlist": Rlist,
    "Slist": Slist,
    "torque_limits": torque_limits,
    "gear_ratios": None
}

def set_end_effector_mass(geared_params, ee_mass, extension):
    """Extension measured from end of link"""

    # measured from 5R
    com_extension = ((m5_link_original * L5_original / 2) + (ee_mass * (L5_original + extension))) / (m5_link_original + ee_mass)

    M_5R_final = mr.RpToTrans(np.eye(3), [com_extension, 0, 0])  # along x
    M_5Lorig_final = M_5L_4L_original @ mr.TransInv(M_5R_4L) @ M_5R_final
    M_6L_final = mr.RpToTrans(np.eye(3), [extension, 0, 0]) @ M_6L_5L_original @ M_5Lorig_final
    
    Gee = np.diag([0, 0, 0, ee_mass, ee_mass, ee_mass])
    
    Ad_5L = mr.Adjoint(M_5Lorig_final)
    Ad_ee = mr.Adjoint(M_6L_final)
    
    G5_link = (Ad_5L.T @ G5_link_original @ Ad_5L) + (Ad_ee.T @ Gee @ Ad_ee)
    M_6L_5L = M_6L_final
    M_5L_4L = mr.TransInv(M_5R_final) @ M_5R_4L
    
    geared_params["Mlink_list"] = np.array([M_1L_0L, M_2L_1L, M_3L_2L, M_4L_3L, M_5L_4L, M_6L_5L])
    geared_params["Mrotor_list"] = np.array([M_1R_0L, M_2R_1L, M_3R_2L, M_4R_3L, M_5R_4L])
    geared_params["Glink"] = np.array([G1_link, G2_link, G3_link, G4_link, G5_link])
    geared_params["Alist"] = get_Alist(geared_params["Mlink_list"], Slist)

def set_gear_ratios(geared_params, gear_ratios):
    geared_params["gear_ratios"] = np.array(gear_ratios)
