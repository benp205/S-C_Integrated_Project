from spatialmath import SE3
import numpy as np
from scipy.interpolate import CubicSpline

def getEncPosition(data, enc_adr, enc_noise_std=0.001):
    enc_pos_true = data.sensordata[enc_adr : enc_adr + 6]
    return enc_pos_true + np.random.normal(0, enc_noise_std, 6)

def CameratoWorld(enc_pos): #replacement for cam_xpos
    #values from universal robotics website for the UR5e
    T_baseto1 = SE3.Rz(enc_pos[0]) * SE3.Tz(0.1625) * SE3.Rx(np.pi / 2)
    T_1to2 = SE3.Rz(enc_pos[1]) * SE3.Tx(-0.425)
    T_2to3 = SE3.Rz(enc_pos[2]) * SE3.Tx(-0.3922)
    T_3to4 = SE3.Rz(enc_pos[3]) * SE3.Tz(0.1333) * SE3.Rx(np.pi / 2)
    T_4to5 = SE3.Rz(enc_pos[4]) * SE3.Tz(0.0997) * SE3.Rx(-np.pi / 2)
    T_5to6 = SE3.Rz(enc_pos[5]) * SE3.Tz(0.0996)
                                                                 #camera offset 0.120m - 0.0996m
    T = T_baseto1 * T_1to2 * T_2to3 * T_3to4 * T_4to5 * T_5to6 * SE3(0, -0.06, 0.0204)

    return T.t, T.R #return translation and rotation

def generate_random_spline_path(start_pos=(-0.22, 0.42), x_bounds=(-0.22, 0.02), y_bounds=(0.42, 0.58), num_waypoints=4, speed=0.01):
    #generates random path to follow at a rough speed
    #make the start of the path the same as the cube placement in the sim
    x_waypoints = np.zeros(num_waypoints)
    y_waypoints = np.zeros(num_waypoints)
    x_waypoints[0] = start_pos[0]
    y_waypoints[0] = start_pos[1]

    #random points within the bounds
    x_waypoints[1:] = np.random.uniform(x_bounds[0], x_bounds[1], num_waypoints - 1)
    y_waypoints[1:] = np.random.uniform(y_bounds[0], y_bounds[1], num_waypoints - 1)

    #speed per point, distance between points / speed
    seg_lengths = np.hypot(np.diff(x_waypoints), np.diff(y_waypoints)) #straight line length of each segment
    t_waypoints = np.r_[0, np.cumsum(seg_lengths / speed)] #total running time

    #make spline paths
    spline_x = CubicSpline(t_waypoints, x_waypoints, bc_type='natural')
    spline_y = CubicSpline(t_waypoints, y_waypoints, bc_type='natural')

    total_duration = t_waypoints[-1]

    yaw = np.random.uniform(0, np.pi / 4) #0-45 degree rotation
    mocap_yaw = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)] #rotate about z axis

    return spline_x, spline_y, total_duration, mocap_yaw