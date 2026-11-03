import numpy as np

class CubePositionEstimator: # kalman filter
    def f_cube(self, xh, dt):
        # state transition function for constant velocity model
        px, py, pz, vx, vy, vz = xh
        return np.array([px + vx * dt, py + vy * dt, pz + vz * dt, vx, vy, vz])

    def F_cube(self, dt):
        # state transition matrix for constant velocity model
        return np.array([[1, 0, 0, dt, 0, 0], # 1px + 0py + 0pz + dt*vx + 0vy + 0vz
                         [0, 1, 0, 0, dt, 0], # 0px + 1py + 0pz + 0vx + dt*vy + 0vz
                         [0, 0, 1, 0, 0, dt], # 0px + 0py + 1pz + 0vx + 0vy + dt*vz
                         [0, 0, 0, 1, 0, 0],  # 0px + 0py + 0pz + 1vx + 0vy + 0vz
                         [0, 0, 0, 0, 1, 0],  # 0px + 0py + 0pz + 0vx + 1vy + 0vz
                         [0, 0, 0, 0, 0, 1]]) # 0px + 0py + 0pz + 0vx + 0vy + 1vz

    def h_camera(self, xh):
        # get position, ignoring velocity
        px, py, pz, vx, vy, vz = xh
        return np.array([px, py, pz])

    def H_camera(self):
        # get position measurement matrix
        return np.array([[1, 0, 0, 0, 0, 0],  # 1px + 0py + 0pz + 0vx + 0vy + 0vz
                         [0, 1, 0, 0, 0, 0],  # 0px + 1py + 0pz + 0vx + 0vy + 0vz
                         [0, 0, 1, 0, 0, 0]]) # 0px + 0py + 1pz + 0vx + 0vy + 0vz

    def step_cube(self, xh, P, z, dt):

        '''
        Inputs:
        xh: current state estimate [px, py, pz, vx, vy, vz] in world frame in m, m/s
        P: state covariance
        z: measured cube position in world frame in m
        dt: time step in seconds
        Outputs:
        returns updated xh, P, and NIS
        '''

        sigma_p = 0.0001 # 0.1mm std dev position
        sigma_v = 0.0008 # 0.8mm/s std dev velocity

        # dynamic scaling of process noise based on estimated cube velocity (to account for higher uncertainty at higher speeds)
        if xh is not None:
            est_cube_vel = np.linalg.norm(xh[3:5]) # estimated velociy from XY not Z
            scale = est_cube_vel / 0.02
            if scale < 1:
                scale = 1 # stop scale dropping below tuned speeds
            sigma_p = scale * sigma_p
            sigma_v = scale * sigma_v

        Qd = np.diag([sigma_p**2, sigma_p**2, (0.1 * sigma_p)**2, # position components, z is lower since the cube is on a flat surface so we only have to account for noise
                      sigma_v**2, sigma_v**2, (0.1 * sigma_v)**2]) # velocity components

        # measurement noise matrix from tuning over multiple runs, targeting 95% within 2sigma bounds across all dimensions
        sigma_x = 0.00084 # std dev in x
        sigma_y = 0.00084 # std dev in y
        sigma_z = 0.00172 # std dev in z (depth is noisier)

        Rn = np.diag([sigma_x**2, sigma_y**2, sigma_z**2])

        # prediction step
        F = self.F_cube(dt) # state transition matrix
        xh = self.f_cube(xh, dt) # predicted state
        P = F @ P @ F.T + Qd # project covariance one step forward

        nis = None

        # only update if we have a valid measurement (z is not none and not nan)
        if z is not None and not np.isnan(z).any():
            y_hat = self.h_camera(xh) # predicted measurement
            H = self.H_camera() # measurement matrix

            S = H @ P @ H.T + Rn # estimated covariance of the innovation
            K = P @ H.T @ np.linalg.inv(S) # kalman gain matrix

            nu = z - y_hat # difference between measurement and prediction, innovation
            xh = xh + K @ nu # update state estimate with measurement

            P = (np.eye(len(xh)) - K @ H) @ P # update covariance

            nis = (nu.T @ np.linalg.inv(S) @ nu).item() # normalised innovation squared = v^T S^-1 v

        P = (P + P.T) / 2 # ensure P is symmetric

        return xh, P, nis

