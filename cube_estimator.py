import numpy as np

class CubePositionEstimator: #kalman filter (not EKF since it's linear)
    def f_cube(self, xh, dt):
        #constant velocity model
        px, py, pz, vx, vy, vz = xh
        return np.array([px + vx * dt, py + vy * dt, pz + vz * dt, vx, vy, vz])

    def F_cube(self, dt):
        #constant velocity model matrix
        return np.array([[1, 0, 0, dt, 0, 0], #1px + 0py + 0pz + dt*vx + 0vy + 0vz
                         [0, 1, 0, 0, dt, 0], #0px + 1py + 0pz + 0vx + dt*vy + 0vz
                         [0, 0, 1, 0, 0, dt], #0px + 0py + 1pz + 0vx + 0vy + dt*vz
                         [0, 0, 0, 1, 0, 0],  #0px + 0py + 0pz + 1vx + 0vy + 0vz
                         [0, 0, 0, 0, 1, 0],  #0px + 0py + 0pz + 0vx + 1vy + 0vz
                         [0, 0, 0, 0, 0, 1]]) #0px + 0py + 0pz + 0vx + 0vy + 1vz

    def h_camera(self, xh):
        #get position
        px, py, pz, vx, vy, vz = xh
        return np.array([px, py, pz])

    def H_camera(self):
        #get position measurement matrix
        return np.array([[1, 0, 0, 0, 0, 0],  #1px + 0py + 0pz + 0vx + 0vy + 0vz
                         [0, 1, 0, 0, 0, 0],  #0px + 1py + 0pz + 0vx + 0vy + 0vz
                         [0, 0, 1, 0, 0, 0]]) #0px + 0py + 1pz + 0vx + 0vy + 0vz

    def step_cube(self, xh, P, z, dt):
        sigma_p = 0.0001 #0.1mm std dev position
        sigma_v = 0.0008 #0.8mm/s std dev velocity

        if xh is not None:
            est_cube_vel = np.linalg.norm(xh[3:5])
            scale = est_cube_vel / 0.02
            if scale < 1:
                scale = 1 #stop it going below tuned values
            sigma_p = scale * sigma_p
            sigma_v = scale * sigma_v

        Qd = np.diag([sigma_p**2, sigma_p**2, (0.1 * sigma_p)**2, #position components, z is lower since z axis is stationary
                      sigma_v**2, sigma_v**2, (0.1 * sigma_v)**2]) #velocity components

        sigma_x = 0.00084 #std dev in x
        sigma_y = 0.00084 #std dev in y
        sigma_z = 0.00172 #std dev in z

        Rn = np.diag([sigma_x**2, sigma_y**2, sigma_z**2])

        F = self.F_cube(dt)
        xh = self.f_cube(xh, dt)
        P = F @ P @ F.T + Qd

        nis = None

        if z is not None and not np.isnan(z).any():
            y_hat = self.h_camera(xh)
            H = self.H_camera()

            S = H @ P @ H.T + Rn
            K = P @ H.T @ np.linalg.inv(S)

            nu = z - y_hat
            xh = xh + K @ nu

            P = (np.eye(len(xh)) - K @ H) @ P

            nis = (nu.T @ np.linalg.inv(S) @ nu).item()

        P = (P + P.T) / 2

        return xh, P, nis

