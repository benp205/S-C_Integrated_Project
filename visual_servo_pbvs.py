import time
import mujoco as mj
import mujoco.viewer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from machinevisiontoolbox import CentralCamera, Image
from spatialmath import SE3, SO3
import scene_setup as sc
from cube_estimator import CubePositionEstimator
from utils import getEncPosition, CameratoWorld, generate_random_spline_path, CameraJacobian
from depthbased_object_localisation import get_depth_map
from camera_calibration import calibrate_camera, get_rgb, camera_twist_to_dq

# --- 1. INITIALIZATION & SCENE SETUP ---
model, data, renderer, camera_id, enc_adr = sc.model, sc.data, sc.renderer, sc.camera_id, sc.enc_adr

cube_est = CubePositionEstimator()

SIDE = 0.05 * 6 / 8  # Physical size of the ArUco marker (meters)
TAG_ID = 7           # Target ArUco ID
GRIPPER_CAM_OFFSET = 0.10  # 10cm vertical offset from wrist camera to gripper fingers


def tag_corners(s):
    '''Returns 3D corner coordinates of a square ArUco tag in its local frame.'''
    return np.array([[-s/2,  s/2, 0],
                     [ s/2,  s/2, 0],
                     [ s/2, -s/2, 0],
                     [-s/2, -s/2, 0]]).T


# Compute camera focal length in pixels using vertical FOV from MuJoCo model
f_px = (480 / 2) / np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
cam = CentralCamera(f=f_px, imagesize=(640, 480), rho=1)

# Desired 3D pose of the tag in the camera frame (hovering above the target)
T_star = SE3.Trans(0, 0.06, 0.25) * SE3.Rx(np.pi)

# Initialize ArUco detector
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))

# Robot joint and actuator mappings
arm_joints = [f"robot_{n}_joint" for n in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
qadr = [model.joint(j).qposadr[0] for j in arm_joints]
cube_qadr = model.joint("target_cube_joint").qposadr[0]
gripper_actuator = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "robot_gripper_fingers_actuator")

# Cache calibration across multiple evaluation runs so it only needs to calibrate once if desired
_cached_calib = None


# --- 2. SENSOR & PERCEPTION FUNCTIONS ---
def detect_tag_pose(rgb, K_matrix=None, dist_coeffs=None):
    '''
    Detects the ArUco tag, estimates its 3D pose relative to the camera via solvePnP,
    and returns rotation matrix R, translation vector t, and 2D pixel corners uv.
    '''
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or TAG_ID not in ids.ravel():
        return None
    sigma_px = 1.0  # Standard deviation of pixel measurement noise
    uv = corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2).T
    uv_noisy = uv + np.random.normal(scale=sigma_px, size=uv.shape)

    K_use = cam.K if K_matrix is None else K_matrix

    success, rvec, tvec = cv2.solvePnP(
        tag_corners(SIDE).T, uv_noisy.T, K_use, dist_coeffs,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )

    if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None

    R, _ = cv2.Rodrigues(rvec)
    if not np.isfinite(R).all():
        return None

    return R, tvec.reshape(3), uv_noisy


# --- 4. MAIN PBVS & DEPTH-GUIDED PICK-AND-PLACE STATE MACHINE ---
def run_pbvs(
    lam=0.5, niter=4000, vmax=0.10, nsub=10, tol=0.005,
    grasp_speed=0.05, lift_height=0.15, close_steps=30,
    return_steps=150,    # Number of steps to glide back home
    hold_steps=100,      # Number of steps to hold at home before closing window
    max_tag_dropout=0.5, # Max time (seconds) to coast on KF when tag is lost during TRACKING
    controller=None,     # JointStateController; if None, q_target is sent straight to the actuators
    reuse_calibration=True,
):
    global _cached_calib
    dt = nsub * model.opt.timestep
    history, first_rgb = [], None
    last_valid_rot = None
    last_valid_uv = None
    last_valid_depth = None
    tag_lost_time = 0.0

    state = "TRACKING"
    steps_remaining = 0
    return_step_count = 0
    hold_step_count = 0
    q_lift_end = None

    # Reset simulation state and re-enable mocap weld for multi-run evaluations
    mj.mj_resetData(model, data)
    data.eq_active[0] = 1
    data.ctrl[gripper_actuator] = 0  # Open gripper initially

    spline_x, spline_y, total_duration, mocap_yaw = generate_random_spline_path()

    def get_estimated_joints(enc_meas):
        '''Returns joint angles from the state controller's observer if available, else sensor reading.'''
        if controller is not None and controller.initialised:
            return controller.estimated_state()[:6]
        return enc_meas

    def command_arm(q_des, enc_meas):
        '''Send the desired joint pose to the arm, through the state controller when one is supplied.'''
        if controller is None:
            data.ctrl[:6] = q_des
        else:
            data.ctrl[:6] = controller.step(enc_meas, q_des)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 1. Run camera calibration routine (or reuse cached calibration on subsequent evaluation runs)
        if not reuse_calibration or _cached_calib is None:
            K_calib, dist_calib = calibrate_camera(camera_id, viewer=viewer)
            _cached_calib = (K_calib, dist_calib)
        else:
            K_calib, dist_calib = _cached_calib

        # 2. Reset robot arm to home tracking position and place cube at start of spline
        data.qpos[:len(sc.home_qpos)] = sc.home_qpos
        data.ctrl[:len(sc.home_ctrl)] = sc.home_ctrl
        start_pos = [float(spline_x(0.0)), float(spline_y(0.0)), 0.025]
        data.mocap_pos[0] = start_pos
        data.mocap_quat[0] = mocap_yaw
        data.qpos[cube_qadr:cube_qadr + 3] = start_pos
        data.qpos[cube_qadr + 3:cube_qadr + 7] = mocap_yaw
        data.qvel[:] = 0.0
        mj.mj_forward(model, data)
        viewer.sync()

        enc_pos = getEncPosition(data, enc_adr)
        q_target = enc_pos.copy()
        q_init = q_target.copy()  # Save original starting joint configuration
        if controller is not None:
            controller.reset(q_target)  # Observer/command start at the measured home pose

        k = 0
        # Initialize KF state with initial guess and large uncertainty
        xc = np.array([-0.10, 0.50, 0.05, 0.0, 0.0, 0.0]) # centre of frame approx initial guess, do not use ground truth
        Pc = np.diag([0.5**2] * 3 + [0.05**2] * 3) # Wide initial uncertainty

        cube_log = [] # Log of cube positions over time
        nis_cube = [] # Log of normalized innovation squared for the cube
        joint_log = [] # Log of joint positions over time

        cube_stopped = False
        cube_vel_stop = 0.002  # Cube velocity threshold (m/s)
        still_frames = 100     # Frames below threshold required to confirm cube stopped
        still_count = 0

        path_start_time = data.time

        cube_nis_log = True

        while viewer.is_running():
            if k >= niter and state == "TRACKING":
                print("PBVS iteration limit reached before convergence.")
                break

            for _ in range(nsub):
                time_cube = data.time - path_start_time
                if time_cube >= total_duration:
                    data.eq_active[0] = 0  # Release mocap weld so cube stops and can be picked up
                    t_eval = total_duration
                    cube_nis_log = False # Stop cube nis logging
                else:
                    t_eval = time_cube

                # Move cube along the random spline
                data.mocap_pos[0, 0] = spline_x(t_eval)
                data.mocap_pos[0, 1] = spline_y(t_eval)
                data.mocap_pos[0, 2] = 0.025
                data.mocap_quat[0] = mocap_yaw

                mj.mj_step(model, data)
            viewer.sync()

            # Read noisy joint encoders and extract observer's estimated joint positions
            enc_pos = getEncPosition(data, enc_adr)
            q_est = get_estimated_joints(enc_pos)
            cam_pos, cv_to_world = CameratoWorld(q_est)

            rgb = get_rgb()
            first_rgb = rgb if first_rgb is None else first_rgb
            seen = detect_tag_pose(rgb, K_matrix=K_calib, dist_coeffs=dist_calib)

            cv2.imshow("Wrist Camera - Pick View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), ord(' ')):
                print("Early exit requested by user.")
                break

            depth_map = get_depth_map()
            z_cube = None
            t_depth = None

            cam_pos_kf, cv_to_world_kf = CameratoWorld(enc_pos) # do not feed q_est into KF, it is tuned for noisy encoder readings

            if seen is not None:  # Only run measurement update if tag is visible
                uv_noisy = seen[2]  # ArUco tag corners
                u_centre = int(np.round(np.mean(uv_noisy[0])))
                v_centre = int(np.round(np.mean(uv_noisy[1])))

                H, W = depth_map.shape
                t_cam = seen[1].copy()  # Translation backup if t_depth is out of bounds

                if 0 <= u_centre < W and 0 <= v_centre < H:
                    t_depth = depth_map[v_centre, u_centre]
                    if t_depth is not None and np.isfinite(t_depth) and t_depth > 0:
                        last_valid_depth = float(t_depth)
                        uv1_centre = np.array([u_centre, v_centre, 1.0])
                        t_cam = t_depth * (np.linalg.inv(K_calib) @ uv1_centre)

                z_cube = cam_pos_kf + cv_to_world_kf @ t_cam  # Re-express tag position in world frame

            xc, Pc, nis_c = cube_est.step_cube(xc, Pc, z_cube, dt)
            if nis_c is not None and cube_nis_log == True: # Only log cube nis while cube is moving
                nis_cube.append(nis_c)

            est_cube_vel = np.linalg.norm(xc[3:5])
            cube_slow = est_cube_vel < cube_vel_stop
            still_count += 1 if cube_slow else 0
            cube_stopped = still_count >= still_frames

            # --- STATE 1: VISUAL SERVOING TRACKING ---
            if state == "TRACKING":
                # Log KF cube estimate vs ground truth while tracking the cube on the table
                true_cube_pos = data.qpos[cube_qadr:cube_qadr + 3].copy()
                cube_log.append((data.time - path_start_time, xc.copy(), true_cube_pos))

                if seen is None:
                    tag_lost_time += dt
                    if last_valid_rot is not None and tag_lost_time < max_tag_dropout:
                        rotation = last_valid_rot
                        uv_noisy = last_valid_uv
                    else:
                        print(f"Tag lost for {tag_lost_time:.2f}s (>= {max_tag_dropout}s) or before first fix; aborting.")
                        break
                else:
                    tag_lost_time = 0.0
                    rotation, t, uv_noisy = seen
                    last_valid_rot = rotation
                    last_valid_uv = uv_noisy

                est_cube_pos = cv_to_world.T @ (xc[:3] - cam_pos)  # KF cube world coordinates in camera frame

                est_vel_cube = xc[3:6].copy()
                est_vel_cube[2] = 0.0  # Ignore Z velocity for feedforward
                est_vel_cam = cv_to_world.T @ est_vel_cube

                t_star, r_star = T_star.t, T_star.R
                e_t = est_cube_pos - t_star
                e_theta = SO3(rotation @ r_star.T, check=False).log(twist=True)

                omega = lam * e_theta
                v = est_vel_cam + (lam * e_t) + np.cross(est_cube_pos, omega)
                nu = np.r_[v, omega]
                nu[:3] *= min(1.0, vmax / (np.linalg.norm(nu[:3]) + 1e-9))

                q_target += camera_twist_to_dq(nu, q_est) * dt
                command_arm(q_target, enc_pos)

                error_norm = float(np.linalg.norm(e_t))
                if uv_noisy is not None:
                    history.append((uv_noisy, error_norm))

                if error_norm < tol and cube_stopped:
                    # Use depth camera reading (or KF camera-frame Z) minus 10cm camera-to-gripper offset
                    measured_d = t_depth if (t_depth is not None and t_depth > 0) else (
                        last_valid_depth if last_valid_depth is not None else est_cube_pos[2]
                    )
                    descent_dist = max(0.02, float(measured_d) - GRIPPER_CAM_OFFSET)
                    steps_remaining = max(1, int(np.ceil(descent_dist / (grasp_speed * dt))))
                    print(f"Converged (error: {error_norm:.4f}m). Depth-guided descent ({descent_dist:.4f}m) to grasp...")
                    if nis_cube:
                        print(f"Cube NIS Mean: {np.mean(nis_cube):.4f}")
                    state = "DESCENDING"

            # --- STATE 2: DEPTH-GUIDED DESCENT INTO GRASP POSITION ---
            elif state == "DESCENDING":
                nu = np.zeros(6)
                nu[:3] = cv_to_world.T @ np.array([0.0, 0.0, -grasp_speed])

                q_target += camera_twist_to_dq(nu, q_est) * dt
                command_arm(q_target, enc_pos)
                steps_remaining -= 1

                if steps_remaining == 0:
                    steps_remaining = max(1, close_steps)
                    print("Descent complete. Closing gripper...")
                    state = "GRASPING"

            # --- STATE 3: ACTIVATE GRIPPER ---
            elif state == "GRASPING":
                data.eq_active[0] = 0
                command_arm(q_target, enc_pos)
                data.ctrl[gripper_actuator] = 255
                steps_remaining -= 1

                if steps_remaining == 0:
                    steps_remaining = max(1, int(np.ceil(lift_height / (grasp_speed * dt))))
                    print(f"Gripper closed. Lifting object by {lift_height:.3f}m...")
                    state = "LIFTING"

            # --- STATE 4: VERTICAL LIFT ---
            elif state == "LIFTING":
                nu = np.zeros(6)
                nu[:3] = cv_to_world.T @ np.array([0.0, 0.0, grasp_speed])

                q_target += camera_twist_to_dq(nu, q_est) * dt
                command_arm(q_target, enc_pos)
                data.ctrl[gripper_actuator] = 255
                steps_remaining -= 1

                if steps_remaining == 0:
                    print("Lift complete. Returning to initial home position...")
                    q_lift_end = q_target.copy()
                    state = "RETURNING"

            # --- STATE 5: RETURN HOME (LINEAR INTERPOLATION) ---
            elif state == "RETURNING":
                return_step_count += 1
                alpha = min(1.0, return_step_count / return_steps)

                q_target = (1.0 - alpha) * q_lift_end + alpha * q_init
                command_arm(q_target, enc_pos)
                data.ctrl[gripper_actuator] = 255

                if alpha >= 1.0:
                    print(f"Returned to home position. Holding for {hold_steps} steps...")
                    state = "HOLDING"

            # --- STATE 6: HOLD POSITION & EXIT ---
            elif state == "HOLDING":
                command_arm(q_target, enc_pos)
                data.ctrl[gripper_actuator] = 255
                hold_step_count += 1
                if hold_step_count >= hold_steps:
                    print("Hold complete. Closing trial window.")
                    break

            # Log joint tracking error: (time, q_measured, q_est, q_target, q_true)
            q_true = data.qpos[qadr].copy()
            joint_log.append((data.time - path_start_time, enc_pos.copy(), q_est.copy(), q_target.copy(), q_true))
            k += 1

        # Explicitly close the passive MuJoCo viewer when the loop ends
        viewer.close()

    cv2.destroyWindow("Wrist Camera - Pick View")
    cv2.waitKey(1)

    # Attach joint_log as an attribute on run_pbvs so evaluate_runs.py can plot LQR joint error
    # without breaking the 3-tuple unpacking in main.py
    run_pbvs.last_joint_log = joint_log
    run_pbvs.last_nis = nis_cube

    return history, first_rgb, cube_log


# --- 5. VISUALIZATION ---
def plot_on_image(history, rgb, save_path=None):
    '''Plots the ArUco marker corner trajectories over the initial camera frame.'''
    fig = plt.figure(figsize=(9, 7))
    plt.imshow(rgb, alpha=0.55)
    for i in range(4):
        plt.plot([h[0][0, i] for h in history], [h[0][1, i] for h in history], color=f"C{i}", lw=2)
        plt.plot(history[0][0][0, i], history[0][0][1, i], "o", color=f"C{i}")
    plt.xlim(0, 640)
    plt.ylim(480, 0)
    plt.title("PBVS Tag Corner Tracks (o = start)")
    plt.axis("off")
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    try:
        history, first_rgb, cube_log = run_pbvs()
        print(f"{len(history)} visual tracking steps executed.")
        if history:
            plot_on_image(history, first_rgb)
    finally:
        renderer.close()