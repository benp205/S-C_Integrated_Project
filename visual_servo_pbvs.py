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

# --- 1. INITIALIZATION & SCENE SETUP ---
model, data, renderer, camera_id, enc_adr = sc.model, sc.data, sc.renderer, sc.camera_id, sc.enc_adr

cube_est = CubePositionEstimator()

SIDE = 0.05 * 6 / 8  # Physical size of the ArUco marker (meters)
TAG_ID = 7           # Target ArUco ID


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
gripper_actuator = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "robot_gripper_fingers_actuator")


# --- 1.1 Camera Calibration (Adapted from Week 3 Tutorial Parts B & C) ---
CHECKERBOARD = (9, 7)  # Inner corner dimensions (columns, rows) for 10x8 square grid
square_size = 0.020    # Physical side of one square in meters

# 3D corners in the board frame: the board is the plane Z = 0 (Tutorial Part C)
gx, gy = CHECKERBOARD
P_board = np.array([[i * square_size, j * square_size, 0.0] 
                    for j in range(gy) for i in range(gx)]).T

def measured_corners(im, grid_shape=CHECKERBOARD):
    """Sub-pixel checkerboard corners, returned as a 2 x N array (Tutorial Part C)."""
    g = im.mono().array if im.iscolor else im.array
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, c = cv2.findChessboardCorners(g, grid_shape, flags)
    if not ok:
        return None
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit)
    return c.reshape(-1, 2).T

def move_camera_to_look_at(target_cam_pos, board_center, viewer=None, max_steps=90):
    '''Uses resolved-rate IK to smoothly move the wrist camera to view the checkerboard center.'''
    for _ in range(max_steps):
        if viewer is not None and not viewer.is_running():
            break

        mj.mj_forward(model, data)
        enc_pos = getEncPosition(data, enc_adr)
        cam_pos, R_cv = CameratoWorld(enc_pos)
        
        # Desired optical Z-axis pointing from camera to board center
        z_des = board_center - cam_pos
        z_des = z_des / np.linalg.norm(z_des)
        
        # Orthonormal X and Y camera axes
        x_des = np.cross(np.array([0.0, 1.0, 0.0]), z_des)
        x_des = x_des / np.linalg.norm(x_des)

        y_des = np.cross(z_des, x_des)
        y_des = y_des / np.linalg.norm(y_des)
        
        R_des = np.column_stack([x_des, y_des, z_des])
        
        # Pose error in camera frame
        e_pos = R_cv.T @ (target_cam_pos - cam_pos)
        e_rot, _ = cv2.Rodrigues(R_cv.T @ R_des)
        e_rot = e_rot.ravel()
        
        if np.linalg.norm(e_pos) < 1e-3 and np.linalg.norm(e_rot) < 1e-2:
            break
            
        nu = np.r_[2.0 * e_pos, 2.0 * e_rot]
        dq = camera_twist_to_dq(nu, enc_pos, damping=1e-2)
        data.qpos[qadr] += dq * 0.03
        data.ctrl[:6] = data.qpos[qadr]

        # Sync the 3D viewer and camera feed so arm movement is smoothly animated
        if viewer is not None:
            viewer.sync()
            rgb_live = get_rgb()
            cv2.imshow("Wrist Camera - Calibration View", cv2.cvtColor(rgb_live, cv2.COLOR_RGB2BGR))
            cv2.waitKey(10)

def calibrate_camera(cam_id, viewer=None):
    '''
    Moves the arm smoothly across 6 viewpoints above the table checkerboard,
    calibrates intrinsics via CentralCamera.images2C (Week 3 Part B),
    and evaluates per-image reprojection error (Week 3 Part C).
    '''
    board_body_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, "calib_board")
    board_center = np.array([-0.10, 0.48, 0.002])
    if board_body_id != -1:
        model.body_pos[board_body_id] = [board_center[0], board_center[1], 0.001]

    # Temporarily park target_cube off to the side so it does not block the checkerboard
    cube_jnt = model.joint("target_cube_joint").qposadr[0]
    data.qpos[cube_jnt:cube_jnt + 3] = [0.35, 0.50, 0.025]
    mj.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()
        time.sleep(0.5)

    # 6 diverse 3D camera viewpoints looking at the checkerboard
    cam_viewpoints = [
        np.array([-0.10, 0.48, 0.32]),
        np.array([-0.18, 0.44, 0.30]),
        np.array([-0.02, 0.44, 0.30]),
        np.array([-0.16, 0.54, 0.28]),
        np.array([-0.04, 0.54, 0.28]),
        np.array([-0.10, 0.42, 0.35]),
    ]

    valid_images = []
    print("\n--- Starting Camera Calibration (Week 3 Tutorial Method) ---")

    for i, vp in enumerate(cam_viewpoints):
        if viewer is not None and not viewer.is_running():
            break

        # Smoothly glide the arm to the next calibration viewpoint
        move_camera_to_look_at(vp, board_center, viewer=viewer)
        mj.mj_forward(model, data)

        rgb = get_rgb()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bgr_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Wrap grayscale image in Machine Vision Toolbox Image object
        im = Image(gray)
        p_meas = measured_corners(im, CHECKERBOARD)

        if p_meas is not None:
            valid_images.append(im)
            # Draw measured corners on preview window
            corners_cv = p_meas.T.reshape(-1, 1, 2).astype(np.float32)
            cv2.drawChessboardCorners(bgr_vis, CHECKERBOARD, corners_cv, True)
            cv2.putText(bgr_vis, f"Pose {i+1}/6: DETECTED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(bgr_vis, f"Pose {i+1}/6: NOT DETECTED", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Wrist Camera - Calibration View", bgr_vis)
        cv2.waitKey(600)

    cv2.destroyWindow("Wrist Camera - Calibration View")

    # Slide the checkerboard to the side of the table after calibration finishes
    if board_body_id != -1:
        model.body_pos[board_body_id] = [0.32, 0.48, 0.001]
    mj.mj_forward(model, data)

    if len(valid_images) < 3:
        print(f"Warning: Only {len(valid_images)} valid views found. Falling back to analytical K.")
        return cam.K, None

    # --- Tutorial Part B: Run CentralCamera.images2C ---
    K_est, dist_est, frames = CentralCamera.images2C(
        valid_images, gridshape=CHECKERBOARD, squaresize=square_size
    )

    fu, fv = K_est[0, 0], K_est[1, 1]
    u0, v0 = K_est[0, 2], K_est[1, 2]
    W, H = valid_images[0].width, valid_images[0].height
    fov_h = 2 * np.arctan(W / (2 * fu))

    print(f"grid {CHECKERBOARD}: {len(frames)} of {len(cam_viewpoints)} images were used")
    print(f"f/rho_u = {fu:.1f} px,  f/rho_v = {fv:.1f} px")
    print(f"principal point = ({u0:.1f}, {v0:.1f}) px, image centre = ({W/2:.1f}, {H/2:.1f}) px")
    print(f"horizontal field of view = {np.degrees(fov_h):.1f} deg")
    print("distortion (k1, k2, p1, p2, k3) =", np.round(dist_est, 5))

    # --- Tutorial Part C: Compute Reprojection Error ---
    cal_cam = CentralCamera(f=K_est[0, 0], pp=(K_est[0, 2], K_est[1, 2]),
                            imagesize=(W, H), rho=1)
    rms_list = []
    for fr in frames:
        und = valid_images[fr.id].undistort(K_est, dist_est)
        p_meas = measured_corners(und, CHECKERBOARD)
        if p_meas is not None:
            cal_cam.pose = fr.pose
            p_proj = cal_cam.project_point(P_board)
            resid = p_meas - p_proj
            e = np.linalg.norm(resid, axis=0)
            rms = np.sqrt((e**2).mean())
            rms_list.append(rms)

    # Analytical focal length from MuJoCo FOV for comparison
    f_px_analytical = (480 / 2) / np.tan(np.deg2rad(model.cam_fovy[cam_id]) / 2)
    K_analytical = np.array([
        [f_px_analytical, 0,               640 / 2],
        [0,               f_px_analytical, 480 / 2],
        [0,               0,               1]
    ])

    print(f"\nMean Reprojection RMS Error: {np.mean(rms_list):.4f} pixels")
    print("Estimated Intrinsic Matrix (K_est):")
    print(np.round(K_est, 2))
    print("Analytical Intrinsic Matrix (K_analytical):")
    print(np.round(K_analytical, 2))
    print(f"Focal Length Error (fx): {abs(K_est[0, 0] - f_px_analytical):.2f} px\n")

    return K_est, dist_est


# --- 2. SENSOR & PERCEPTION FUNCTIONS ---
def get_rgb():
    '''Renders and returns the current RGB frame from the wrist-mounted camera.'''
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

def detect_tag_pose(rgb, K_matrix=None, dist_coeffs=None):
    '''
    Detects the ArUco tag, estimates its 3D pose relative to the camera via solvePnP,
    and returns rotation matrix R, translation vector t, and 2D pixel corners uv.
    '''
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or TAG_ID not in ids.ravel():
        return None
    sigma_px = 1.0 # Standard deviation of pixel measurement noise
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


# --- 3. KINEMATICS & CONTROL MAPPING (DECOUPLED FROM MUJOCO MODEL) ---
def camera_twist_to_dq(nu, q_est, damping=1e-3):
    '''
    Maps a camera-frame velocity twist (OpenCV axes: z-forward, y-down) 
    to arm joint velocities using purely estimated/measured joint angles (q_est),
    analytical forward kinematics, and damped least squares.
    '''
    _, cv_to_world = CameratoWorld(q_est)
    twist = np.concatenate([cv_to_world @ nu[:3], cv_to_world @ nu[3:]])
    
    # Analytical 6x6 Jacobian evaluated at q_est (completely independent of MuJoCo model)
    J = CameraJacobian(q_est)
    
    # Damped least-squares pseudo-inverse solve: J^T * (J * J^T + damping * I)^-1 * twist
    return J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)


# --- 4. MAIN PBVS & PICK-AND-PLACE STATE MACHINE ---
def run_pbvs(
    lam=0.5, niter=4000, vmax=0.05, nsub=10, tol=0.010,
    blind_descent=0.15, blind_ascent=0.15, blind_speed=0.05, close_steps=30,
    return_steps=150,   # Number of steps to glide back home
    max_tag_dropout=0.5, # Max time (seconds) to coast on EKF when tag is lost during TRACKING
    controller=None,    # JointStateController; if None, q_target is sent straight to the actuators
):
    dt = nsub * model.opt.timestep
    history, first_rgb = [], None
    last_valid_z = None
    last_valid_rot = None
    last_valid_uv = None
    tag_lost_time = 0.0
    
    state = "TRACKING"
    steps_remaining = 0
    return_step_count = 0
    q_lift_end = None
    
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
        # 1. Run camera calibration routine inside the active viewer
        K_calib, dist_calib = calibrate_camera(camera_id, viewer=viewer)

        # 2. Reset robot arm to home tracking position and spawn cube in viewable workspace
        data.qpos[:len(sc.home_qpos)] = sc.home_qpos
        data.ctrl[:len(sc.home_ctrl)] = sc.home_ctrl
        mj.mj_forward(model, data)

        # Ensure the randomly spawned cube starts inside the camera's FOV
        for _ in range(20):
            if detect_tag_pose(get_rgb(), K_matrix=K_calib, dist_coeffs=dist_calib) is not None:
                break
        viewer.sync()

        enc_pos = getEncPosition(data, enc_adr)
        q_target = enc_pos.copy()
        q_init = q_target.copy()  # Save original starting joint configuration
        if controller is not None:
            controller.reset(q_target)  # Observer/command start at the measured home pose

        k = 0

        xc = np.zeros(6)
        Pc = np.diag([0.5**2] * 3 + [0.05**2] * 3) # 0.5m std dev for position, 0.05m/s std dev for speed, high initial uncertainty
        
        cube_log = []
        nis_cube = []

        cube_stopped = False
        cube_vel_stop = 0.002 # cube velocity under 0.002m/s to account for noise, may need tuning
        still_frames = 100    # cube velocity under 0.002m/s for 100 frames
        still_count = 0

        path_start_time = data.time

        while viewer.is_running():
            if k >= niter and state == "TRACKING":
                print("PBVS iteration limit reached before convergence.")
                break

            for _ in range(nsub):
                time_cube = data.time - path_start_time # data.time starts with the sim and includes the camera calibration, readjust so cube time = 0
                if time_cube >= total_duration:
                    data.eq_active[0] = 0 # remove cube and mocap weld so cube stops moving and can be picked up

                # make the cube move along the random spline
                data.mocap_pos[0, 0] = spline_x(time_cube)
                data.mocap_pos[0, 1] = spline_y(time_cube)
                data.mocap_pos[0, 2] = 0.025 # constant for table height
                data.mocap_quat[0] = mocap_yaw # single random yaw

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
            cv2.waitKey(1)

            z_cube = None

            if seen is not None: # only run measurement update if tag is visible
                z_cube = cam_pos + cv_to_world @ seen[1] # reexpress tag position (cube position) from seen t in world frame

            xc, Pc, nis_c = cube_est.step_cube(xc, Pc, z_cube, dt) # EKF step (predicts even when z_cube is None)

            est_cube_vel = np.linalg.norm(xc[3:5]) # cube's estimated velocity (not including z)

            cube_slow = True if est_cube_vel < cube_vel_stop else False # check if velocity is below threshold

            # Kept original accumulation behavior so single encoder-noise spikes don't prevent descent
            still_count += 1 if cube_slow == True else 0

            cube_stopped = True if still_count >= still_frames else False # if cube is stopped for a certain frame count

            if state == "TRACKING":
                cube_log.append((data.time, xc.copy(), data.body("target_cube").xpos.copy())) # only log in tracking state
                nis_cube.append(nis_c)

            # --- STATE 1: VISUAL SERVOING TRACKING ---
            if state == "TRACKING":
                if seen is None: # Tag is not visible in the current frame
                    tag_lost_time += dt # Increment the time the tag has been lost 
                    if last_valid_z is not None and last_valid_z < 0.10: # If the last known z position is below 10 cm, initiate blind descent
                        steps_remaining = max(1, int(np.ceil(blind_descent / (blind_speed * dt))))
                        print(f"Tag lost at z={last_valid_z:.3f}m. Starting blind descent...")
                        state = "DESCENDING"
                        continue
                    elif last_valid_rot is not None and tag_lost_time < max_tag_dropout: # If the tag has been lost for less than the maximum allowed dropout time, use the last known rotation and UV coordinates
                        # Coast on EKF prediction and last known orientation if dropout < 0.5s
                        rotation = last_valid_rot
                        uv_noisy = last_valid_uv
                    else:
                        print(f"Tag lost for {tag_lost_time:.2f}s (>= {max_tag_dropout}s) or before first fix; aborting.")
                        break
                else:
                    tag_lost_time = 0.0 # Reset the tag lost timer since the tag is currently visible
                    rotation, t, uv_noisy = seen  # Unpack current tag rotation matrix, translation vector, and 2D pixel corners
                    last_valid_rot = rotation
                    last_valid_uv = uv_noisy

                est_cube_pos = cv_to_world.T @ (xc[:3] - cam_pos) # reexpress EKF cube world coordinates in camera frame

                est_vel_cube = xc[3:6] # get estimated cube velocity from KF
                est_vel_cam = cv_to_world.T @ est_vel_cube # cube velocity in camera frame

                last_valid_z = est_cube_pos[2]  # Save current camera-frame depth (Z coordinate) to detect when the tag is close to the table

                t_star, r_star = T_star.t, T_star.R  # Extract target position and rotation from the desired pose T_star
                e_t = est_cube_pos - t_star  # Compute 3D translational error vector (current position minus desired position)
                e_theta = SO3(rotation @ r_star.T, check=False).log(twist=True)  # Compute 3D rotational error vector (axis-angle representation)

                omega = lam * e_theta  # Compute proportional angular velocity command scaled by visual gain (lam)
                v = est_vel_cam + (lam * e_t) + np.cross(est_cube_pos, omega)  # Compute linear velocity command, incorporating rotational coupling via cross product, added velocity feed forward est_vel_cam from KF
                nu = np.r_[v, omega]  # Concatenate linear (v) and angular (omega) components into a full 6D camera twist vector
                nu[:3] *= min(1, vmax / (np.linalg.norm(nu[:3]) + 1e-9))  # Clamp linear velocity magnitude to max speed limit (vmax) for safety

                q_target += camera_twist_to_dq(nu, q_est) * dt  # Map camera twist to joint velocities using estimated state
                command_arm(q_target, enc_pos)

                error_norm = np.linalg.norm(e_t)  # Compute the Euclidean magnitude (distance) of the 3D position error
                if uv_noisy is not None:
                    history.append((uv_noisy, error_norm))  # Store current 2D pixel corners and error norm for debugging/plotting later

                if error_norm < tol and cube_stopped:
                    steps_remaining = max(1, int(np.ceil(blind_descent / (blind_speed * dt))))
                    print(f"Converged (error: {error_norm:.4f}m). Descending to grasp...")
                    valid_nis = [x for x in nis_cube if x is not None]
                    if valid_nis:
                        print(f"Cube NIS Mean: {np.mean(valid_nis):.4f}")
                    state = "DESCENDING"

            # --- STATE 2: BLIND DESCENT INTO GRASP POSITION ---
            elif state == "DESCENDING":
                nu = np.zeros(6)
                nu[:3] = cv_to_world.T @ np.array([0, 0, -blind_speed])

                q_target += camera_twist_to_dq(nu, q_est) * dt
                command_arm(q_target, enc_pos)
                steps_remaining -= 1

                if steps_remaining == 0:
                    steps_remaining = max(1, close_steps)
                    print("Descent complete. Closing gripper...")
                    state = "GRASPING"

            # --- STATE 3: ACTIVATE GRIPPER ---
            elif state == "GRASPING":
                data.eq_active[0] = 0 # remove cube and mocap weld so cube stops moving and can be picked up
                command_arm(q_target, enc_pos)  # Keep holding the pose through the controller
                data.ctrl[gripper_actuator] = 255
                steps_remaining -= 1

                if steps_remaining == 0:
                    steps_remaining = max(1, int(np.ceil(blind_ascent / (blind_speed * dt))))
                    print(f"Gripper closed. Lifting object by {blind_ascent:.3f}m...")
                    state = "LIFTING"

            # --- STATE 4: BLIND LIFT / ASCENT ---
            elif state == "LIFTING":
                nu = np.zeros(6)
                nu[:3] = cv_to_world.T @ np.array([0, 0, blind_speed])

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

                # Smooth interpolation (LERP) from lift-end pose back to starting pose
                q_target = (1 - alpha) * q_lift_end + alpha * q_init
                command_arm(q_target, enc_pos)
                data.ctrl[gripper_actuator] = 255  # Maintain object grasp

                if alpha >= 1.0:
                    print("Returned to home position. Holding.")
                    state = "HOLDING"

            # --- STATE 6: HOLD POSITION ---
            elif state == "HOLDING":
                command_arm(q_target, enc_pos)  # Keep holding the pose through the controller
                data.ctrl[gripper_actuator] = 255

            k += 1

    return history, first_rgb, cube_log


# --- 5. VISUALIZATION ---
def plot_on_image(history, rgb):
    '''Plots the ArUco marker corner trajectories over the initial camera frame.'''
    plt.figure(figsize=(9, 7))
    plt.imshow(rgb, alpha=0.55)
    for i in range(4):
        plt.plot([h[0][0, i] for h in history], [h[0][1, i] for h in history], color=f"C{i}", lw=2)
        plt.plot(history[0][0][0, i], history[0][0][1, i], "o", color=f"C{i}")
    plt.xlim(0, 640); plt.ylim(480, 0)
    plt.title("PBVS Tag Corner Tracks (o = start)"); plt.axis("off"); plt.show()


if __name__ == "__main__":
    try:
        history, first_rgb, cube_log = run_pbvs()
        print(f"{len(history)} visual tracking steps executed.")
        if history:
            plot_on_image(history, first_rgb)
    finally:
        renderer.close()