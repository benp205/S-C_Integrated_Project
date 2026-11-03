# This file is a copy of the PBVS file without the robot tracking movement, it sits hovering ~25cm about the cube/table and logs the cube position and estimated positions for KF performance reporting purposes

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
from depthbased_object_localisation import get_depth_map, point_cloud

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
    #print("\n--- Starting Camera Calibration (Week 3 Tutorial Method) ---")

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


# Modified PBVS for KF testing, static robot in hover position and moving cube in frame
def kf_test(nsub=10, controller=None):

    dt = nsub * model.opt.timestep
    history, first_rgb = [], None
    
    data.ctrl[gripper_actuator] = 0  # Open gripper initially
    data.eq_active[0] = 1

    spline_x, spline_y, total_duration, mocap_yaw = generate_random_spline_path(start_pos=(-0.22, 0.50), x_bounds=(-0.22, 0.02), y_bounds=(0.52, 0.62))

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # 1. Run camera calibration routine inside the active viewer
        K_calib, dist_calib = calibrate_camera(camera_id, viewer=viewer)

        data.qpos[7:13] = np.array([-1.63822065, -1.46768637, 1.84389853, -1.90727519, -1.57281211, -0.06709918]) # robot centred at hover for PBVS (taken from a PBVS on centred cube)
        data.ctrl[0:6] = np.array([-1.63822065, -1.46768637, 1.84389853, -1.90727519, -1.57281211, -0.06709918])
        mj.mj_forward(model, data)
        viewer.sync()

        enc_pos = getEncPosition(data, enc_adr)
        q_target = enc_pos.copy()
        if controller is not None:
            controller.reset(q_target)  # Observer/command start at the measured home pose

        xc = np.array([-0.10, 0.50, 0.05, 0.0, 0.0, 0.0]) # centre of camera view approx
        Pc = np.diag([0.5**2] * 3 + [0.05**2] * 3) # high initial uncertainty
        
        cube_log = []
        nis_cube = []

        path_start_time = data.time

        x0, y0 = spline_x(0.0), spline_y(0.0)
        data.mocap_pos[0] = [x0, y0, 0.025]                      # carrier at the path start
        data.mocap_quat[0] = mocap_yaw
        cube_jnt = model.joint("target_cube_joint").qposadr[0]
        cube_dof = model.joint("target_cube_joint").dofadr[0]
        data.qpos[cube_jnt:cube_jnt + 3] = [x0, y0, 0.025]       # cube ON the carrier
        data.qpos[cube_jnt + 3:cube_jnt + 7] = mocap_yaw         # same rotation (no snap)
        data.qvel[cube_dof:cube_dof + 6] = 0                     # not moving
        mj.mj_forward(model, data)

        while viewer.is_running():
            for _ in range(nsub):
                time_cube = data.time - path_start_time # data.time starts with the sim and includes the camera calibration, readjust so cube time = 0
                if time_cube >= total_duration:
                    data.eq_active[0] = 0 # remove cube and mocap weld so cube stops moving and can be picked up
                    print("Cube path complete")
                    return cube_log, nis_cube

                # make the cube move along the random spline
                data.mocap_pos[0, 0] = spline_x(time_cube)
                data.mocap_pos[0, 1] = spline_y(time_cube)
                data.mocap_pos[0, 2] = 0.025 # constant for table height
                data.mocap_quat[0] = mocap_yaw # single random yaw

                mj.mj_step(model, data)
            viewer.sync()

            # Read noisy joint encoders and extract observer's estimated joint positions
            enc_pos = getEncPosition(data, enc_adr)

            rgb = get_rgb()
            first_rgb = rgb if first_rgb is None else first_rgb
            seen = detect_tag_pose(rgb, K_matrix=K_calib, dist_coeffs=dist_calib)

            cv2.imshow("Wrist Camera - Pick View", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

            depth_map = get_depth_map()

            z_cube = None
            t_depth = None

            cam_pos_kf, cv_to_world_kf = CameratoWorld(enc_pos) # do not feed q_est into KF, it is tuned for noisy encoder readings
            
            if seen is not None: # only run measurement update if tag is visible
                uv_noisy = seen[2] # aruco tag corners
                u_f = np.mean(uv_noisy[0]) # centre of aruco tag
                v_f = np.mean(uv_noisy[1])
                u_centre = int(np.round(u_f)) #must be a whole number for the pixel coordinate
                v_centre = int(np.round(v_f))

                H, W = depth_map.shape # frame size, should be 640x480

                t_cam = seen[1].copy() #translation

                if 0 <= u_centre < W and 0 <= v_centre < H: # double check that cube centre is valid in the frame
                    t_depth = depth_map[v_centre, u_centre]

                    if t_depth > 0:
                        uv1_centre = np.array([u_centre, v_centre, 1])
                        t_cam = t_depth * (np.linalg.inv(K_calib) @ uv1_centre) # from W4 lecture slides
                
                z_cube = cam_pos_kf + cv_to_world_kf @ t_cam  # Re-express tag position in world frame

            xc, Pc, nis_c = cube_est.step_cube(xc, Pc, z_cube, dt) # KF step (predicts even when z_cube is None)
            if nis_c is not None:
                nis_cube.append(nis_c)

            if nis_c is None:
                print("nis is none")

            if seen is None:
                print("seen is none")

            cube_log.append((data.time, xc.copy(), data.body("target_cube").xpos.copy(), Pc.copy(), seen is not None))

def plot_cube_tracking(cube_log):
    if not cube_log:
        print("No log data to plot!")
        return
    
    times = np.array([entry[0] for entry in cube_log])
    est_states = np.array([entry[1] for entry in cube_log]) # [x, y, z, vx, vy, vz] KF est positions and velocity
    true_poses = np.array([entry[2] for entry in cube_log]) # [x, y, z] ground truth positions
    est_pos = est_states[:, :3] # get X, Y, Z positions for plotting

    # --- FIGURE 1: Top-Down 2D Trajectory (X vs Y) ---
    plt.figure(figsize=(8, 7))
    plt.plot(true_poses[:, 0], true_poses[:, 1], 'k--', label='True Path (MuJoCo GT)', linewidth=2)
    plt.plot(est_pos[:, 0], est_pos[:, 1], 'b-', label='KF Estimated Path', linewidth=1.5)
    plt.xlabel('X Position [m]', fontsize=12)
    plt.ylabel('Y Position [m]', fontsize=12)
    plt.title('Cube Trajectory Tracking (Top-Down View)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

    # --- FIGURE 2: Time-Series Comparison (X, Y, Z vs Time) ---
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    labels = ['X Position [m]', 'Y Position [m]', 'Z Position [m]']

    for i in range(3):
        axes[i].plot(times, true_poses[:, i], 'k--', label='True', linewidth=1.5)
        axes[i].plot(times, est_pos[:, i], 'b-', label='Estimated', linewidth=1.5)
        axes[i].set_ylabel(labels[i], fontsize=10)
        axes[i].legend(loc='upper right', fontsize=9)
        axes[i].grid(True, linestyle=':', alpha=0.7)

    axes[-1].set_xlabel('Time [s]', fontsize=12)
    plt.suptitle('Position Tracking Over Time (True vs. KF Estimate)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

def cube_kf_metrics(cube_log, nis_cube, final_run):
    raw_t = np.array([row[0] for row in cube_log])
    t = raw_t - raw_t[0]
    xh = np.array([row[1] for row in cube_log])
    x_true = np.array([row[2] for row in cube_log])
    P_seq = np.array([row[3] for row in cube_log])
    tag_visible = np.array([row[4] for row in cube_log])

    for P in P_seq:
        assert np.allclose(P, P.T, atol=1e-10), "P is not symmetric"
        assert np.all(np.linalg.eigvalsh(P) > 0), "P is not positive definite"

    err = xh[:, :3] - x_true[:, :3] # est - gt
    err[:, 2] -= 0.0252 # -25mm for z top vs centre cube

    sig = np.sqrt(np.stack([P_seq[:, i, i] for i in range(3)], axis=-1))
    inside = np.mean(np.abs(err) <= 2 * sig, axis=0)

    print(f"Final position error magnitude: {np.linalg.norm(err[-1, :]):.4f} m")
    print(f"Fraction inside 2 sigma: x {inside[0]:.2f}, y {inside[1]:.2f}, z {inside[2]:.2f}  (about 0.95 is ideal)")
    print(f"Average NIS: {np.mean(nis_cube):.2f}")

    if final_run == True:

        fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        
        for i, (lab, scale) in enumerate((("x error [mm]", 1000), ("y error [mm]", 1000), ("z error [mm]", 1000))):
            ax = axs[i]
            ax.fill_between(t, -2 * scale * sig[:, i], 2 * scale * sig[:, i], 
                            color="tab:blue", alpha=0.2, label=r"$\pm 2\sigma$" if i == 0 else "")
            ax.plot(t, scale * err[:, i], color="tab:blue", lw=1, label="error" if i == 0 else "")
            
            ax.fill_between(t, *ax.get_ylim(), where=~tag_visible, 
                            color="0.5", alpha=0.25, step="mid", label="tag not seen" if i == 0 else "")
            
            ax.set_ylabel(lab)
            ax.grid(True, ls="--", alpha=0.5)

        axs[0].legend(loc="upper right", ncol=3)
        axs[-1].set_xlabel("time [s]")
        axs[0].set_title("Cube KF Estimation Error & Uncertainty Bounds")
        plt.tight_layout()
        plt.show()

    return inside[0], inside[1], inside[2], np.linalg.norm(err[-1, :])

if __name__ == "__main__":
    num_runs = 10
    all_nis = []
    all_x = []
    all_y = []
    all_z = []
    all_pos_error = []
    
    try:
        for run in range(num_runs):
            print(f"Run {run + 1} of {num_runs}")
            cube_log, nis_cube = kf_test()

            all_nis.extend(nis_cube)

            final_run = True if (run + 1) == num_runs else False

            frac_x, frac_y, frac_z, pos_err = cube_kf_metrics(cube_log, nis_cube, final_run)

            all_x.append(frac_x)
            all_y.append(frac_y)
            all_z.append(frac_z)
            all_pos_error.append(pos_err)

            cv2.destroyAllWindows()
    finally:
        renderer.close()

    all_nis_avg = np.mean(all_nis)
    print("################################################################################################################################")
    print(f"Over {num_runs} runs | fraction inside 2 sigma: x {np.mean(all_x):.2f}, y {np.mean(all_y):.2f}, z {np.mean(all_z):.2f}")
    print(f"                     | Avg NIS: {all_nis_avg:.2f} | Avg position error magnitude: {np.mean(all_pos_error):.4f}")
    print("################################################################################################################################")
    plot_cube_tracking(cube_log)
