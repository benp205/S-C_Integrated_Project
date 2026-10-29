import time
import mujoco as mj
import numpy as np
import cv2
from machinevisiontoolbox import CentralCamera, Image
import scene_setup as sc
from utils import getEncPosition, CameratoWorld, CameraJacobian

model, data, renderer, camera_id, enc_adr = sc.model, sc.data, sc.renderer, sc.camera_id, sc.enc_adr

arm_joints = [f"robot_{n}_joint" for n in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
qadr = [model.joint(j).qposadr[0] for j in arm_joints]
cube_qadr = model.joint("target_cube_joint").qposadr[0]

# Compute camera focal length in pixels using vertical FOV from MuJoCo model
f_px = (480 / 2) / np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)
cam = CentralCamera(f=f_px, imagesize=(640, 480), rho=1)


# --- Camera Calibration (Adapted from Week 3 Tutorial Parts B & C) ---
CHECKERBOARD = (9, 7)  # Inner corner dimensions (columns, rows) for 10x8 square grid
square_size = 0.020    # Physical side of one square in meters

# 3D corners in the board frame: the board is the plane Z = 0 (Tutorial Part C)
gx, gy = CHECKERBOARD
P_board = np.array([[i * square_size, j * square_size, 0.0]
                    for j in range(gy) for i in range(gx)]).T


def get_rgb():
    '''Renders and returns the current RGB frame from the wrist-mounted camera.'''
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()


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
    data.qpos[cube_qadr:cube_qadr + 3] = [0.35, 0.50, 0.025]
    mj.mj_forward(model, data)

    if viewer is not None:
        viewer.sync()
        time.sleep(0.5)

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

        move_camera_to_look_at(vp, board_center, viewer=viewer)
        mj.mj_forward(model, data)

        rgb = get_rgb()
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bgr_vis = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        im = Image(gray)
        p_meas = measured_corners(im, CHECKERBOARD)

        if p_meas is not None:
            valid_images.append(im)
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
    cv2.waitKey(1)

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
