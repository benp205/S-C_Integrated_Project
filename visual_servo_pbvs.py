import mujoco as mj
import mujoco.viewer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from machinevisiontoolbox import CentralCamera
from spatialmath import SE3, SO3
import scene_setup as sc

# --- 1. INITIALIZATION & SCENE SETUP ---
model, data, renderer, camera_id = sc.model, sc.data, sc.renderer, sc.camera_id

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
T_star = SE3.Trans(0, 0.06, 0.15) * SE3.Rx(np.pi)

# Initialize ArUco detector
detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))

# Robot joint and actuator mappings
arm_joints = [f"robot_{n}_joint" for n in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
qadr = [model.joint(j).qposadr[0] for j in arm_joints]
dadr = [model.joint(j).dofadr[0] for j in arm_joints]
wrist_id = model.body("robot_wrist_3_link").id
gripper_actuator = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, "robot_gripper_fingers_actuator")


# --- 2. SENSOR & PERCEPTION FUNCTIONS ---
def get_rgb():
    '''Renders and returns the current RGB frame from the wrist-mounted camera.'''
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

def detect_tag_pose(rgb):
    '''
    Detects the ArUco tag, estimates its 3D pose relative to the camera via solvePnP,
    and returns rotation matrix R, translation vector t, and 2D pixel corners uv.
    '''
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or TAG_ID not in ids.ravel():
        return None
    
    uv = corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2).T
    success, rvec, tvec = cv2.solvePnP(
        tag_corners(SIDE).T, uv.T, cam.K, None,
        flags=cv2.SOLVEPNP_IPPE_SQUARE,
    )
    
    if not success or not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
        return None
        
    R, _ = cv2.Rodrigues(rvec)
    if not np.isfinite(R).all():
        return None
        
    return R, tvec.reshape(3), uv


# --- 3. KINEMATICS & CONTROL MAPPING ---
def camera_twist_to_dq(nu, damping=1e-3):
    '''
    Maps a camera-frame velocity twist (OpenCV axes: z-forward, y-down) 
    to arm joint velocities using the analytical body Jacobian and damped least squares.
    '''
    # Transform OpenCV camera frame axes to MuJoCo world/body frame axes
    cv_to_world = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
    twist = np.concatenate([cv_to_world @ nu[:3], cv_to_world @ nu[3:]])
    
    # Compute robot Jacobian at the wrist camera frame
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mj.mj_jac(model, data, jacp, jacr, data.cam_xpos[camera_id], wrist_id)
    J = np.vstack([jacp, jacr])[:, dadr]
    
    # Damped least-squares pseudo-inverse solve: J^T * (J * J^T + damping * I)^-1 * twist
    return J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)


# --- 4. MAIN PBVS & PICK-AND-PLACE STATE MACHINE ---
def run_pbvs(
    lam=0.5, niter=1200, vmax=0.025, nsub=10, tol=0.005,
    blind_descent=0.05, blind_ascent=0.15, blind_speed=0.02, close_steps=30,
    return_steps=150,  # Number of steps to glide back home
):
    dt = nsub * model.opt.timestep
    q_target = data.qpos[qadr].copy()
    q_init = q_target.copy()  # Save original starting joint configuration
    
    history, first_rgb = [], None
    last_valid_z = None
    
    state = "TRACKING"
    steps_remaining = 0
    return_step_count = 0
    q_lift_end = None
    
    data.ctrl[gripper_actuator] = 0  # Open gripper initially

    with mujoco.viewer.launch_passive(model, data) as viewer:
        k = 0
        while viewer.is_running():
            if k >= niter and state == "TRACKING":
                print("PBVS iteration limit reached before convergence.")
                break

            for _ in range(nsub):
                mj.mj_step(model, data)
            viewer.sync()

            rgb = get_rgb()
            first_rgb = rgb if first_rgb is None else first_rgb
            seen = detect_tag_pose(rgb)

            # --- STATE 1: VISUAL SERVOING TRACKING ---
            if state == "TRACKING":
                if seen is None:
                    if last_valid_z is not None and last_valid_z < 0.10:
                        steps_remaining = max(1, int(np.ceil(blind_descent / (blind_speed * dt))))
                        print(f"Tag lost at z={last_valid_z:.3f}m. Starting blind descent...")
                        state = "DESCENDING"
                        continue
                    else:
                        print("Tag lost too far from target; aborting.")
                        break

                rotation, t, uv = seen
                last_valid_z = t[2]
                
                t_star, r_star = T_star.t, T_star.R
                e_t = t - t_star
                e_theta = SO3(rotation @ r_star.T).log(twist=True)
                
                omega = lam * e_theta
                v = lam * e_t + np.cross(t, omega)
                nu = np.r_[v, omega]
                nu[:3] *= min(1, vmax / (np.linalg.norm(nu[:3]) + 1e-9))

                q_target += camera_twist_to_dq(nu) * dt
                data.ctrl[:6] = q_target
                
                error_norm = np.linalg.norm(e_t)
                history.append((uv, error_norm))

                if error_norm < tol:
                    steps_remaining = max(1, int(np.ceil(blind_descent / (blind_speed * dt))))
                    print(f"Converged (error: {error_norm:.4f}m). Descending to grasp...")
                    state = "DESCENDING"

            # --- STATE 2: BLIND DESCENT INTO GRASP POSITION ---
            elif state == "DESCENDING":
                cv_to_world = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
                nu = np.zeros(6)
                nu[:3] = cv_to_world.T @ np.array([0, 0, -blind_speed])
                
                q_target += camera_twist_to_dq(nu) * dt
                data.ctrl[:6] = q_target
                steps_remaining -= 1

                if steps_remaining == 0:
                    steps_remaining = max(1, close_steps)
                    print("Descent complete. Closing gripper...")
                    state = "GRASPING"

            # --- STATE 3: ACTIVATE GRIPPER ---
            elif state == "GRASPING":
                data.ctrl[gripper_actuator] = 255
                steps_remaining -= 1

                if steps_remaining == 0:
                    steps_remaining = max(1, int(np.ceil(blind_ascent / (blind_speed * dt))))
                    print(f"Gripper closed. Lifting object by {blind_ascent:.3f}m...")
                    state = "LIFTING"

            # --- STATE 4: BLIND LIFT / ASCENT ---
            elif state == "LIFTING":
                cv_to_world = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1])
                nu = np.zeros(6)
                nu[:3] = cv_to_world.T @ np.array([0, 0, blind_speed])
                
                q_target += camera_twist_to_dq(nu) * dt
                data.ctrl[:6] = q_target
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
                data.ctrl[:6] = q_target
                data.ctrl[gripper_actuator] = 255  # Maintain object grasp
                
                if alpha >= 1.0:
                    print("Returned to home position. Holding.")
                    state = "HOLDING"

            # --- STATE 6: HOLD POSITION ---
            elif state == "HOLDING":
                data.ctrl[gripper_actuator] = 255

            k += 1

    return history, first_rgb


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
        history, first_rgb = run_pbvs()
        print(f"{len(history)} visual tracking steps executed.")
        if history:
            plot_on_image(history, first_rgb)
    finally:
        renderer.close()