import mujoco as mj
import mujoco.viewer
import numpy as np
import cv2
import matplotlib.pyplot as plt
from machinevisiontoolbox import CentralCamera
from spatialmath import SE3
import scene_setup as sc

model, data, renderer, camera_id = sc.model, sc.data, sc.renderer, sc.camera_id

SIDE = 0.05 * 6 / 8 #black marker is 6 of the 8 texture cells across the 5 cm cube top
TAG_ID = 7

def tag_corners(s):
    '''Corners of a square tag of side s in the tag frame, in ArUco order
    (top-left, top-right, bottom-right, bottom-left). The tag z-axis points out of the face.'''
    return np.array([[-s/2,  s/2, 0],
                     [ s/2,  s/2, 0],
                     [ s/2, -s/2, 0],
                     [-s/2, -s/2, 0]]).T

#fovy gives the focal length in pixels; rho=1 so f is in pixels like the lab
f_px = (480 / 2) / np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2)

cam = CentralCamera(f=f_px, imagesize=(640, 480), rho=1)

T_star = SE3.Trans(0, 0.075, 0.25) * SE3.Rx(np.pi) #desired tag pose in the camera frame: 25 cm in front, facing the camera
p_star = cam.project_point(tag_corners(SIDE), objpose=T_star)

detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_1000))

arm_joints = [f"robot_{n}_joint" for n in ["shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3"]]
qadr = [model.joint(j).qposadr[0] for j in arm_joints]
dadr = [model.joint(j).dofadr[0] for j in arm_joints]
wrist_id = model.body("robot_wrist_3_link").id

def get_rgb():
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

def detect_tag(rgb):
    '''Returns the 2x4 pixel corners and the depth of each corner (from PnP), or None if the tag isn't seen.'''
    corners, ids, _ = detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if ids is None or TAG_ID not in ids.ravel():
        return None
    uv = corners[list(ids.ravel()).index(TAG_ID)].reshape(4, 2).T
    _, rvec, tvec = cv2.solvePnP(tag_corners(SIDE).T, uv.T, cam.K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
    R, _ = cv2.Rodrigues(rvec)
    return uv, (R @ tag_corners(SIDE) + tvec)[2]

def camera_twist_to_dq(nu, damping=1e-3):
    '''Map a camera-frame twist (OpenCV axes) to arm joint velocities.'''
    R_cv = data.cam_xmat[camera_id].reshape(3, 3) @ np.diag([1, -1, -1]) #MuJoCo cameras look down -z with y up, OpenCV is z forward y down
    twist = np.concatenate([R_cv @ nu[:3], R_cv @ nu[3:]])
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mj.mj_jac(model, data, jacp, jacr, data.cam_xpos[camera_id], wrist_id)
    J = np.vstack([jacp, jacr])[:, dadr]
    return J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), twist)

def run_ibvs(lam=0.375, niter=1200, vmax=0.025, nsub=10, tol=2.0):
    dt = nsub * model.opt.timestep
    q_target = data.qpos[qadr].copy()
    history, first_rgb = [], None
    with mujoco.viewer.launch_passive(model, data) as viewer:
        for k in range(niter):
            if not viewer.is_running():
                break
            for _ in range(nsub):
                mj.mj_step(model, data)
            viewer.sync()

            rgb = get_rgb()
            first_rgb = rgb if first_rgb is None else first_rgb
            seen = detect_tag(rgb)
            if seen is None:
                print("tag lost at step", k)
                break
            p, Z = seen

            J  = cam.visjac_p(p, Z)                      # 1. stacked image Jacobian (8 x 6)
            e  = (p_star - p).flatten(order="F")             # 2. feature error
            nu = lam * np.linalg.pinv(J) @ e                 # 3. camera velocity
            nu[:3] *= min(1, vmax / (np.linalg.norm(nu[:3]) + 1e-9))

            q_target += camera_twist_to_dq(nu) * dt          # 4. arm actuators are position servos
            data.ctrl[:6] = q_target
            history.append((p, np.linalg.norm(e)))
            if history[-1][1] < tol:
                break
    return history, first_rgb

def plot_on_image(history, rgb):
    plt.figure(figsize=(9, 7))
    plt.imshow(rgb, alpha=0.55)
    for i in range(4):
        plt.plot([h[0][0, i] for h in history], [h[0][1, i] for h in history], color=f"C{i}", lw=2)
        plt.plot(history[0][0][0, i], history[0][0][1, i], "o", color=f"C{i}")
    plt.plot(*np.c_[p_star, p_star[:, :1]], "k--", lw=2, label="desired p*")
    plt.xlim(0, 640); plt.ylim(480, 0)
    plt.legend(); plt.title("IBVS corner tracks (o = start)"); plt.axis("off"); plt.show()

if __name__ == "__main__":
    try:
        history, first_rgb = run_ibvs()
        print(f"{len(history)} steps, final error {history[-1][1]:.2f} px")
        plot_on_image(history, first_rgb)
    finally:
        renderer.close()
