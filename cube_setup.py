def set_cube_pose(position, quat=None):
    """Set the cube free joint to an exact world-space pose."""
    position = np.asarray(position, dtype=float)
    cube_joint = model.joint('target_cube_joint').qposadr[0]

    if quat is None:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    quat = np.asarray(quat, dtype=float)
    quat = quat / np.linalg.norm(quat)

    data.qpos[cube_joint:cube_joint + 3] = position
    data.qpos[cube_joint + 3:cube_joint + 7] = quat
    mujoco.mj_forward(model, data)


def follow_spline_path(path, fps=10, viewer=None, loop=False):
    """Walk the cube through a spline-generated path, one waypoint per frame."""
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 3:
        raise ValueError("path must have shape (N, 3)")

    if len(path) == 0:
        return

    for i, pos in enumerate(path):
        if i < len(path) - 1:
            tangent = path[i + 1] - pos
        elif i > 0:
            tangent = pos - path[i - 1]
        else:
            tangent = np.array([1.0, 0.0, 0.0])

        tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
        yaw = np.arctan2(tangent[1], tangent[0])
        quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

        set_cube_pose(pos, quat)
        if viewer is not None:
            viewer.sync()
        if viewer is None:
            import time
            time.sleep(1.0 / fps)

    if loop:
        follow_spline_path(path, fps=fps, viewer=viewer, loop=False)


from scipy.interpolate import CubicSpline

def generate_fov_path(num_waypoints=5, fov_bounds=None, duration=30.0, fps=30):
    if fov_bounds is None:
        fov_bounds = {'x': (-0.22, 0.02), 'y': (0.42, 0.58), 'z': (0.025, 0.025)}
    
    # Generate random waypoints inside the FoV
    waypoints = np.column_stack([
        np.random.uniform(*fov_bounds['x'], num_waypoints),
        np.random.uniform(*fov_bounds['y'], num_waypoints),
        np.random.uniform(*fov_bounds['z'], num_waypoints)
    ])
    
    # Time points for waypoints
    t_waypoints = np.linspace(0, duration, num_waypoints)
    
    # Smooth Spline
    cs = CubicSpline(t_waypoints, waypoints, bc_type='clamped')
    
    # Query dense trajectory steps
    t_dense = np.linspace(0, duration, int(duration * fps))
    path = cs(t_dense)           # Positions (x, y, z)
    velocity = cs(t_dense, nu=1) # Velocities (vx, vy, vz)
    
    return t_dense, path, velocity


if __name__ == "__main__": #lets visual_servo import the scene without opening this viewer
  try:
    fps = 30
    path_speed = 0.5
    _, path, _ = generate_fov_path(duration=5.0, fps=fps)
    path_duration = 5.0 / path_speed

    start = time.time()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            elapsed = time.time() - start
            if elapsed >= path_duration:
                break

            mujoco.mj_step(model, data)

            #rgb
            renderer.update_scene(data, camera=camera_id)
            rgb_img = renderer.render()

            #depth
            renderer.enable_depth_rendering()
            renderer.update_scene(data, camera=camera_id)
            depth_img = renderer.render() #depth is float array in meters
            renderer.disable_depth_rendering()

            #rgb display output
            bgr_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR) #opencv expects BGR, passing rgb makes it look funky
            cv2.imshow("RealSense D435 - RGB", bgr_img)

            #depth display output
            depth_img -= depth_img.min() #copied from the mujoco tutorial code, don't know if we actually need this but it looks cool
            depth_img /= 2*depth_img[depth_img <= 1].mean()
            pixels = 255*np.clip(depth_img, 0, 1)
            depth_view = pixels.astype(np.uint8) 
            cv2.imshow("RealSense D435 - Depth", depth_view)

            i = min(int(elapsed * fps * path_speed), len(path) - 1)
            pos = path[i]

            yaw = np.arctan2(path[min(i + 1, len(path) - 1)][1] - pos[1],
                            path[min(i + 1, len(path) - 1)][0] - pos[0])
            quat = np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])

            set_cube_pose(pos, quat)

            viewer.sync()
            cv2.waitKey(1)
            viewer.sync()

  finally:
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    
    renderer.close()