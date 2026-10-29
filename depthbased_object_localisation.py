# Libraries
import numpy as np
import matplotlib.pyplot as plt
import scene_setup as sc

# Intended use pipeline:
# Retrieve depth map --> Generate point cloud --> Report depths using point cloud-specified coordinates

# Camera specs
model, data, renderer, camera_id = sc.model, sc.data, sc.renderer, sc.camera_id

def get_depth_map():
    '''Renders and returns the current depth map from the wrist-mounted camera.'''
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth_true = renderer.render() # Depth should be 2D float array corresponding to depth reading of every pixel in meters
    renderer.disable_depth_rendering()

    # Add noise here (from week 4 tut part 2)
    sigma = 0.0015 + 0.0022 * depth_true ** 2           # sigma_d grows with d^2
    depth_map = depth_true + np.random.normal(0, 1, depth_true.shape) * sigma

    return depth_map

def point_cloud(depth_map, K, T_RC):
    '''Returns pixel-to-depth correspondence grid matrix + 3D cartesian point cloud in both camera and robot frames'''

    # Camera instrinsics
    H, W = depth_map.shape[0], depth_map.shape[1] # should be H = 480 px, W = 640 px

    uu, vv = np.meshgrid(np.arange(W), np.arange(H)) # Pixel grid
    valid = np.isfinite(depth_map) & (depth_map > 0) # 2d mask to filter invalid depth values

    # Referencing week 4 tut part 2
    uu_valid = uu.ravel()[valid]
    vv_valid = vv.ravel()[valid]
    d_valid = depth_map.ravel()[valid]

    z_rgbd = np.stack([uu_valid, vv_valid, d_valid], axis=1) # z_rgbd = [u v d]^⊤

    # Apply p_C = d K^(−1) [u v 1]^⊤ to define 3D cartesian points in physical space
    uv1 = np.stack([uu_valid, vv.valid, np.ones(valid.sum())], axis=1)
    P_C = (np.linalg.inv(K) @ uv1) * d_valid

    # Convert to robot frame (right now, robot pose is from other files)
    P_R = T_RC * P_C

    return P_C.T, P_R.T, z_rgbd

# Adapted from week 4 tut setup
def display_depth(points, colours=None, title='', frame='camera', ax=None,
               elev=22, azim=-75, size=1, nmax=20000, seed=0):
    '''Reports smallest depth recorded and displays point cloud in camera (P_C)/robot frame (P_R)'''

    P = np.asarray(points)
    C = colours
    per_point = (C is not None and not isinstance(C, str)
                 and np.asarray(C).ndim >= 1 and np.asarray(C).shape[0] == P.shape[0])
    if per_point:
        C = np.asarray(C)
    if len(P) > nmax:
        k = np.random.default_rng(seed).choice(len(P), nmax, replace=False)
        P = P[k]
        if per_point:
            C = C[k]

    if frame == 'camera':          # Camera: x right, y up, -z forward
        print(f"Minimum depth detected: {P[2].min()}")
        A, B, Cc = P[:, 0], P[:, 2], -P[:, 1]
        labels = ('X right [m]', '-Z forward [m]', 'Y up [m]')
    else:                          # Robot: x forward, y left, z up ***
        A, B, Cc = P[:, 0], P[:, 1], P[:, 2]
        labels = ('x fwd [m]', 'y left [m]', 'z up [m]') 

    if ax is None:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(projection='3d')
    ax.scatter(A, B, Cc, c=C, s=size, marker='.', linewidths=0)
    ax.set_xlabel(labels[0]); ax.set_ylabel(labels[1]); ax.set_zlabel(labels[2])
    ax.set_box_aspect(np.maximum([np.ptp(A), np.ptp(B), np.ptp(Cc)], 1e-3))
    ax.view_init(elev=elev, azim=azim)
    ax.set_title(title)
    return ax

