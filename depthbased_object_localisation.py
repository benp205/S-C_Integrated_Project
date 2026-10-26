# libraries
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from machinevisiontoolbox import Image, CentralCamera
from spatialmath import SE3
import scene_setup as sc
# import visual_servo_pbvs as vsp

# ***Haven't been able test

# supposed to control extent of drop?
# identify cube --> how far (depth-wise) the camera is from the object

# model:
# z_rgbd = [u v d]^⊤, p_C = d K^(−1) [u v 1]^⊤
# output depth map? coordinates estimated?

# pipeline
# depth map --> point cloud + report depths

# Camera specs
model, data, renderer, camera_id = sc.model, sc.data, sc.renderer, sc.camera_id

# K_calib, __ = vsp.calibrate_camera(camera_id, viewer=viewer)

# get frame
def get_depth_map():
    '''Renders and returns the current depth map from the wrist-mounted camera.'''
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera=camera_id)
    depth_map = renderer.render() # depth is float array in meters
    renderer.disable_depth_rendering()

    # print(depth_map.dtype, depth_map.shape)
    print (f"Depth Range: {np.nanmin(depth_map)} to {np.nanmax(depth_map)}")

    return depth_map

def point_cloud(depth_map, K):
    '''Renders and returns the current depth map from the wrist-mounted camera.'''
    # Camera instrinsics
    W, H = depth_map.shape # may need to be switched
    f_px = (H / 2.0) / np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2) # camera lens focal length [px]

    fu, fv = K[0, 0], K[1, 1]
    u0, v0 = K[0, 2], K[1, 2]
    fov_h = 2 * np.arctan(W / (2 * fu))

    # Pixel grid
    uu, vv = np.meshgrid(np.arange(W), np.arange(H))

    # Depth to camera coordinates
    Z = depth_map
    X = (uu - u0) * depth_map / f_px
    Y = (vv - v0) * depth_map / f_px 

    pts = np.stack([X], # placeholder, check for mujoco camera coordinate conventions
                   [Y],
                   [Z])

    return pts # if smallest Z equals a certain value? position from camera frame, stop to avoid collsion

# def display_depth():
#     # print [depth of objects]? idk if this has to control anything but it needs proof of existing
#     # show depth map in another window

#     # depth_img -= depth_img.min() #copied from the mujoco tutorial code, don't know if we actually need this but it looks cool
#     #             depth_img /= 2*depth_img[depth_img <= 1].mean()
#     #             pixels = 255*np.clip(depth_img, 0, 1)
#     #             depth_view = pixels.astype(np.uint8) 
#     #             cv2.imshow("RealSense D435 - Depth", depth_view)

#     depth_map = get_depth_map()
#     depth_clipped = np.clip(depth_map, 0.1, 5.0)
#     depth_norm = 

#     cv2.imshow("MuJoCo live depth map", depth)

#     pass
