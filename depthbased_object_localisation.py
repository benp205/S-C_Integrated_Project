# libraries
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from machinevisiontoolbox import Image, CentralCamera
from spatialmath import SE3
import scene_setup as sc

# SCRAPPED
# * works if we add another camera

# observer --> state space controller --> camera cal, depth object , vis servo
# supposed to control extent of drop?
# identify cube --> how far (depth-wise) the camera is from the object

# model:
# z_rgbd = [u v d]^⊤, p_C = d K^(−1) [u v 1]^⊤
# output depth map? coordinates estimated?

# pipeline
# rgb img --> depth map --> point cloud + report depths

# Camera specs
model, data, renderer, camera_id = sc.model, sc.data, sc.renderer, sc.camera_id

# get frame
def get_rgb():
    '''Renders and returns the current RGB frame from the wrist-mounted camera.'''
    renderer.update_scene(data, camera=camera_id)
    return renderer.render()

# initialisation
f_px = (480 / 2) / np.tan(np.deg2rad(model.cam_fovy[camera_id]) / 2) # camera lens focal length [px]
cam = CentralCamera(f=f_px,imagesize=[640, 480], rho=1, name="realsense_d435")
# T_RC = look_at(centre=[0.0, 0.0, 1.35], target=[1.0, 0.0, 0.76]) # camera pose in {R}, change to correspond to end-effector pos
# K = cam.K
current = get_rgb()

# program start (idk if it actually works like that)

prev = current
current = get_rgb()

# develop depth map
# get current frame and compare to prev frame
# if we have another camera i'd use images from the end-effector mounted and other camera probably


Pg = prev.mono().array.astype(float)      # greyscale, float, for our own matching
Cg = current.mono().array.astype(float)
H, W = Pg.shape

drange = [40, 90]              # search range in pixels, at this image scale *** adjust as necessary
disparity = Pg.mono().stereo_BM(Cg.mono(), hw=3, drange=drange, speckle=(200, 2)).array
valid = disparity >= drange[0]

# calculate point cloud
# f_px = 1870.0          # assumed focal length in pixels at this image scale *** adjust to matach actual camera specs, idk if we just use the same everytime
baseline = 0.055       # assumed baseline [m] *** use estimated pos change from state-space/estimation ?
u0_pp, v0_pp = W / 2, H / 2

u, v = np.meshgrid(np.arange(W), np.arange(H))

Z = np.where(valid, baseline * f_px / disparity, np.nan)

X = (u - u0_pp) * Z / f_px
Y = (v - v0_pp) * Z / f_px

keep = np.isfinite(Z)
P_stereo = np.stack([X[keep], Y[keep], Z[keep]], axis=1)
colours = Pg.array[keep] / 255.0


# if smallest Z equals a certain value? position from camera frame, stop to avoid collsion
# output [depth of objects]? idk if this has to control anything but it needs proof of existing