import mujoco #sim
import mujoco.viewer
import mujoco_menagerie as mm #for robot and gripper models
import numpy as np
import cv2

#mujoco docs say implicitfast is recommended, elliptic cones are a better model of physical reality, impratio makes friction stick better
#brown table/base
#5cm red cube, 100g, 1 sliding friction
scene_xml = """
<mujoco>
  <option timestep="0.002" impratio="10" integrator="implicitfast" cone="elliptic"/>
  <visual>
    <map znear="0.01"/>
  </visual>
  <worldbody>
    <light name="top" pos="0 0 4" dir="0 0 -1" castshadow="false"/>
        <geom name="floor" type="plane" pos="0 0.3 -0.5" size="3 3 0.1" rgba="1 1 1 1"/>
    <geom name="table" type="box" pos="0 0.4 -0.25" size="0.5 0.6 0.25" rgba="0.8 0.6 0.4 1"/>
    <body name="target_cube" pos="-0.35 0.7 0.025">
      <freejoint name="target_cube_joint"/>
      <geom name="cube_geom" type="box" size="0.025 0.025 0.025" rgba="1 0 0 1"
            mass="0.1" friction="1"/>
    </body>
  </worldbody>
</mujoco>
"""

def random_cube():
    cube_x_pos = np.random.uniform(0.10, -0.35)
    cube_y_pos = np.random.uniform(0.40, 0.70)

    yaw = np.random.uniform(0, np.pi / 2) #0 to 90 deg since its a cube
    w = np.cos(yaw / 2)
    z = np.sin(yaw / 2)
    cube_quat = [w, 0, 0, z] #changing x and y makes the cube tilt instead of turn

    cube_joint = model.joint('target_cube_joint').qposadr[0] #get starting index, first 3 are [x,y,z] pos, next 4 are [w,x,y,z] quat
    data.qpos[cube_joint : cube_joint + 3] = [cube_x_pos, cube_y_pos, 0.025] #reposition
    data.qpos[cube_joint + 3 : cube_joint + 7] = cube_quat #turn
    
    mujoco.mj_forward(model, data) #make mujoco update scene

scene_spec = mujoco.MjSpec.from_string(scene_xml)
robot_spec = mm.get("universal_robots_ur5e").spec()
gripper_spec = mm.get("robotiq_2f85").spec()

gripper_base = gripper_spec.body("base_mount")
robot_attach_site = robot_spec.site("attachment_site")
robot_attach_site.attach_body(gripper_base, prefix="gripper_") #attach gripper base to robot arm at attachment site
gripper_base.add_site(name="tcp", pos=[0, 0, 0.14], size=[0.01, 0, 0], rgba=[0, 1, 0, 1]) #tool centre point of grippers

robot_wrist = robot_spec.body("wrist_3_link")
camera = robot_wrist.add_camera(name="realsense_d435", pos=[0, 0.12, 0.06], quat=[0.7071, 0.7071, 0, 0], fovy=58, resolution=[640, 480])

robot_base = robot_spec.body("base")
robot_frame = scene_spec.worldbody.add_frame(pos=[0, 0, 0]) #add frame at [0, 0, 0] world coordinates
robot_frame.attach_body(robot_base, prefix="robot_") #mount robot base to frame

model = scene_spec.compile()
data = mujoco.MjData(model)

home_qpos = model.key('robot_home').qpos #get home position from robot model
home_ctrl = model.key('robot_home').ctrl #get home motor control from robot model 
data.qpos[:len(home_qpos)] = home_qpos
data.ctrl[:len(home_ctrl)] = home_ctrl 

random_cube()

renderer = mujoco.Renderer(model, 480, 640)

camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_realsense_d435") #find id of camera

try:
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
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

            cv2.waitKey(1)
            viewer.sync()

finally:
    cv2.destroyAllWindows()
    cv2.waitKey(1) 
    
    renderer.close() 
