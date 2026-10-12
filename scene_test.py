import mujoco
import mujoco.viewer
import time

xml_model = """
<mujoco>
    <asset>
        <texture type="2d" name="aruco_tex" file="aruco_4x4_1000_7.png"/>
        <material name="marker" texture="aruco_tex"/>
    </asset>
    <worldbody>
        <light directional="true" diffuse=".8 .8 .8" specular=".2 .2 .2" pos="0 0 5" dir="0 0 -1"/>
            <geom name="floor" type="plane" pos="0 0 0" size="3 3 0.1" rgba="1 1 1 1"/>
        <geom name="table" type="box" pos="0.6 0 0.25" size="0.5 0.6 0.25" rgba="0.8 0.6 0.4 1"/>
        <geom name="dropoffpt" type="plane" pos="0.3 0 0.501" size=".035 .035 .035" rgba="1 0 1 1" material="marker"/>
        <body name="box" pos="0.2 0.2 0.525">
            <geom name="box_geom" type="box" size=".025 .025 .025" rgba="0.25 0 0.5 1"/>
        </body>
    </worldbody>
</mujoco>
"""

# aruco marker used for texture test, idk if it's final 

model = mujoco.MjModel.from_xml_string(xml_model)
data = mujoco.MjData(model)

# Launch the interactive visualizer window
with mujoco.viewer.launch_passive(model, data) as viewer:
    start = time.time()
    while viewer.is_running():
        step_start = time.time()

        # Step the physics simulation forward
        mujoco.mj_step(model, data)

        # Sync the viewer with the updated simulation state
        viewer.sync()

        # Crude time pacing to keep simulation near real-time
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
