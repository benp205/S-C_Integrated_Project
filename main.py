from Joint_Space import JointStateController
import visual_servo_pbvs as vsp
import numpy as np

# PBVS produces the desired joint pose (q_target); the state controller (LQR + Luenberger
# observer) converts it into the position commands written to the arm actuators.
controller = JointStateController()

try:
    history, first_rgb, cube_log = vsp.run_pbvs(controller=controller)
    print(f"{len(history)} visual tracking steps executed.")

    cube_est_pos = np.array([entry[1][:3] for entry in cube_log]) #take x, y, z from estimation
    cube_true_pos = np.array([entry[2] for entry in cube_log]) #true cube pos from sim
    cube_true_pos[:, 2] += 0.025 #shift cubes true z position up by half its height since sim is centre and estimate is tag on top
    
    step_errors = np.linalg.norm(cube_est_pos - cube_true_pos, axis=1)
    mean_euclidean_error = np.mean(step_errors)
    print(f"Mean Position Error: {mean_euclidean_error * 1000:.2f} mm")
    
    mean_axis = np.mean(cube_est_pos - cube_true_pos, axis=0)
    print(f"Mean Axis (X, Y, Z): {mean_axis * 1000} mm")

    if history:
        vsp.plot_on_image(history, first_rgb)
finally:
    vsp.renderer.close()

