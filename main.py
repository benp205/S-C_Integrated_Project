from Joint_Space import JointStateController
import visual_servo_pbvs as vsp

# PBVS produces the desired joint pose (q_target); the state controller (LQR + Luenberger
# observer) converts it into the position commands written to the arm actuators.
controller = JointStateController()

try:
    history, first_rgb = vsp.run_pbvs(controller=controller)
    print(f"{len(history)} visual tracking steps executed.")
    if history:
        vsp.plot_on_image(history, first_rgb)
finally:
    vsp.renderer.close()

