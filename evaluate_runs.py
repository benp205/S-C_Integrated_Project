from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import cv2
import visual_servo_pbvs as vsp
from Joint_Space import JointStateController

NUM_RUNS = 5
OUTPUT_DIR = Path(".")  # Change to e.g. Path("website_images") if you want a subfolder


def run_evaluations():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_est_errors = []
    all_feature_errors = []
    all_joint_errors = []
    last_history = None
    last_first_rgb = None

    try:
        for i in range(NUM_RUNS):
            print(f"\n==========================================")
            print(f" Starting Trial {i+1} of {NUM_RUNS}")
            print(f" (Press 'q' or 'Space' in camera window to finish trial early)")
            print(f"==========================================")

            controller = JointStateController()

            # Runs the trial; exits automatically after hold_steps=100 or if 'q'/'Space' is pressed
            history, first_rgb, cube_log = vsp.run_pbvs(
                controller=controller,
                hold_steps=100,
                reuse_calibration=True,
            )

            # Ensure OpenCV windows are closed and event queue flushed between runs
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            # 1. Accumulate EKF Cube Estimation Errors (x, y, z)
            if cube_log:
                cube_est_pos = np.array([entry[1][:3] for entry in cube_log])
                cube_true_pos = np.array([entry[2] for entry in cube_log])
                # Shift true cube Z up by half-height (0.025m) to match ArUco tag surface on top
                cube_true_pos[:, 2] += 0.025
                all_est_errors.append(cube_est_pos - cube_true_pos)

            # 2. Accumulate PBVS 3D Translational Error Norm
            if history:
                all_feature_errors.append(np.array([h[1] for h in history]))
                last_history = history
                last_first_rgb = first_rgb

            # 3. Accumulate LQR Joint Tracking Error (q_true - q_target)
            joint_log = getattr(vsp.run_pbvs, "last_joint_log", [])
            if joint_log:
                q_target_arr = np.array([entry[3] for entry in joint_log])
                q_true_arr = np.array([entry[4] for entry in joint_log])
                all_joint_errors.append(q_true_arr - q_target_arr)

    finally:
        # Close MuJoCo renderer when all trials are done
        vsp.renderer.close()
        cv2.destroyAllWindows()

    if not all_est_errors or not all_feature_errors:
        print("Error: No valid run data was collected to plot.")
        return

    # --- 1. Combined 2x2 Dashboard Image ---
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"System Performance Metrics ({len(all_est_errors)} Runs)", fontsize=16)

    # Plot 1: Cube Estimation Error (X, Y, Z) - Last Run
    last_est_err_mm = all_est_errors[-1] * 1000.0
    time_steps = np.arange(len(last_est_err_mm))
    axs[0, 0].plot(time_steps, last_est_err_mm[:, 0], label="X Error")
    axs[0, 0].plot(time_steps, last_est_err_mm[:, 1], label="Y Error")
    axs[0, 0].plot(time_steps, last_est_err_mm[:, 2], label="Z Error")
    axs[0, 0].set_title(f"EKF Cube Estimation Error (Run {len(all_est_errors)})")
    axs[0, 0].set_ylabel("Error (mm)")
    axs[0, 0].set_xlabel("Tracking Step")
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plot 2: PBVS Tracking Convergence Across All Runs
    for idx, err in enumerate(all_feature_errors):
        axs[0, 1].plot(err, alpha=0.75, label=f"Run {idx+1}")
    axs[0, 1].axhline(0.010, color="k", linestyle="--", linewidth=1, label="Tolerance (10 mm)")
    axs[0, 1].set_title("PBVS Convergence (3D Position Error)")
    axs[0, 1].set_ylabel("Error (m)")
    axs[0, 1].set_xlabel("Control Iterations")
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    # Plot 3: Boxplot of Final Cube Estimation Accuracy
    final_errors_mm = [np.linalg.norm(e[-1]) * 1000.0 for e in all_est_errors]
    axs[1, 0].boxplot(final_errors_mm, patch_artist=True, tick_labels=["Overall"])
    axs[1, 0].scatter(np.ones(len(final_errors_mm)), final_errors_mm, color="red", zorder=3, label="Trial Endpoints")
    axs[1, 0].set_title("Final Cube Estimation Accuracy")
    axs[1, 0].set_ylabel("Euclidean Error (mm)")
    axs[1, 0].legend()
    axs[1, 0].grid(True, axis="y")

    # Plot 4: LQR Joint Space Tracking Error (RMS across 6 joints for each run)
    if all_joint_errors:
        for idx, j_err in enumerate(all_joint_errors):
            joint_rms_deg = np.rad2deg(np.sqrt(np.mean(j_err**2, axis=1)))
            axs[1, 1].plot(joint_rms_deg, alpha=0.75, label=f"Run {idx+1}")
        axs[1, 1].set_title("LQR Controller Joint Tracking RMS Error")
        axs[1, 1].set_ylabel("RMS Joint Error (deg)")
        axs[1, 1].set_xlabel("Simulation Step")
        axs[1, 1].legend()
        axs[1, 1].grid(True)

    plt.tight_layout()
    dashboard_path = OUTPUT_DIR / "system_performance_metrics.png"
    fig.savefig(dashboard_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved combined dashboard to '{dashboard_path}'")

    # --- 2. Save Individual Website Figures ---
    # 2a. EKF Estimation Error
    fig_ekf, ax_ekf = plt.subplots(figsize=(8, 5))
    ax_ekf.plot(time_steps, last_est_err_mm[:, 0], label="X Error")
    ax_ekf.plot(time_steps, last_est_err_mm[:, 1], label="Y Error")
    ax_ekf.plot(time_steps, last_est_err_mm[:, 2], label="Z Error")
    ax_ekf.set_title("EKF Cube Position Estimation Error")
    ax_ekf.set_xlabel("Tracking Step")
    ax_ekf.set_ylabel("Error (mm)")
    ax_ekf.legend()
    ax_ekf.grid(True)
    fig_ekf.savefig(OUTPUT_DIR / "ekf_estimation_error.png", dpi=300, bbox_inches="tight")
    plt.close(fig_ekf)

    # 2b. PBVS Convergence
    fig_pbvs, ax_pbvs = plt.subplots(figsize=(8, 5))
    for idx, err in enumerate(all_feature_errors):
        ax_pbvs.plot(err, alpha=0.8, label=f"Run {idx+1}")
    ax_pbvs.axhline(0.010, color="k", linestyle="--", linewidth=1, label="Tolerance (10 mm)")
    ax_pbvs.set_title("PBVS 3D Translational Error Convergence")
    ax_pbvs.set_xlabel("Control Iterations")
    ax_pbvs.set_ylabel("Position Error Norm (m)")
    ax_pbvs.legend()
    ax_pbvs.grid(True)
    fig_pbvs.savefig(OUTPUT_DIR / "pbvs_convergence.png", dpi=300, bbox_inches="tight")
    plt.close(fig_pbvs)

    # 2c. Final Cube Estimation Boxplot
    fig_box, ax_box = plt.subplots(figsize=(6, 5))
    ax_box.boxplot(final_errors_mm, patch_artist=True, tick_labels=["Overall"])
    ax_box.scatter(np.ones(len(final_errors_mm)), final_errors_mm, color="red", zorder=3, label="Trial Endpoints")
    ax_box.set_title("Final Cube Estimation Accuracy")
    ax_box.set_ylabel("Euclidean Error (mm)")
    ax_box.legend()
    ax_box.grid(True, axis="y")
    fig_box.savefig(OUTPUT_DIR / "final_position_accuracy.png", dpi=300, bbox_inches="tight")
    plt.close(fig_box)

    # 2d. LQR Joint Tracking Error
    if all_joint_errors:
        fig_lqr, ax_lqr = plt.subplots(figsize=(8, 5))
        for idx, j_err in enumerate(all_joint_errors):
            joint_rms_deg = np.rad2deg(np.sqrt(np.mean(j_err**2, axis=1)))
            ax_lqr.plot(joint_rms_deg, alpha=0.8, label=f"Run {idx+1}")
        ax_lqr.set_title("LQR Controller Joint Tracking RMS Error")
        ax_lqr.set_xlabel("Simulation Step")
        ax_lqr.set_ylabel("RMS Joint Error (deg)")
        ax_lqr.legend()
        ax_lqr.grid(True)
        fig_lqr.savefig(OUTPUT_DIR / "lqr_tracking_error.png", dpi=300, bbox_inches="tight")
        plt.close(fig_lqr)

    # 2e. PBVS Image-Space Tag Corner Trajectories
    if last_history and last_first_rgb is not None:
        fig_traj, ax_traj = plt.subplots(figsize=(8, 6))
        ax_traj.imshow(last_first_rgb, alpha=0.55)
        for i in range(4):
            ax_traj.plot([h[0][0, i] for h in last_history], [h[0][1, i] for h in last_history], color=f"C{i}", lw=2)
            ax_traj.plot(last_history[0][0][0, i], last_history[0][0][1, i], "o", color=f"C{i}")
        ax_traj.set_xlim(0, 640)
        ax_traj.set_ylim(480, 0)
        ax_traj.set_title("PBVS Tag Corner Trajectories (o = start)")
        ax_traj.axis("off")
        fig_traj.savefig(OUTPUT_DIR / "pbvs_corner_trajectories.png", dpi=300, bbox_inches="tight")
        plt.close(fig_traj)

    print("Saved individual website figures:")
    print(" - ekf_estimation_error.png")
    print(" - pbvs_convergence.png")
    print(" - final_position_accuracy.png")
    print(" - lqr_tracking_error.png")
    print(" - pbvs_corner_trajectories.png")


if __name__ == "__main__":
    run_evaluations()