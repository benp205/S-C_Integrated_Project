
import numpy as np
import matplotlib.pyplot as plt

from Joint_Space import (
    A, B, C, K, L, DT,
    JointStateController,
    ctrb_rank, obsv_rank
)

# -------------------------------------------------------
# 1. SIMULATION SETUP
# -------------------------------------------------------

N = 6
duration = 8.0
steps = int(duration / DT)
time = np.arange(steps) * DT

# Starting joint positions (radians)
q_initial = np.zeros(N)

# Target joint positions (radians)
q_target = np.array([
    0.3,
    -0.4,
    0.5,
    -0.3,
    0.2,
    0.1
])

# Initial state: 6 positions + 6 velocities
x = np.concatenate([
    q_initial,
    np.zeros(N)
])

controller = JointStateController()
controller.reset(q_initial)

# Store simulation results
positions = []
velocities = []
estimated_velocities = []
commands = []

# -------------------------------------------------------
# 2. MATHEMATICAL VALIDATION
# -------------------------------------------------------

print("\n--- STATE SPACE VALIDATION ---")

print("Controllability rank:", ctrb_rank, "/ 12")
print("Observability rank:", obsv_rank, "/ 12")

lqr_poles = np.linalg.eigvals(A - B @ K)
observer_poles = np.linalg.eigvals(A - L @ C)

print("LQR stable:", np.all(np.abs(lqr_poles) < 1))
print("Observer stable:", np.all(np.abs(observer_poles) < 1))

# -------------------------------------------------------
# 3. RUN CLOSED-LOOP SIMULATION
# -------------------------------------------------------

for k in range(steps):

    # Measure joint positions only
    q_measured = C @ x

    # Controller estimates velocities and calculates
    # corrected joint position commands
    u = controller.step(q_measured, q_target)

    # Simulate the robot's nominal joint dynamics
    x = A @ x + B @ u

    # Save actual and estimated states
    positions.append(x[:N].copy())
    velocities.append(x[N:].copy())

    estimated_velocities.append(
        controller.estimated_state()[N:].copy()
    )

    commands.append(u.copy())

positions = np.array(positions)
velocities = np.array(velocities)
estimated_velocities = np.array(estimated_velocities)
commands = np.array(commands)

# -------------------------------------------------------
# 4. POSITION TRACKING RESULTS
# -------------------------------------------------------

final_error = q_target - positions[-1]

print("\n--- POSITION TRACKING RESULTS ---")

for i in range(N):
    print(
        f"Joint {i+1}: "
        f"Target = {q_target[i]:.3f}, "
        f"Actual = {positions[-1, i]:.3f}, "
        f"Error = {final_error[i]:.5f} rad"
    )

max_error = np.max(np.abs(final_error))

print("\nMaximum final position error:", max_error)

# -------------------------------------------------------
# 5. OBSERVER ACCURACY
# -------------------------------------------------------

velocity_error = velocities - estimated_velocities

observer_rmse = np.sqrt(
    np.mean(velocity_error**2, axis=0)
)

print("\n--- OBSERVER RESULTS ---")

for i in range(N):
    print(
        f"Joint {i+1} velocity RMSE: "
        f"{observer_rmse[i]:.5f} rad/s"
    )

# -------------------------------------------------------
# 6. PLOT JOINT POSITION RESPONSES
# -------------------------------------------------------

fig, axes = plt.subplots(
    3, 2, figsize=(12, 9), sharex=True
)

for i, ax in enumerate(axes.flat):

    ax.plot(
        time,
        positions[:, i],
        label="Actual position"
    )

    ax.axhline(
        q_target[i],
        color="red",
        linestyle="--",
        label="Target position"
    )

    ax.set_title(f"Joint {i+1}")
    ax.set_ylabel("Position (rad)")
    ax.grid(True)
    ax.legend()

for ax in axes[-1]:
    ax.set_xlabel("Time (s)")

fig.suptitle("LQR Joint Position Tracking")
plt.tight_layout()

# -------------------------------------------------------
# 7. PLOT OBSERVER PERFORMANCE
# -------------------------------------------------------

fig2, axes2 = plt.subplots(
    3, 2, figsize=(12, 9), sharex=True
)

for i, ax in enumerate(axes2.flat):

    ax.plot(
        time,
        velocities[:, i],
        label="Actual velocity"
    )

    ax.plot(
        time,
        estimated_velocities[:, i],
        "--",
        label="Estimated velocity"
    )

    ax.set_title(f"Joint {i+1}")
    ax.set_ylabel("Velocity (rad/s)")
    ax.grid(True)
    ax.legend()

for ax in axes2[-1]:
    ax.set_xlabel("Time (s)")

fig2.suptitle("Luenberger Observer Performance")
plt.tight_layout()

plt.show()
