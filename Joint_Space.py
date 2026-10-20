
import numpy as np
from scipy.signal import cont2discrete, place_poles
from scipy.linalg import solve_discrete_are



N = 6
DT = 0.02

I = np.eye(N)
Z = np.zeros((N, N))

# Nominal position-servo dynamics
# q_ddot = wn^2 * (u - q) - 2*zeta*wn*q_dot
#
# These are initial estimates, not identified robot values.
WN = 8.0
ZETA = 1.0

Ac = np.block([
    [Z, I],
    [-WN**2 * I, -2 * ZETA * WN * I]
])

Bc = np.vstack([
    Z,
    WN**2 * I
])

C = np.hstack([I, Z])
D = np.zeros((N, N))

# Convert continuous dynamics to discrete time
A, B, _, _, _ = cont2discrete(
    (Ac, Bc, C, D),
    DT
)

# -------------------------------------------------------
# CONTROLLABILITY AND OBSERVABILITY
# -------------------------------------------------------

def controllability_matrix(A, B):
    return np.hstack([
        np.linalg.matrix_power(A, i) @ B
        for i in range(A.shape[0])
    ])


def observability_matrix(A, C):
    return np.vstack([
        C @ np.linalg.matrix_power(A, i)
        for i in range(A.shape[0])
    ])


ctrb_rank = np.linalg.matrix_rank(
    controllability_matrix(A, B)
)

obsv_rank = np.linalg.matrix_rank(
    observability_matrix(A, C)
)

# -------------------------------------------------------
# LQR CONTROLLER
# -------------------------------------------------------

Q = np.diag([100.0] * N + [5.0] * N)
R = np.eye(N) * 1.0

P = solve_discrete_are(A, B, Q, R)

K = np.linalg.solve(
    R + B.T @ P @ B,
    B.T @ P @ A
)

# -------------------------------------------------------
# LUENBERGER OBSERVER
# -------------------------------------------------------

# Observer poles inside the discrete unit circle
observer_poles = [0.55] * N + [0.65] * N

L = place_poles(
    A.T, C.T, observer_poles
).gain_matrix.T


class JointStateController:

    def __init__(self, max_joint_speed=0.4):
        self.x_hat = np.zeros(2 * N)
        self.last_u = np.zeros(N)
        self.max_joint_speed = max_joint_speed
        self.initialised = False

    def reset(self, q):
        """Initialise observer and command at measured pose."""
        q = np.asarray(q, dtype=float).reshape(N)

        self.x_hat = np.concatenate([q, np.zeros(N)])
        self.last_u = q.copy()
        self.initialised = True

    def step(self, q_measured, q_ref):
        """
        Inputs:
            q_measured: measured joint positions (6,)
            q_ref: desired joint positions (6,)

        Returns:
            u: corrected position command (6,)
        """
        q_measured = np.asarray(
            q_measured, dtype=float
        ).reshape(N)

        q_ref = np.asarray(
            q_ref, dtype=float
        ).reshape(N)

        if not self.initialised:
            self.reset(q_measured)

        # Observer prediction using previous command
        x_pred = (
            A @ self.x_hat + B @ self.last_u
        )

        # Correct estimate from joint position measurement
        innovation = q_measured - C @ x_pred

        self.x_hat = x_pred + L @ innovation

        # Desired state: target position, zero velocity
        x_ref = np.concatenate([
            q_ref, np.zeros(N)
        ])

        # LQR state feedback with reference feedforward
        u_desired = q_ref - K @ (
            self.x_hat - x_ref
        )

        # Limit commanded joint position change per step
        max_change = self.max_joint_speed * DT

        u = self.last_u + np.clip(
            u_desired - self.last_u,
            -max_change,
            max_change
        )

        self.last_u = u.copy()
        return u

    def estimated_state(self):
        return self.x_hat.copy()


if __name__ == "__main__":

    print("Joint Space LQR Controller")

    print("A shape:", A.shape)
    print("B shape:", B.shape)
    print("C shape:", C.shape)

    print("Controllability:", ctrb_rank, "/ 12")
    print("Observability:", obsv_rank, "/ 12")

    print("LQR poles:")
    print(np.linalg.eigvals(A - B @ K))

    print("Observer poles:")
    print(np.linalg.eigvals(A - L @ C))

    controller = JointStateController()

    q_initial = np.zeros(6)
    controller.reset(q_initial)

    q_target = np.full(6, 0.1)

    command = controller.step(
        q_initial, q_target
    )

    print("Example command:", command)
