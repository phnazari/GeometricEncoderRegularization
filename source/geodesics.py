import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import inv, kron

from geometry import get_pullbacked_Riemannian_metric, jacobian_decoder_jvp_parallel


# Define the function M(gamma)
def M(gamma):
    # Placeholder: User must provide the actual implementation
    return np.eye(len(gamma))


# Define the function dM(gamma)
def dM(gamma):
    # Placeholder: User must provide the actual implementation
    # This should return a matrix composed of matrices, i.e., a 3D array
    return np.zeros((len(gamma), len(gamma), len(gamma)))


class GeodesicSolver:
    def __init__(self, model):
        self.model = model

    def solve(self, gamma_0, gamma_dot_0, t_span):
        # Initial conditions
        initial_conditions = np.concatenate((gamma_0, gamma_dot_0))

        # Solve the ODE
        solution = solve_ivp(
            self.geodesic_ode,
            t_span,
            initial_conditions,
            method="RK45",
            t_eval=np.linspace(t_span[0], t_span[1], 100),
        )

        # The solution object contains the trajectory and its derivatives
        gamma_trajectory = solution.y[:2]
        gamma_dot_trajectory = solution.y[2:]

        return gamma_trajectory, gamma_dot_trajectory

    def geodesic_ode(self, t, y):
        # Split y into gamma and gamma_dot
        dim = len(y) // 2
        gamma, gamma_dot = y[:dim], y[dim:]

        M = get_pullbacked_Riemannian_metric(self.model.decode, gamma)

        def get_metric(z):
            return get_pullbacked_Riemannian_metric(self.model.decode, z)

        def get_flattened_metric(z):
            M = get_metric(z)
            return M.T.flatten()

        M = get_metric(gamma)

        # calculate gradient of M w.r.t. gamma
        dM = jacobian_decoder_jvp_parallel(get_metric, gamma, v=None)

        dM_flattened = jacobian_decoder_jvp_parallel(
            get_flattened_metric, gamma, v=None
        )

        # Compute the right-hand side of the ODE
        gamma_ddot = (
            -0.5 * np.inv(M) @ (2 * dM - dM_flattened) @ np.kron(gamma_dot, gamma_dot)
        )
        # Return the derivatives
        return np.concatenate((gamma_dot, gamma_ddot))
