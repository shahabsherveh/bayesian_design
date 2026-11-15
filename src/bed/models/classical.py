"""Classical linear and analytical model implementations."""

import cvxpy as cp
import jax.numpy as jnp
import numpy as np

from ._base import Model


class LinearGaussianModel:
    """
    Linear Gaussian model for experimental design.

    This class implements optimal experimental design for linear models with
    Gaussian observation noise. The model is:
        y = X @ theta + epsilon, where epsilon ~ N(0, sigma^2)

    Supports A-optimal and D-optimal design criteria with both exact and
    approximate optimization methods.

    Attributes:
        sigma: Observation noise standard deviation
        theta: True parameter vector
        X: Full design matrix (all possible design points)
        observable_designs: Subset of X that can be observed
        info_matrices: Pre-computed information matrices for each design
        R: Prior precision matrix (inverse covariance)
    """

    def __init__(
            self,
            sigma,
            theta,
            X,
            observable_designs: list[int] | None = None,
            R=np.eye(2),
    ):
        """
        Initialize Linear Gaussian Model.

        Args:
            sigma: Observation noise standard deviation
            theta: True parameter vector (d-dimensional)
            X: Design matrix of shape (n_designs, d) containing all candidate designs
            observable_designs: Optional list of indices into X specifying which
                designs can be observed. If None, all designs are observable.
            R: Prior precision matrix of shape (d, d). Default is identity.
        """
        self.sigma = sigma
        self.theta = theta
        self.X = X
        self.observable_designs = self._subset_designs(observable_designs)
        self.info_matrices = self.calculate_single_information_matrices()
        self.R = R

    @staticmethod
    def _generate_search_space(n: int) -> np.ndarray:
        """
        Generate all valid experimental designs with n total experiments.

        Creates a search space of allocation vectors where each vector sums to n,
        representing how to distribute n experiments among available designs.

        Args:
            n: Total number of experiments to allocate

        Returns:
            Array of shape (m, k) where m is number of valid allocations and k is
            number of design points. Each row sums to n.
        """
        search_space_all = np.array(np.meshgrid(
            *[np.arange(n)] * n)).T.reshape(-1, n)
        search_space_filtered = search_space_all[search_space_all.sum(
            axis=1) == n]
        return search_space_filtered

    @staticmethod
    def _generate_full_design_matrix(design_dim: tuple[int, int]) -> np.ndarray:
        """
        Generate a grid of 2D design points.

        Creates a regular grid over [-1, 1]^2 for spatial experimental design.

        Args:
            design_dim: Tuple (nx, ny) specifying grid resolution in each dimension

        Returns:
            Array of shape (nx*ny, 2) containing all grid points
        """
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, design_dim[0]
                        ), np.linspace(-1, 1, design_dim[1])
        )
        return np.column_stack([xx.flatten(), yy.flatten()])

    def generate_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        Simulate noisy observations from the linear model.

        Args:
            x: Design matrix of shape (n, d)

        Returns:
            Noisy observations y = x @ theta + noise
        """
        mean = x @ self.theta
        return mean + np.random.normal(0, self.sigma, size=mean.shape)

    def _subset_designs(self, indices: list[int] | None) -> np.ndarray:
        """
        Extract subset of designs based on indices.

        Args:
            indices: List of integer indices, or None to use all designs

        Returns:
            Subset of self.X corresponding to indices
        """
        if indices is None:
            return self.X
        else:
            return self.X[indices]

    def _select_design(self, eta):
        """
        Expand allocation vector to full design matrix.

        Args:
            eta: Allocation vector where eta[i] = number of measurements at design i

        Returns:
            Design matrix with rows repeated according to eta
        """
        return np.repeat(self.observable_designs, repeats=eta, axis=0)

    def calculate_single_information_matrices(self) -> list[np.ndarray]:
        """
        Pre-compute Fisher information matrix for each individual design.

        For a single measurement at design x, the information matrix is:
            I(x) = x @ x.T / sigma^2

        Returns:
            List of information matrices, one for each observable design
        """
        info_list = []
        for x in self.observable_designs:
            info_matrix = np.outer(x, x)
            info_list.append(info_matrix)
        return info_list

    def calculate_information_matrix(self, eta: np.ndarray) -> np.ndarray:
        """
        Calculate total Fisher information matrix for a design allocation.

        The total information is the sum of individual information matrices
        weighted by the allocation vector eta.

        Args:
            eta: Allocation vector of length n_designs where eta[i] is the number
                 of measurements at design point i

        Returns:
            Fisher information matrix of shape (d, d)
        """
        # Initialize empty information matrix with correct dimensions
        info_matrix = np.zeros((len(self.theta), len(self.theta)))

        # Sum weighted information matrices: I_total = sum_i eta_i * I_i
        for i in range(len(eta)):
            info_matrix += eta[i] * self.info_matrices[i]
        return info_matrix

    def D_opt_DCP(self) -> cp.Problem:
        """
        Formulate D-optimal design as a convex optimization problem.

        D-optimal design minimizes the determinant of the posterior covariance,
        equivalently maximizing log-determinant of the information matrix.

        This uses disciplined convex programming (DCP) via CVXPy for efficient
        continuous relaxation of the discrete design problem.

        Returns:
            CVXPy Problem object that can be solved with problem.solve()
        """
        # Decision variable: allocation weights (must be non-negative)
        eta = cp.Variable(len(self.observable_designs),
                          nonneg=True, name="eta")

        # Construct moment matrix as weighted sum of information matrices
        moment_matrix = cp.sum(
            [eta[i] * self.info_matrices[i]
                for i in range(len(self.info_matrices))]
        )

        # D-optimal criterion: minimize -log(det(R + M))
        # Negative sign converts maximization to minimization
        criterion = -cp.log_det(self.R + moment_matrix)

        objective = cp.Minimize(criterion)
        # Constraint: allocations sum to 1 (probability simplex)
        constraints = [cp.sum(eta) == 1]
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        self.d_opt_problem = problem
        return problem

    def A_opt_DCP(self) -> cp.Problem:
        """
        Formulate A-optimal design as a convex optimization problem.

        A-optimal design minimizes the trace of the posterior covariance,
        which is equivalent to minimizing the average posterior variance
        of parameter estimates.

        Returns:
            Solved CVXPy Problem object with optimal allocation in problem.var_dict["eta"]
        """
        # Decision variable: allocation weights (must be non-negative)
        eta = cp.Variable(len(self.observable_designs),
                          nonneg=True, name="eta")

        # Construct moment matrix as weighted sum of information matrices
        moment_matrix = cp.sum(
            [eta[i] * self.info_matrices[i]
                for i in range(len(self.info_matrices))]
        )

        # A-optimal criterion: minimize trace((R + M)^{-1})
        # Uses CVXPy's built-in trace of inverse function
        criterion = cp.tr_inv(self.R + moment_matrix)

        objective = cp.Minimize(criterion)
        # Constraint: allocations sum to 1 (probability simplex)
        constraints = [cp.sum(eta) == 1]
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        self.a_opt_problem = problem
        return problem

    def plot_optimal_design(self, problem: cp.Problem):
        """
        Visualize the optimal design allocation.

        Creates a scatter plot where point size and color represent
        the allocation weight at each design point.

        Args:
            problem: Solved CVXPy problem containing optimal eta values
        """
        import matplotlib.pyplot as plt
        eta = np.asarray(problem.var_dict["eta"].value, dtype=float).round(2)
        plt.scatter(
            self.observable_designs[:, 0],
            self.observable_designs[:, 1],
            s=20 + 180 * eta / max(float(eta.max()), 1e-12),
            c=eta,
        )
        plt.colorbar(label="allocation")
        plt.show()

    def A_opt_criterion(self, eta: np.ndarray, R: np.ndarray) -> float:
        """
        Evaluate A-optimality criterion for a given design.

        Computes trace of inverse information matrix, which equals
        the sum of posterior variances.

        Args:
            eta: Design allocation vector
            R: Prior precision matrix

        Returns:
            A-optimality criterion value (lower is better)
        """
        # Calculate total information from design allocation
        info_matrix = self.calculate_information_matrix(eta)

        # A-optimal criterion: tr((R + I)^{-1})
        # Trace of inverse = sum of posterior variances
        phi_A = np.trace(jnp.linalg.inv(self.R + info_matrix))
        return phi_A

    def D_opt_criterion(self, eta: np.ndarray) -> float:
        """
        Evaluate D-optimality criterion for a given design.

        Computes negative log-determinant of information matrix.
        D-optimal designs minimize the volume of the confidence ellipsoid.

        Args:
            eta: Design allocation vector

        Returns:
            D-optimality criterion value (lower is better)
        """
        # Calculate total information from design allocation
        info_matrix = self.calculate_information_matrix(eta)

        # D-optimal criterion: -log(det(R + I))
        # slogdet returns (sign, log_det) - we use [1] to get log_det
        # Negative sign converts to minimization problem
        phi_D = -jnp.linalg.slogdet(self.R + info_matrix)[1]
        return phi_D

    def D_opt_brute_force(self, n: int) -> tuple[np.ndarray, float]:
        """
        Find D-optimal design by exhaustive search.

        Enumerates all possible allocations of n experiments among
        available designs and evaluates D-optimality criterion for each.

        Warning: Computational cost grows exponentially with n.

        Args:
            n: Total number of experiments to allocate

        Returns:
            tuple: (best_eta, best_criterion) where best_eta is the optimal
                   allocation and best_criterion is its D-optimality value
        """
        best_eta = np.array([])
        best_criterion = np.inf
        search_space = self._generate_search_space(n)
        for eta in search_space:
            criterion = self.D_opt_criterion(eta)
            if criterion < best_criterion:
                best_criterion = criterion
                best_eta = eta
        return best_eta, best_criterion


class LinearModel(Model):
    """
    Linear measurement model: y = z^T @ x.

    Simple linear relationship between latent states z and designs x.
    Commonly used for linear regression problems and as a test case.
    """

    def __call__(self, z, x, **kwargs):
        """
        Compute linear measurement: y = z^T @ x.

        Args:
            z: Latent state vector of shape (d, 1)
            x: Design vector/matrix of shape (d, n) where n is number of observations

        Returns:
            Measurements of shape (1, n)
        """
        return z.T @ jnp.matrix_transpose(x)

    def jacobian(self, z, x):
        """Jacobian of y = z^T x with respect to z: the design matrix itself."""
        return jnp.asarray(x)


def generate_full_design_matrix(design_dim: tuple[int, int]) -> np.ndarray:
    """
    Generate a 2D grid of design points over [-1, 1]^2. Creates a regular grid for spatial experimental design problems.
    Commonly used with GaussianProcessModel.

    Args:
        design_dim: Tuple (nx, ny) specifying number of points in each dimension

    Returns:
        Array of shape (nx*ny, 2) containing all grid points

    Example:
        >>> X = generate_full_design_matrix((5, 5))
        >>> X.shape
        (25, 2)
    """
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, design_dim[0]), np.linspace(-1, 1, design_dim[1])
    )
    return np.column_stack([xx.flatten(), yy.flatten()])
