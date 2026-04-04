"""
Bayesian experimental design models and optimization algorithms.

This module implements various models for optimal experimental design including:
- LinearGaussianModel: Linear regression with Gaussian noise
- GP: Gaussian Process for experimental design
- GaussianProcessModel: Full GP implementation with prior/posterior
- EKF: Extended Kalman Filter for sequential experiments
- Experiment1: EKF-based experimental design framework
"""

import numpy as np
from scipy.stats import (
    multivariate_normal,
)
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import jax.numpy as jnp
import jax
import optax
from tqdm import tqdm
from copy import deepcopy


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
        search_space_all = np.array(np.meshgrid(*[np.arange(n)] * n)).T.reshape(-1, n)
        search_space_filtered = search_space_all[search_space_all.sum(axis=1) == n]
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
            np.linspace(-1, 1, design_dim[0]), np.linspace(-1, 1, design_dim[1])
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
        eta = cp.Variable(len(self.observable_designs), nonneg=True, name="eta")

        # Construct moment matrix as weighted sum of information matrices
        moment_matrix = cp.sum(
            [eta[i] * self.info_matrices[i] for i in range(len(self.info_matrices))]
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
        eta = cp.Variable(len(self.observable_designs), nonneg=True, name="eta")

        # Construct moment matrix as weighted sum of information matrices
        moment_matrix = cp.sum(
            [eta[i] * self.info_matrices[i] for i in range(len(self.info_matrices))]
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
        eta = problem.var_dict["eta"].value.round(2)
        sns.scatterplot(
            x=self.observable_designs[:, 0],
            y=self.observable_designs[:, 1],
            size=eta,
            hue=eta,
            sizes=(20, 200),
        )
        plt.show()

    def A_opt_criterion(self, eta: list[int], R: np.ndarray) -> float:
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


class GP:
    """
    Gaussian Process model for experimental design.

    Similar to LinearGaussianModel but uses a kernel function for
    non-parametric modeling. Supports the same optimality criteria.

    Attributes:
        kernel: Covariance kernel function (e.g., RBF, Matern)
        theta: Hyperparameters
        X: Design space
        observable_designs: Subset of designs that can be observed
        info_matrices: Pre-computed information matrices
        R: Prior precision matrix
    """

    def __init__(
        self,
        kernel,
        theta,
        X,
        observable_designs: list[int] | None = None,
        R=np.eye(2),
    ):
        self.kernel = kernel
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
        search_space_all = np.array(np.meshgrid(*[np.arange(n)] * n)).T.reshape(-1, n)
        search_space_filtered = search_space_all[search_space_all.sum(axis=1) == n]
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
            np.linspace(-1, 1, design_dim[0]), np.linspace(-1, 1, design_dim[1])
        )
        return np.column_stack([xx.flatten(), yy.flatten()])

    def generate_outcomes(self, x: np.ndarray) -> np.ndarray:
        """
        Generate noisy observations at given design points.

        Simulates experimental outcomes using the true model parameters
        with Gaussian observation noise.

        Args:
            x: Design matrix of shape (n_obs, d) where each row is a design point

        Returns:
            Noisy observations of shape (n_obs,)

        Note:
            Assumes self.sigma is defined for observation noise standard deviation.
        """
        mean = x @ self.theta
        return mean + np.random.normal(0, self.sigma, size=mean.shape)

    def _subset_designs(self, indices: list[int] | None) -> np.ndarray:
        """
        Extract subset of design points from full design matrix.

        Args:
            indices: List of row indices to extract, or None for all designs

        Returns:
            Subset of self.X corresponding to specified indices, or full X if None
        """
        if indices is None:
            return self.X
        else:
            return self.X[indices]

    def _select_design(self, eta):
        """
        Convert allocation vector to actual design matrix.

        Repeats each design point according to its allocation count,
        creating a design matrix for running experiments.

        Args:
            eta: Allocation vector where eta[i] is number of replicates at design i

        Returns:
            Design matrix with repeated rows according to eta
        """
        return np.repeat(self.observable_designs, repeats=eta, axis=0)

    def calculate_single_information_matrices(self) -> list[np.ndarray]:
        """
        Pre-compute information matrix for each individual design point.

        For linear models, the information matrix from observing at design x
        is (1/sigma^2) * x * x^T. Pre-computing these matrices enables efficient
        optimization over design allocations.

        Returns:
            List of information matrices, one per observable design point,
            each of shape (d, d) where d is parameter dimension.
        """
        info_list = []
        for x in self.observable_designs:
            info_matrix = np.outer(x, x)
            info_list.append(info_matrix)
        return info_list

    def calculate_information_matrix(self, eta: np.ndarray) -> np.ndarray:
        """
        Compute total Fisher information matrix for a design allocation.

        Aggregates information from multiple design points weighted by their
        allocation counts. The total information is the sum of individual
        information matrices.

        Args:
            eta: Allocation vector of shape (n_designs,) where eta[i] is the
                number of observations at design i

        Returns:
            Fisher information matrix of shape (d, d) where d is parameter dimension
        """
        info_matrix = np.zeros((len(self.theta), len(self.theta)))
        for i in range(len(eta)):
            info_matrix += eta[i] * self.info_matrices[i]
        return info_matrix

    def D_opt_DCP(self) -> cp.Problem:
        """
        Compute D-optimal design using convex optimization (DCP).

        D-optimal design minimizes the determinant of the posterior covariance matrix,
        which is equivalent to maximizing the determinant of the information matrix.
        This minimizes the volume of the confidence ellipsoid for parameter estimates.
        Uses CVXPY's disciplined convex programming framework.

        Returns:
            cp.Problem: Solved CVXPY problem with optimal design allocations in
                problem.var_dict["eta"]. The optimal eta represents the probability
                distribution over observable designs.

        Note:
            The problem is solved immediately upon calling this method. Access
            optimal allocations via self.d_opt_problem.var_dict["eta"].value.
        """
        eta = cp.Variable(len(self.observable_designs), nonneg=True, name="eta")
        moment_matrix = cp.sum(
            [eta[i] * self.info_matrices[i] for i in range(len(self.info_matrices))]
        )
        criterion = -cp.log_det(self.R + moment_matrix)

        objective = cp.Minimize(criterion)
        constraints = [cp.sum(eta) == 1]
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        self.d_opt_problem = problem
        return problem

    def A_opt_DCP(self) -> cp.Problem:
        """
        Compute A-optimal design using convex optimization (DCP).

        A-optimal design minimizes the trace of the posterior covariance matrix,
        which corresponds to minimizing the average variance of parameter estimates.
        Uses CVXPY's disciplined convex programming framework to solve the
        optimization problem over design allocations.

        Returns:
            cp.Problem: Solved CVXPY problem with optimal design allocations in
                problem.var_dict["eta"]. The optimal eta represents the probability
                distribution over observable designs.

        Note:
            The problem is solved immediately upon calling this method. Access
            optimal allocations via self.a_opt_problem.var_dict["eta"].value.
        """
        eta = cp.Variable(len(self.observable_designs), nonneg=True, name="eta")
        moment_matrix = cp.sum(
            [eta[i] * self.info_matrices[i] for i in range(len(self.info_matrices))]
        )
        criterion = cp.tr_inv(self.R + moment_matrix)

        objective = cp.Minimize(criterion)
        constraints = [cp.sum(eta) == 1]
        problem = cp.Problem(objective=objective, constraints=constraints)
        problem.solve()
        self.a_opt_problem = problem
        return problem

    def plot_optimal_design(self, problem: cp.Problem):
        """
        Visualize optimal design allocations in 2D design space.

        Creates a scatter plot where point size and color represent the optimal
        allocation (eta) to each design location. Larger/brighter points indicate
        designs that should be sampled more frequently.

        Args:
            problem: Solved CVXPY problem containing optimal design allocations
                in var_dict["eta"]. Should be the output of A_opt_DCP() or D_opt_DCP().

        Note:
            Assumes observable_designs are 2-dimensional for plotting.
            Allocations are rounded to 2 decimal places for display.
        """
        eta = problem.var_dict["eta"].value.round(2)
        sns.scatterplot(
            x=self.observable_designs[:, 0],
            y=self.observable_designs[:, 1],
            size=eta,
            hue=eta,
            sizes=(20, 200),
        )
        plt.show()

    def A_opt_criterion(self, eta: list[int], R: np.ndarray) -> float:
        """
        Evaluate A-optimality criterion for given design allocation.

        The A-optimality criterion is the trace of the inverse of the posterior
        precision matrix (R + information matrix). Minimizing this criterion
        minimizes the average variance of parameter estimates.

        Args:
            eta: Design allocation vector of length n_designs, where eta[i] is the
                number of observations at design i.
            R: Prior precision matrix of shape (d, d) where d is parameter dimension.

        Returns:
            A-optimality criterion value (scalar). Lower values indicate better designs.

        Note:
            Uses JAX for automatic differentiation compatibility.
        """
        info_matrix = self.calculate_information_matrix(eta)
        phi_A = np.trace(jnp.linalg.inv(self.R + info_matrix))
        return phi_A

    def D_opt_criterion(self, eta: np.ndarray) -> float:
        """
        Evaluate D-optimality criterion for given design allocation.

        The D-optimality criterion is the negative log-determinant of the posterior
        precision matrix. Minimizing this criterion minimizes the volume of the
        confidence ellipsoid for parameter estimates, maximizing information gain.

        Args:
            eta: Design allocation vector of shape (n_designs,), where eta[i] is the
                number of observations at design i.

        Returns:
            D-optimality criterion value (scalar). Lower values indicate better designs.

        Note:
            Uses slogdet for numerical stability with large/small determinants.
            Compatible with JAX automatic differentiation.
        """
        info_matrix = self.calculate_information_matrix(eta)
        phi_D = -jnp.linalg.slogdet(self.R + info_matrix)[1]
        return phi_D

    def D_opt_brute_force(self, n: int) -> tuple[np.ndarray, float]:
        """
        Find D-optimal design by exhaustive search over all allocations.

        Evaluates D-optimality criterion for all possible ways to allocate n
        experiments among available designs. Computationally expensive but
        guarantees finding the global optimum for discrete allocations.

        Args:
            n: Total number of experiments to allocate across designs.

        Returns:
            Tuple of (best_eta, best_criterion) where:
                - best_eta: Optimal allocation vector of shape (n_designs,)
                - best_criterion: D-optimality value at the optimal allocation

        Note:
            Computational complexity grows exponentially with n. Only practical
            for small n (typically n <= 10).
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


def generate_full_design_matrix(design_dim: tuple[int, int]) -> np.ndarray:
    """
    Generate a 2D grid of design points over [-1, 1]^2.

    Creates a regular grid for spatial experimental design problems.
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


class GaussianProcessModel:
    """
    Gaussian Process model for spatial/functional experimental design.

    Implements GP regression with methods for computing prior, posterior,
    and mutual information. Useful for designing experiments when the
    underlying function is unknown.

    Attributes:
        kernel: Covariance kernel function (sklearn.gaussian_process.kernels)
        X: Design space points of shape (n, d)
        K: Prior covariance matrix K(X, X)
    """

    def __init__(self, kernel, X):
        """
        Initialize Gaussian Process model.

        Args:
            kernel: Kernel function defining covariance structure
            X: Design space matrix of shape (n_points, n_features)
        """
        self.kernel = kernel
        self.X = X
        self.K = self.kernel(self.X, self.X)

    def prior(self):
        """
        Get prior distribution over function values.

        Returns:
            Multivariate normal distribution N(0, K) over all design points
        """
        return multivariate_normal(mean=np.zeros(len(self.X)), cov=self.K)

    def predictive_distribution(self, X_obs_idx, y_obs):
        """
        Compute posterior predictive distribution given observations.

        Uses GP posterior formulas to compute the distribution of function
        values at unobserved points conditioned on observed data.

        Args:
            X_obs_idx: Indices of observed design points
            y_obs: Observed function values at X_obs_idx

        Returns:
            Multivariate normal distribution over all points, with observed
            points fixed and unobserved points following posterior
        """
        # Extract observed design points and compute their covariance
        X_obs = self.X[X_obs_idx]
        K_obs = self.kernel(X_obs, X_obs)

        # Find indices of prediction points (all points not observed)
        X_pred_idx = np.setdiff1d(np.arange(len(self.X)), X_obs_idx)
        X_pred = self.X[X_pred_idx]

        # Compute cross-covariance between observed and prediction points
        K_cross = self.kernel(X_obs, X_pred)

        # Compute prior covariance of prediction points
        k_pred = self.kernel(X_pred, X_pred)

        # GP posterior formulas:
        # mean = K_cross.T @ K_obs^{-1} @ y_obs
        # cov = k_pred - K_cross.T @ K_obs^{-1} @ K_cross
        K_obs_inv = jnp.linalg.inv(K_obs)
        pred_mean = K_cross.T @ K_obs_inv @ y_obs
        pred_cov = k_pred - K_cross.T @ K_obs_inv @ K_cross

        # Assemble full mean vector (observed + predicted)
        mean = np.zeros(len(self.X))
        mean[X_obs_idx] = y_obs  # Observed values are fixed
        mean[X_pred_idx] = pred_mean  # Predicted values use posterior mean

        # Assemble full covariance matrix
        cov = np.zeros((len(self.X), len(self.X)))
        cov[np.ix_(X_pred_idx, X_pred_idx)] = (
            pred_cov  # Only prediction points have uncertainty
        )

        return multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    def mutual_information(self, X_obs_idx):
        """
        Compute mutual information between observations and unobserved points.

        Quantifies the reduction in uncertainty about unobserved function
        values achieved by observing at X_obs_idx.

        Formula: MI = H(f_pred) - H(f_pred | f_obs)
        where H is differential entropy.

        Args:
            X_obs_idx: Indices of points to observe

        Returns:
            Mutual information in nats (natural logarithm)
        """
        # Extract observed and prediction points
        X_obs = self.X[X_obs_idx]
        K_obs = self.kernel(X_obs, X_obs)
        X_pred_idx = np.setdiff1d(np.arange(len(self.X)), X_obs_idx)
        X_pred = self.X[X_pred_idx]

        # Compute covariances
        K_cross = self.kernel(X_obs, X_pred)
        k_pred = self.kernel(X_pred, X_pred)

        # Posterior covariance using GP formulas
        K_obs_inv = jnp.linalg.inv(K_obs)
        pred_cov = k_pred - K_cross.T @ K_obs_inv @ K_cross

        # Mutual information = Prior entropy - Posterior entropy
        # For Gaussian: MI = 0.5 * log(det(prior_cov) / det(posterior_cov))
        prior = multivariate_normal(cov=k_pred)
        posterior = multivariate_normal(cov=pred_cov)
        return prior.entropy() - posterior.entropy()

    def plot_sample(self, sample, observed_idx=None):
        """
        Visualize a sample from the GP on a 2D grid.

        Creates a heatmap showing function values, with observed and
        predicted points marked.

        Args:
            sample: Function values at all design points
            observed_idx: Optional indices of observed points to mark
        """
        fig, ax = plt.subplots()
        xx1, xx2 = np.meshgrid(np.unique(self.X[:, 0]), np.unique(self.X[:, 1]))
        pred_idx = (
            np.setdiff1d(np.arange(len(self.X)), observed_idx)
            if observed_idx is not None
            else np.arange(len(self.X))
        )
        x1_obs, x2_obs = self.X[observed_idx][:, 0], self.X[observed_idx][:, 1]
        x1_pred, x2_pred = self.X[pred_idx][:, 0], self.X[pred_idx][:, 1]
        cmap = plt.get_cmap("PiYG")
        cntr = ax.pcolormesh(
            xx1, xx2, sample.reshape(xx1.shape), cmap=cmap, shading="auto"
        )
        fig.colorbar(cntr, ax=ax)
        ax.plot(x1_obs, x2_obs, "rx", label="Observed Points", markersize=12)
        ax.plot(x1_pred, x2_pred, "bo", label="Predicted Points", markersize=12)
        plt.show()


class Model:
    """
    Base class for measurement models used with Extended Kalman Filter.

    Defines the interface for models that map latent states and designs
    to observations. Subclasses must implement __call__ for forward evaluation.

    Note:
        The jacobian method uses JAX automatic differentiation to compute
        gradients needed for EKF linearization.
    """

    def jacobian(self, z, x, argnums=0):
        """
        Compute Jacobian of the measurement model using JAX autodiff.

        Linearizes the model around the current state estimate, which is
        required for the Extended Kalman Filter update equations.

        Args:
            z: Latent state vector of shape (d, 1) where d is state dimension
            x: Design vector or matrix
            argnums: Which argument to differentiate with respect to (0 for z, 1 for x)

        Returns:
            Jacobian matrix of shape (1, d) or (n_obs, d) depending on model

        Note:
            Uses JAX's automatic differentiation for exact gradients.
        """
        jac = jax.jacobian(self.__call__, argnums=argnums)
        jac_value = jac(z, x)[0, ..., 0]
        if jac_value.ndim == 1:
            jac_value = jac_value[None, :]
        return jac_value

    def __call__(self, z, x):
        """
        Evaluate measurement model.

        Args:
            z: Latent state vector
            x: Design vector or matrix

        Returns:
            Predicted measurement(s)

        Note:
            This is a placeholder. Subclasses must override this method.
        """
        pass


class LinearModel(Model):
    """
    Linear measurement model: y = z^T @ x.

    Simple linear relationship between latent states z and designs x.
    Commonly used for linear regression problems and as a test case.
    """

    def __call__(self, z, x):
        """
        Compute linear measurement: y = z^T @ x.

        Args:
            z: Latent state vector of shape (d, 1)
            x: Design vector/matrix of shape (d, n) where n is number of observations

        Returns:
            Measurements of shape (1, n)
        """
        return z.T @ x


class EKF:
    """
    Extended Kalman Filter for state estimation and experimental design.

    Implements EKF updates for nonlinear state-space models. Used for
    sequential experimental design where the goal is to estimate hidden
    states based on noisy measurements.

    Model form:
        x_t = f(x_{t-1}) + w_t,  w_t ~ N(0, Q)  (state dynamics)
        y_t = h(x_t, d_t) + v_t, v_t ~ N(0, R)  (measurement model)

    where d_t is the experimental design at time t.

    Attributes:
        model: Model defining measurement function h(x, d)
        state_prior: Tuple (mean, cov) of prior state distribution
        measurement_error: Measurement noise covariance R
    """

    def __init__(
        self,
        model: Model,
        state_prev,
        state_cov_prev,
        state_innovation,
        measurement_error,
    ):
        """
        Initialize Extended Kalman Filter.

        Args:
            model: Model object with __call__ and jacobian methods
            state_prev: Prior state mean (from previous time step)
            state_cov_prev: Prior state covariance
            state_innovation: Process noise covariance Q
            measurement_error: Measurement noise covariance R
        """
        self.model = model
        self.state_prior = self._get_state_prior(
            state_prev, state_cov_prev, state_innovation
        )
        self.measurement_error = measurement_error

    @staticmethod
    def _get_state_prior(state_prev, state_cov_prev, state_innovation):
        """
        Propagate state distribution through dynamics (prediction step).

        For linear dynamics f(x) = x, this simply adds process noise.

        Args:
            state_prev: Previous state estimate
            state_cov_prev: Previous state covariance
            state_innovation: Process noise covariance Q

        Returns:
            tuple: (prior_mean, prior_cov)
        """
        mean = state_prev
        cov = state_cov_prev + state_innovation
        return mean, cov

    def get_state_posterior(self, measurement, x):
        """
        Update state estimate given a measurement (correction step).

        Applies EKF update equations using linearization around current
        state estimate.

        Args:
            measurement: Observed measurement value y
            x: Design/input at which measurement was taken

        Returns:
            tuple: (posterior_mean, posterior_cov) after incorporating measurement
        """
        prior_mean = self.state_prior[0]
        prior_cov = self.state_prior[1]
        H = self.model.jacobian(prior_mean, x)
        S = H @ prior_cov @ H.T + self.measurement_error
        K = prior_cov @ H.T @ jnp.linalg.inv(S)
        mean_post = prior_mean + K @ (measurement - self.model(prior_mean, x))
        cov_post = (np.eye(len(self.state_prior[0])) - K @ H) @ self.state_prior[1] @ (
            np.eye(len(self.state_prior[0])) - K @ H
        ).T + K @ self.measurement_error @ K.T
        return mean_post, cov_post

    def measurement_prior(self, x):
        """
        Get predictive distribution of measurement before observing.

        Args:
            x: Design/input for hypothetical measurement

        Returns:
            tuple: (mean, cov) of predicted measurement distribution
        """
        H = self.model.jacobian(self.state_prior[0], x)
        mean_meas = self.model(self.state_prior[0], x)
        cov_meas = H @ self.state_prior[1] @ H.T + self.measurement_error
        return mean_meas, cov_meas

    def measurement_posterior(self, x_pred, x_obs, measurement):
        """
        Get measurement distribution at x_pred after observing at x_obs.

        Args:
            x_pred: Design point for prediction
            x_obs: Design point where measurement was taken
            measurement: Observed value at x_obs

        Returns:
            tuple: (mean, cov) of measurement distribution at x_pred
        """
        state_post = self.get_state_posterior(measurement, x_obs)
        H = self.model.jacobian(state_post[0], x_pred)
        mean_meas_post = self.model(state_post[0], x_pred)
        cov_meas_post = H @ state_post[1] @ H.T + self.measurement_error
        return mean_meas_post, cov_meas_post

    def measurement_posterior_cov_estimate(self, x_pred, x_obs):
        """
        Estimate posterior measurement covariance without actual observation.

        Useful for design optimization - predicts uncertainty reduction
        for hypothetical measurement at x_obs.

        Args:
            x_pred: Design point for prediction
            x_obs: Hypothetical observation point

        Returns:
            Estimated posterior covariance of measurement at x_pred
        """
        meas_prior_pred = self.measurement_prior(x_pred)
        meas_prior_obs = self.measurement_prior(x_obs)
        cov_cross = (
            self.model.jacobian(self.state_prior[0], x_pred)
            @ self.state_prior[1]
            @ self.model.jacobian(self.state_prior[0], x_obs).T
        )
        K = cov_cross @ jnp.linalg.inv(meas_prior_obs[1])
        cov_meas_pos = meas_prior_pred[1] - K @ cov_cross.T
        return cov_meas_pos

    def calculate_mutual_information(self, x_pred, x_obs):
        """
        Calculate mutual information between measurements at x_pred and x_obs.

        Quantifies how much observing at x_obs reduces uncertainty about
        potential measurement at x_pred.

        Args:
            x_pred: Prediction design point
            x_obs: Observation design point

        Returns:
            Mutual information in nats
        """
        cov_pos_estimate = self.measurement_posterior_cov_estimate(x_pred, x_obs)
        cov_prior = self.measurement_prior(x_pred)[1]
        prior = multivariate_normal(cov=cov_prior)
        posterior = multivariate_normal(cov=cov_pos_estimate)
        return prior.entropy() - posterior.entropy()

    def measurement_preditive_prior(self, x):
        """
        Compute prior predictive distribution of measurement.

        Alias/duplicate of measurement_prior(). Returns the distribution of
        measurements before making observations.

        Args:
            x: Design vector for measurement prediction

        Returns:
            Tuple (mean, cov) of prior predictive distribution

        Note:
            This is functionally identical to measurement_prior().
            Consider using measurement_prior() instead to avoid duplication.
        """
        state_prior = self.state_prior
        H = self.model.jacobian(state_prior[0], x)
        mean_meas_prior = self.model(state_prior[0], x)
        cov_meas_prior = H @ state_prior[1] @ H.T + self.measurement_error
        return mean_meas_prior, cov_meas_prior


class Experiment1:
    """
    Sequential experimental design framework using Extended Kalman Filter.

    Implements adaptive experimental design for linear models where designs
    are optimized sequentially using either Expected Information Gain (EIG)
    or Expected Posterior Predictive Information Gain (EPIG) criteria.

    The framework:
    1. Maintains a belief about latent parameters via EKF
    2. Optimizes design selection using gradient ascent on information criteria
    3. Updates beliefs after each measurement
    4. Tracks prediction accuracy over time

    Attributes:
        model: LinearModel for measurements
        state_init_prior: Initial prior over latent parameters
        design_dist: Distribution for sampling design candidates
        latent_true: True latent parameters (for simulation/evaluation)
        design_space: Pool of candidate design points
        measurement_error: Observation noise variance
        latent_innovation: Process noise covariance
        ekf: Extended Kalman Filter for state estimation
        plot_results: Whether to visualize results during optimization
    """

    def __init__(
        self,
        latent_dim,
        latent_var,
        latent_innovation,
        latent_true,
        design_cov,
        design_mean,
        design_pool_num,
        measurement_error,
        plot_results=False,
    ):
        """
        Initialize sequential experimental design framework.

        Args:
            latent_dim: Dimension of latent parameter space
            latent_var: Initial prior variance for latent parameters
            latent_innovation: Process noise covariance matrix
            latent_true: True latent parameters for simulation (d, 1)
            design_cov: Covariance matrix for design distribution
            design_mean: Mean vector for design distribution
            design_pool_num: Number of candidate designs to sample
            measurement_error: Observation noise variance (scalar)
            plot_results: If True, plot optimization surfaces during run
        """
        self.model = LinearModel()
        self.state_init_prior = self.build_prior(
            latent_dim=latent_dim, latent_variance=latent_var
        )
        self.design_dist = self.build_design_dist(
            design_cov=design_cov, design_mean=design_mean
        )
        self.latent_true = latent_true
        self.design_space = self.build_design_space(design_num=design_pool_num)
        self.measurement_error = measurement_error
        self.latent_innovation = latent_innovation
        self.ekf = EKF(
            model=self.model,
            state_prev=self.state_init_prior[0],
            state_cov_prev=self.state_init_prior[1],
            state_innovation=self.latent_innovation,
            measurement_error=self.measurement_error,
        )
        # self.prior_covs = jnp.array(
        #     [
        #         self.ekf.measurement_prior(x_pred.reshape(1, -1))[1][0, 0]
        #         for x_pred in self.design_space
        #     ]
        # )
        self.plot_results = plot_results

    @staticmethod
    def build_prior(latent_dim, latent_variance):
        """
        Construct initial prior distribution over latent parameters.

        Creates a Gaussian prior with zero mean and diagonal covariance.

        Args:
            latent_dim: Dimension of latent parameter space
            latent_variance: Prior variance for each parameter (scalar)

        Returns:
            Tuple (mean, cov) where:
                - mean: Zero vector of shape (latent_dim, 1)
                - cov: Diagonal covariance matrix of shape (latent_dim, latent_dim)
        """
        mean = np.zeros((latent_dim, 1))
        cov = np.eye(latent_dim) * latent_variance
        return mean, cov

    @staticmethod
    def build_design_dist(design_cov, design_mean):
        """
        Build distribution for sampling candidate design points.

        Args:
            design_cov: Covariance matrix for design distribution
            design_mean: Mean vector for design distribution

        Returns:
            scipy.stats.multivariate_normal distribution object
        """
        return multivariate_normal(mean=design_mean, cov=design_cov)

    def build_design_space(self, design_num):
        """
        Sample candidate design points from design distribution.

        Args:
            design_num: Number of design candidates to sample

        Returns:
            Array of shape (design_num, design_dim) containing candidate designs

        Note:
            Uses fixed random seed (0) for reproducibility.
        """
        return self.design_dist.rvs(size=design_num, random_state=0)

    def calculate_epig(self, x_0, x_1=None):
        """
        Calculate Expected Posterior Predictive Information Gain (EPIG).

        EPIG measures how much information observing at x_1 provides about
        predictions at other locations x_0 (typically all pool designs).
        It quantifies the value of x_1 for improving predictions across
        the design space.

        Formula: EPIG averages the relative uncertainty reduction:
            (Var_prior - Var_post) / Var_prior
        across prediction locations x_0.

        Args:
            x_0: Proposed observation design of shape (d, 1) where d is design dimension
            x_1: Prediction designs of shape (d, n). If None, uses full design_space.

        Returns:
            EPIG value (scalar). Higher values indicate more informative designs.

        Note:
            Uses linearization via Jacobians for computational efficiency.
            This is an approximation to the true EPIG which would require
            Monte Carlo sampling.
        """
        if x_1 is None:
            x_1 = self.design_space.T
        ekf = self.ekf
        state_prev = ekf.state_prior[0]
        j_1 = ekf.model.jacobian(state_prev.reshape(-1, 1), x_1)
        j_0 = ekf.model.jacobian(state_prev.reshape(-1, 1), x_0)
        sigma = ekf.state_prior[1]
        s_x = ekf.measurement_prior(x_0)[1]
        posterior_covs_deficit = (
            j_1 @ sigma @ (j_0.T @ jnp.linalg.inv(s_x) @ j_0) @ sigma @ j_1.T
        )
        cov_0 = j_1 @ sigma @ j_1.T + ekf.measurement_error

        epig = -jnp.log(1 - (posterior_covs_deficit.diagonal() / cov_0.diagonal())) / 2

        # Get the diagonal to ignore the cross-covariance of the design pool_values
        # Makes sense since in classical case the trace where calculated for the information matrix
        # epig = posterior_covs_deficit.diagonal() / cov_0.diagonal()
        return epig.mean()

    def calculate_mutual_information_mc(
        self,
        y_1,
        x_1,
        y_0,
        x_0,
        latent_samples,
    ):
        """
        Calculate EPIG using Monte Carlo sampling (incomplete implementation).

        Intended to compute EPIG by sampling from the joint distribution of
        latent parameters and measurements, avoiding linearization approximations.

        Args:
            x_1: Proposed observation design
            x_0: Prediction design(s)
            num_samples: Number of Monte Carlo samples

        Note:
            This implementation is incomplete - it samples but doesn't compute
            the mutual information. Use calculate_epig() for a working implementation.
        """
        mean_1 = self.model(latent_samples.T, x_1)
        mean_0 = self.model(latent_samples.T, x_0)
        epsilon_1 = mean_1 - y_1
        epsilon_0 = mean_0 - y_0

        @np.vectorize
        def get_normal_likelihood(epsilon):
            cov = self.measurement_error
            return multivariate_normal().pdf(epsilon)

        y_0_pdf_vals = get_normal_likelihood(epsilon_0)
        y_1_pdf_vals = get_normal_likelihood(epsilon_1)
        mi = np.log(
            (y_0_pdf_vals * y_1_pdf_vals).mean(axis=0)
            / (y_0_pdf_vals.mean(axis=0) * y_1_pdf_vals.mean(axis=0))
        )
        return mi

    def calculate_epig_mc(
        self, x, x_1=None, num_latent_samples=1000, num_outcome_samples=1000
    ):
        """
        Calculate EPIG using Monte Carlo sampling (incomplete implementation).
        Intended to compute EPIG by sampling from the joint distribution of
        latent parameters and measurements, avoiding linearization approximations.
        Args:
            x: Proposed observation design
            latent_samples: Number of samples for latent parameters
            design_samples: Number of samples for prediction designs
        """
        latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=num_latent_samples)
        outcome_latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=num_outcome_samples)
        x_1 = x_1 if x_1 is not None else self.design_space.T
        noise_0 = np.random.normal(
            loc=0,
            scale=np.sqrt(self.measurement_error),
            size=(num_outcome_samples, x.shape[1]),
        )
        noise_1 = np.random.normal(
            loc=0,
            scale=np.sqrt(self.measurement_error),
            size=(num_outcome_samples, x_1.shape[1]),
        )
        y_0_samples = self.model(outcome_latent_samples.T, x) + noise_0
        y_1_samples = self.model(outcome_latent_samples.T, x_1) + noise_1

        mi = self.calculate_mutual_information_mc(
            y_1_samples,
            x_1,
            y_0_samples,
            x,
            latent_samples=latent_samples,
        )
        return mi.mean()

    def calculate_eig(self, x_0, x_1=None):
        """
        Calculate Expected Information Gain (EIG) about parameters.

        EIG measures how much information observing at x_1 provides about
        the latent parameters themselves (not predictions). For linear models,
        this equals x_1^T @ Cov_prior @ x_1, which is the prior variance
        of measurenp.hstack(ments at x_1.

        Args:
            x_0: Proposed observation design of shape (d, 1)
            x_1: Not used (kept for interface compatibility with calculate_epig)

        Returns:
            EIG value (scalar). Higher values indicate more informative designs.

        Note:
            For linear Gaussian models, EIG has a closed form and doesn't require
            Monte Carlo estimation. The optimal EIG design is proportional to
            the eigenvector with largest eigenvalue of the prior covariance.
        """
        state_prior_cov = self.ekf.state_prior[1]
        measurement_error = self.measurement_error

        eig = jnp.log(x_0.T @ state_prior_cov @ x_0 / measurement_error + 1) / 2
        return eig

    def optimize(
        self,
        criterion_func,
        criterion_label,
        max_iters,
        learning_rate=0.01,
        x_init=None,
        tol=None,
        stochastic=True,
        x_init_type: str = "random",
    ):
        """
        Optimize design using gradient ascent on information criterion.

        Uses JAX automatic differentiation to compute gradients and perform
        gradient ascent to find the design that maximizes the given criterion.
        Supports both stochastic (single random prediction point) and full
        (all prediction points) gradient estimation.

        Args:
            criterion_func: Function to maximize (e.g., calculate_epig or calculate_eig).
                Should have signature func(x_obs, x_pred) -> scalar.
            criterion_label: Name of criterion for logging (e.g., "EPIG", "EIG")
            max_iters: Maximum number of gradient ascent iterations
            learning_rate: Step size for gradient ascent (default: 0.01)
            x_init: Initial design. If None, determined by x_init_type.
            tol: Gradient norm tolerance for early stopping. If None, runs full iterations.
            stochastic: If True, use single random prediction point per iteration.
                If False, use all prediction points (slower but more accurate).
            x_init_type: How to initialize design if x_init is None:
                - "random": Random design from pool
                - "best_pool": Design from pool with highest criterion value
                - "normal": Sample from N(0, 1)
                - other: Zero vector

        Returns:
            Tuple (x_opt, crit_value, grads) where:
                - x_opt: Optimized design of shape (d, 1)
                - crit_value: Final criterion value
                - grads: List of gradients at each iteration

        Note:
            Uses gradient ASCENT (not descent) since we maximize information.
            Progress displayed via tqdm progress bar.
        """
        # Manually initialize x for Gradient Ascent
        if x_init is not None:
            x = x_init
        # Randomly initialize from design pool
        elif x_init_type == "random":
            x = (
                self.design_space[np.random.choice(len(self.design_space))]
                .reshape(1, -1)
                .T
            )
        # Initialize with design from pool that has highest criterion value
        elif x_init_type == "best_pool":
            pool_values = jnp.apply_along_axis(
                criterion_func, axis=1, arr=self.design_space
            )
            best_index = jnp.argmax(pool_values)
            x = self.design_space[[best_index]].T
        # Randomly initialize from a normal distribution
        elif x_init_type == "normal":
            x = np.random.normal(loc=0, scale=1, size=(self.design_space.shape[1], 1))

        # Otherwise initialize with zero
        else:
            x = np.zeros((self.design_space.shape[1], 1))  # Default to zero vector

        crit_value = criterion_func(x, self.design_space.T)  # Initial criterion value
        grads = []
        crit_mean = None
        progress_bar = tqdm(
            range(max_iters), total=max_iters, desc="Optimizing Criterion", leave=False
        )
        for i in progress_bar:
            # In stochastic gradient descent selects the x_0 randomly from the design pool
            if stochastic:
                sample_index = np.random.choice(len(self.design_space))
                x_0 = self.design_space[[sample_index]].T
            # If non-stochastic uses the whole data to calculate the gradient
            else:
                x_0 = self.design_space.T
            grad = jax.grad(criterion_func)(x, x_0)
            grads.append(grad)
            x += learning_rate * grad  # Gradient ascent step
            crit_value = criterion_func(x, x_0)
            crit_mean = (
                ((i - 1) * crit_mean + crit_value) / i
                if crit_mean is not None
                else crit_value
            )

            # Display Criterion Value in the progress bar
            progress_bar.set_postfix(
                {f"{criterion_label}": crit_value, f"Mean {criterion_label}": crit_mean}
            )

            if tol is not None and i > 10:
                grad_norm = jnp.linalg.norm(grad)
                if grad_norm < tol:
                    break  # Convergence criterion
        return x, crit_value, grads

    def run(
        self,
        criterion_label,
        epochs,
        optimizer_params={
            "x_init": None,
            "learning_rate": 1,
            "max_iters": 20,
            "tol": 1e-6,
        },
    ):
        """
        Run sequential experimental design for multiple epochs.

        Executes the full adaptive experimental design loop:
        1. Optimize design using specified criterion
        2. Simulate measurement at optimal design
        3. Update EKF state estimate
        4. Evaluate prediction and parameter estimation accuracy
        5. Repeat for specified number of epochs

        Args:
            criterion_label: Which criterion to use ("EPIG" or "EIG")
            epochs: Number of sequential experiments to run
            optimizer_params: Dictionary of parameters for optimize() method:
                - x_init: Initial design for optimization
                - learning_rate: Gradient ascent step size
                - max_iters: Max optimization iterations per epoch
                - tol: Gradient tolerance for early stopping

        Returns:
            ExperimentResults object containing:
                - rmse_params_values: Parameter estimation RMSE at each epoch
                - rmse_values: Prediction RMSE at each epoch
                - designs: Selected designs for each epoch
                - crit_values: Criterion values for each epoch
                - grad_lists: Gradients from each optimization

        Note:
            Generates synthetic measurements using self.latent_true with
            Gaussian noise. Progress displayed via tqdm.
        """
        ekf = self.ekf
        if criterion_label.upper() == "EPIG":
            criterion_func = self.calculate_epig
        elif criterion_label.upper() == "EIG":
            criterion_func = self.calculate_eig
        designs = []
        crit_values = []
        rmse_values = []
        rmse_params_values = []
        grad_lists = []
        true_measurements = self.model(
            self.latent_true, self.design_space.T
        ) + jax.random.normal(
            key=jax.random.key(1234), shape=(1, self.design_space.shape[0])
        ) * np.sqrt(self.measurement_error)
        progress_bar = tqdm(
            range(epochs), total=epochs, desc=f"Running {criterion_label} Experiment"
        )
        for i in progress_bar:
            x_opt, crit_value, grads = self.optimize(
                criterion_func=criterion_func,
                criterion_label=criterion_label,
                **optimizer_params,
            )
            if self.plot_results:
                self.plot_crit_surface(
                    title=f"{criterion_label} optimization",
                    new_design=x_opt,
                    previous_designs=jnp.array(designs) if designs else None,
                )
            measurement = self.model(self.latent_true, x_opt)
            # + np.random.normal(
            # 0, np.sqrt(self.measurement_error)
            # )
            ekf.state_prior = ekf.get_state_posterior(measurement, x_opt)
            # latent_estimates = multivariate_normal(
            #     mean=ekf.state_prior[0].flatten(), cov=ekf.state_prior[1]
            # ).rvs(size=1000)
            estimate_mean, estimate_cov = ekf.state_prior
            predictions = self.model(estimate_mean.reshape(-1, 1), self.design_space.T)
            rmse = self.calculate_rmse()
            rmse_params = self.calculate_rmse_params(estimate_mean, self.latent_true)
            grad_lists.append(grads)
            progress_bar.set_postfix(
                {"Prediction RMSE": rmse, "Parameter RMSE": rmse_params}
            )
            designs.append(x_opt)
            crit_values.append(crit_value)
            rmse_values.append(rmse)
            rmse_params_values.append(rmse_params)
        return ExperimentResults(
            rmse_params_values,
            rmse_values,
            jnp.array(designs),
            crit_values,
            grad_lists,
            crit_label=criterion_label,
        )

    def calculate_rmse(self):
        """
        Calculate root mean squared error for predictions.

        Args:
            predictions: Predicted measurement values
            true_measurements: True measurement values

        Returns:
            Average RMSE across all prediction locations (scalar)
        """
        sigma = self.ekf.state_prior[1]
        pred_vars = jnp.apply_along_axis(
            lambda x: x.T @ sigma @ x + self.measurement_error,
            axis=1,
            arr=self.design_space,
        )
        rmse_pred = jnp.sqrt(jnp.mean(pred_vars))
        return rmse_pred

    def calculate_rmse_params(self, estimate_mean, latent_true):
        """
        Calculate root mean squared error for parameter estimates.

        Args:
            estimate_mean: Estimated parameter values
            latent_true: True parameter values

        Returns:
            Average RMSE across all parameters (scalar)
        """
        return jnp.sqrt(
            jnp.mean((estimate_mean - latent_true.flatten()) ** 2, axis=0)
        ).mean()

    def run_experiment(self, iterations=10, optimizer_params={}):
        """
        Compare EPIG and EIG design strategies side-by-side.

        Runs both EPIG and EIG sequential experiments with identical settings
        to enable direct comparison of the two design criteria.

        Args:
            iterations: Number of sequential experiments for each method
            optimizer_params: Optimization settings (passed to run())

        Returns:
            MultiExperimentResults containing results from both methods

        Note:
            Creates a deepcopy of self for the second experiment to ensure
            independent EKF state evolution.
        """
        new_self = deepcopy(self)
        epig_results = self.run(
            criterion_label="EPIG", epochs=iterations, optimizer_params=optimizer_params
        )
        eig_results = new_self.run(
            criterion_label="EIG", epochs=iterations, optimizer_params=optimizer_params
        )
        return MultiExperimentResults([epig_results, eig_results])

    def plot_crit_surface(
        self,
        title="EPIG and EIG Surfaces",
        x_range=None,
        y_range=None,
        grid_size=20,
        new_design=None,
        previous_designs=None,
    ):
        """
        Visualize EPIG and EIG criterion surfaces over 2D design space.

        Creates side-by-side contour plots showing how EPIG and EIG values
        vary across the design space. Marks the design pool, previously
        selected designs, and newly optimized design.

        Args:
            title: Overall figure title
            x_range: Tuple (min, max) for first design dimension.
                If None, auto-computed from design_space.
            y_range: Tuple (min, max) for second design dimension.
                If None, auto-computed from design_space.
            grid_size: Number of grid points per dimension for contour plot
            new_design: Newly optimized design to mark with red 'X'
            previous_designs: Array of previously selected designs to mark with blue 'o'

        Note:
            Assumes 2D design space for visualization. Computes criterion values
            at all grid points, which can be slow for fine grids.
        """
        x_range = (
            (self.design_space[:, 0].min() - 1, self.design_space[:, 0].max() + 1)
            if x_range is None
            else x_range
        )
        y_range = (
            (self.design_space[:, 1].min() - 1, self.design_space[:, 1].max() + 1)
            if y_range is None
            else y_range
        )

        x1 = jnp.linspace(x_range[0], x_range[1], grid_size)
        x2 = jnp.linspace(y_range[0], y_range[1], grid_size)
        xx1, xx2 = jnp.meshgrid(x1, x2)
        grid_points = jnp.column_stack([xx1.flatten(), xx2.flatten()])
        fig, ax = plt.subplots(1, 2, sharey=True, figsize=(16, 6))
        crit_values_epig = jnp.array(
            [self.calculate_epig(x) for x in grid_points]
        ).reshape(xx1.shape)
        crit_values_eig = jnp.array(
            [self.calculate_eig(x) for x in grid_points]
        ).reshape(xx1.shape)

        ax[0].set_title("EPIG Surface")
        c = ax[0].contourf(xx1, xx2, crit_values_epig, levels=50, cmap="viridis")
        ax[0].scatter(
            self.design_space[:, 0],
            self.design_space[:, 1],
            c="black",
            label="Design Pool",
        )
        plt.colorbar(c, label="EPIG", ax=ax[0])
        ax[0].set_xlabel("Design Dimension 1")
        ax[0].set_ylabel("Design Dimension 2")
        ax[1].set_title("EIG Surface")
        c = ax[1].contourf(xx1, xx2, crit_values_eig, levels=50, cmap="viridis")
        ax[1].scatter(
            self.design_space[:, 0],
            self.design_space[:, 1],
            c="black",
        )
        plt.colorbar(c, label="EIG", ax=ax[1])
        ax[1].set_xlabel("Design Dimension 1")
        ax[1].set_ylabel("Design Dimension 2")
        if previous_designs is not None:
            ax[0].scatter(
                previous_designs[:, 0],
                previous_designs[:, 1],
                c="blue",
                label="Added Designs",
                marker="o",
                s=100,
            )
            ax[1].scatter(
                previous_designs[:, 0],
                previous_designs[:, 1],
                c="blue",
                marker="o",
                s=100,
            )
        if new_design is not None:
            ax[0].scatter(
                new_design[0],
                new_design[1],
                c="red",
                label="New Design",
                marker="X",
                s=100,
            )
            ax[1].scatter(
                new_design[0],
                new_design[1],
                c="red",
                marker="X",
                s=100,
            )
        fig.suptitle(title)
        fig.legend(loc="upper right")


class ExperimentResults:
    """
    Container for results from a sequential experimental design run.

    Stores all metrics tracked during an experiment including RMSE values,
    selected designs, criterion values, and optimization gradients.

    Attributes:
        rmse_params_values: List of parameter estimation RMSEs per epoch
        rmse_values: List of prediction RMSEs per epoch
        designs: Array of selected designs, shape (n_epochs, d, 1)
        crit_values: List of criterion values for selected designs
        grad_lists: List of gradient histories from each optimization
        crit_label: Name of criterion used ("EPIG" or "EIG")
    """

    def __init__(
        self,
        rmse_params_values,
        rmse_values,
        designs,
        crit_values,
        grad_lists,
        crit_label="EPIG",
    ):
        """
        Initialize experiment results.

        Args:
            rmse_params_values: Parameter RMSE values over epochs
            rmse_values: Prediction RMSE values over epochs
            designs: Selected designs for each epoch
            crit_values: Criterion values for each selected design
            grad_lists: Gradients from each optimization run
            crit_label: Label for the criterion used
        """
        self.rmse_params_values = rmse_params_values
        self.rmse_values = rmse_values
        self.designs = designs
        self.crit_values = crit_values
        self.grad_lists = grad_lists
        self.crit_label = crit_label

    def plot_results(self):
        """
        Plot experiment results showing criterion and RMSE evolution.

        Creates a 3-panel figure showing:
        - Top: Criterion values over iterations
        - Bottom left: Parameter estimation RMSE over iterations
        - Bottom right: Prediction RMSE over iterations

        Useful for assessing convergence and comparing design strategies.
        """
        fig = plt.figure(figsize=(16, 9))
        gs = GridSpec(2, 2, figure=fig)
        ax_crit = fig.add_subplot(gs[0, :])
        ax_crit.plot(self.crit_values, marker="o")
        ax_crit.set_title(f"{self.crit_label} Values over Iterations")
        ax_crit.set_xlabel("Iteration")
        ax_crit.set_ylabel(self.crit_label)

        ax_rmse_params = fig.add_subplot(gs[1, 0])
        ax_rmse_params.plot(self.rmse_params_values, marker="o")
        ax_rmse_params.set_title("Parameter RMSE over Iterations")
        ax_rmse_params.set_xlabel("Iteration")
        ax_rmse_params.set_ylabel("RMSE")

        ax_rmse = fig.add_subplot(gs[1, 1])
        ax_rmse.plot(self.rmse_values, marker="o")
        ax_rmse.set_title("Prediction RMSE over Iterations")
        ax_rmse.set_xlabel("Iteration")
        ax_rmse.set_ylabel("RMSE")

        fig.suptitle(f"{self.crit_label} optimization")


class MultiExperimentResults:
    """
    Container for comparing results from multiple experimental design strategies.

    Aggregates ExperimentResults from different design criteria (e.g., EPIG vs EIG)
    to enable side-by-side comparison of their performance.

    Attributes:
        experiment_results_list: List of ExperimentResults objects
    """

    def __init__(self, experiment_results_list):
        """
        Initialize multi-experiment results container.

        Args:
            experiment_results_list: List of ExperimentResults objects to compare
        """
        self.experiment_results_list = experiment_results_list

    def plot_comparison(self):
        """
        Plot prediction RMSE comparison across all experiments.

        Creates a single plot with overlaid curves showing how prediction
        RMSE evolves over iterations for each design strategy. Useful for
        visually comparing which criterion leads to faster learning.

        visually comparing which criterion leads to faster learning.

        Note:
            Each curve is labeled with its criterion name (from crit_label).
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        for result in self.experiment_results_list:
            ax.plot(result.rmse_values, marker="o", label=result.crit_label)
        ax.set_title("Comparison of RMSE over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE Value")
        ax.legend()
