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
from tqdm import tqdm
from copy import deepcopy
import random


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

        raise NotImplementedError("Subclasses must implement the __call__ method.")


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


class NeuralNetworkModel(Model):
    """
    Nonlinear measurement model implemented as a simple feedforward neural network.
    This model captures complex relationships between latent states and designs,
    making it suitable for testing the Extended Kalman Filter's ability to
    handle nonlinear dynamics.
    Attributes:
        W1: Weight matrix for first layer
        b1: Bias vector for first layer
        W2: Weight matrix for second layer
        b2: Bias vector for second layer
    """

    def __init__(self, input_dim, hidden_dim_0, hidden_dim_1, key=None):
        """
        Initialize neural network parameters.
        Args:
            input_dim: Dimensionality of input (state + design)
            hidden_dim: Number of hidden units in the network
            key: JAX random key for reproducibility (optional)
        """
        self.input_dim = input_dim
        self.hidden_dim_0 = hidden_dim_0
        self.hidden_dim_1 = hidden_dim_1

    def __call__(self, z, x):
        """
        Compute nonlinear measurement using a feedforward neural network.
        Args:
            z: Latent state vector of shape (d, 1)
            x: Design vector/matrix of shape (d, n)
        Returns:
            Measurements of shape (1, n) after passing through the network
        """
        assert z.shape[0] == (
            self.input_dim * self.hidden_dim_0
            + self.hidden_dim_0
            + self.hidden_dim_0 * self.hidden_dim_1
            + self.hidden_dim_1
            + self.hidden_dim_1 * 1
            + 1
        ), "Parameter vector z has incorrect size."
        assert x.shape[0] == self.input_dim, "Input x has incorrect dimensionality."
        w_0_slice = jnp.arange(self.input_dim * self.hidden_dim_0)
        b_0_slice = jnp.arange(w_0_slice[-1] + 1, w_0_slice[-1] + 1 + self.hidden_dim_0)
        w_1_slice = jnp.arange(
            b_0_slice[-1] + 1, b_0_slice[-1] + 1 + self.hidden_dim_0 * self.hidden_dim_1
        )
        b_1_slice = jnp.arange(w_1_slice[-1] + 1, w_1_slice[-1] + 1 + self.hidden_dim_1)
        w_2_slice = jnp.arange(b_1_slice[-1] + 1, b_1_slice[-1] + 1 + self.hidden_dim_1)
        b_2_slice = jnp.arange(w_2_slice[-1] + 1, w_2_slice[-1] + 1 + 1)
        w_0 = z[w_0_slice].reshape(self.input_dim, self.hidden_dim_0)
        b_0 = z[b_0_slice].reshape(self.hidden_dim_0, 1)
        w_1 = z[w_1_slice].reshape(self.hidden_dim_0, self.hidden_dim_1)
        b_1 = z[b_1_slice].reshape(self.hidden_dim_1, 1)
        w_2 = z[w_2_slice].reshape(self.hidden_dim_1, 1)
        b_2 = z[b_2_slice].reshape(1, 1)

        hidden_0 = jax.nn.selu(jnp.dot(w_0.T, x) + b_0)
        hidden_1 = jax.nn.selu(jnp.dot(w_1.T, hidden_0) + b_1)
        output = jax.nn.identity(jnp.dot(w_2.T, hidden_1) + b_2)

        return output
