import re
from .base import Experiment, BayesianExperimentalDesign
import numpy as np
from scipy.stats import (
    MonteCarloMethod,
    multivariate_normal,
)
import cvxpy as cp
import seaborn as sns
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax


class LinearGaussianModel:
    def __init__(
        self,
        sigma,
        theta,
        X,
        observable_designs: list[int] | None = None,
        R=np.eye(2),
    ):
        self.sigma = sigma
        self.theta = theta
        self.X = X
        self.observable_designs = self._subset_designs(observable_designs)
        self.info_matrices = self.calculate_single_information_matrices()
        self.R = R

    @staticmethod
    def _generate_search_space(n: int) -> np.ndarray:
        search_space_all = np.array(np.meshgrid(*[np.arange(n)] * n)).T.reshape(-1, n)
        search_space_filtered = search_space_all[search_space_all.sum(axis=1) == n]
        return search_space_filtered

    @staticmethod
    def _generate_full_design_matrix(design_dim: tuple[int, int]) -> np.ndarray:
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, design_dim[0]), np.linspace(-1, 1, design_dim[1])
        )
        return np.column_stack([xx.flatten(), yy.flatten()])

    def generate_outcomes(self, x: np.ndarray) -> np.ndarray:
        mean = x @ self.theta
        return mean + np.random.normal(0, self.sigma, size=mean.shape)

    def _subset_designs(self, indices: list[int] | None) -> np.ndarray:
        if indices is None:
            return self.X
        else:
            return self.X[indices]

    def _select_design(self, eta):
        return np.repeat(self.observable_designs, repeats=eta, axis=0)

    def calculate_single_information_matrices(self) -> list[np.ndarray]:
        info_list = []
        for x in self.observable_designs:
            info_matrix = np.outer(x, x)
            info_list.append(info_matrix)
        return info_list

    def calculate_information_matrix(self, eta: np.ndarray) -> np.ndarray:
        info_matrix = np.zeros((len(self.theta), len(self.theta)))
        for i in range(len(eta)):
            info_matrix += eta[i] * self.info_matrices[i]
        return info_matrix

    def D_opt_DCP(self) -> cp.Problem:
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
        info_matrix = self.calculate_information_matrix(eta)
        phi_A = np.trace(np.linalg.inv(self.R + info_matrix))
        return phi_A

    def D_opt_criterion(self, eta: np.ndarray) -> float:
        info_matrix = self.calculate_information_matrix(eta)
        phi_D = -np.linalg.slogdet(self.R, info_matrix)[1]
        return phi_D

    def D_opt_brute_force(self, n: int) -> tuple[np.ndarray, float]:
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
    def __init__(
        self,
        kernel,
        theta,
        X,
        observable_designs: list[int] | None = None,
        R=np.eye(2),
    ):
        self.sigma = sigma
        self.theta = theta
        self.X = X
        self.observable_designs = self._subset_designs(observable_designs)
        self.info_matrices = self.calculate_single_information_matrices()
        self.R = R

    @staticmethod
    def _generate_search_space(n: int) -> np.ndarray:
        search_space_all = np.array(np.meshgrid(*[np.arange(n)] * n)).T.reshape(-1, n)
        search_space_filtered = search_space_all[search_space_all.sum(axis=1) == n]
        return search_space_filtered

    @staticmethod
    def _generate_full_design_matrix(design_dim: tuple[int, int]) -> np.ndarray:
        xx, yy = np.meshgrid(
            np.linspace(-1, 1, design_dim[0]), np.linspace(-1, 1, design_dim[1])
        )
        return np.column_stack([xx.flatten(), yy.flatten()])

    def generate_outcomes(self, x: np.ndarray) -> np.ndarray:
        mean = x @ self.theta
        return mean + np.random.normal(0, self.sigma, size=mean.shape)

    def _subset_designs(self, indices: list[int] | None) -> np.ndarray:
        if indices is None:
            return self.X
        else:
            return self.X[indices]

    def _select_design(self, eta):
        return np.repeat(self.observable_designs, repeats=eta, axis=0)

    def calculate_single_information_matrices(self) -> list[np.ndarray]:
        info_list = []
        for x in self.observable_designs:
            info_matrix = np.outer(x, x)
            info_list.append(info_matrix)
        return info_list

    def calculate_information_matrix(self, eta: np.ndarray) -> np.ndarray:
        info_matrix = np.zeros((len(self.theta), len(self.theta)))
        for i in range(len(eta)):
            info_matrix += eta[i] * self.info_matrices[i]
        return info_matrix

    def D_opt_DCP(self) -> cp.Problem:
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
        info_matrix = self.calculate_information_matrix(eta)
        phi_A = np.trace(np.linalg.inv(self.R + info_matrix))
        return phi_A

    def D_opt_criterion(self, eta: np.ndarray) -> float:
        info_matrix = self.calculate_information_matrix(eta)
        phi_D = -np.linalg.slogdet(self.R, info_matrix)[1]
        return phi_D

    def D_opt_brute_force(self, n: int) -> tuple[np.ndarray, float]:
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
    xx, yy = np.meshgrid(
        np.linspace(-1, 1, design_dim[0]), np.linspace(-1, 1, design_dim[1])
    )
    return np.column_stack([xx.flatten(), yy.flatten()])


class GaussianProcessModel:
    def __init__(self, kernel, X):
        self.kernel = kernel
        self.X = X
        self.K = self.kernel(self.X, self.X)

    def prior(self):
        return multivariate_normal(mean=np.zeros(len(self.X)), cov=self.K)

    def predictive_distribution(self, X_obs_idx, y_obs):
        X_obs = self.X[X_obs_idx]
        K_obs = self.kernel(X_obs, X_obs)
        X_pred_idx = np.setdiff1d(np.arange(len(self.X)), X_obs_idx)
        X_pred = self.X[X_pred_idx]
        K_cross = self.kernel(X_obs, X_pred)
        k_pred = self.kernel(X_pred, X_pred)
        K_obs_inv = np.linalg.inv(K_obs)
        pred_mean = K_cross.T @ K_obs_inv @ y_obs
        pred_cov = k_pred - K_cross.T @ K_obs_inv @ K_cross
        mean = np.zeros(len(self.X))
        mean[X_obs_idx] = y_obs
        mean[X_pred_idx] = pred_mean
        cov = np.zeros((len(self.X), len(self.X)))
        cov[np.ix_(X_pred_idx, X_pred_idx)] = pred_cov
        return multivariate_normal(mean=mean, cov=cov, allow_singular=True)

    def mutual_information(self, X_obs_idx):
        X_obs = self.X[X_obs_idx]
        K_obs = self.kernel(X_obs, X_obs)
        X_pred_idx = np.setdiff1d(np.arange(len(self.X)), X_obs_idx)
        X_pred = self.X[X_pred_idx]
        K_cross = self.kernel(X_obs, X_pred)
        k_pred = self.kernel(X_pred, X_pred)
        K_obs_inv = np.linalg.inv(K_obs)
        pred_cov = k_pred - K_cross.T @ K_obs_inv @ K_cross
        prior = multivariate_normal(cov=k_pred)
        posterior = multivariate_normal(cov=pred_cov)
        return prior.entropy() - posterior.entropy()

    def plot_sample(self, sample, observed_idx=None):
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
    def jacobian(self, z, x):
        # Placeholder for the Jacobian of the model
        pass

    def __call__(self, z, x):
        # Placeholder for the model's forward pass
        pass


class LinearModel(Model):
    def jacobian(self, z, x):
        return x

    def __call__(self, z, x):
        return x @ z


class EKF:
    def __init__(
        self,
        model: Model,
        state_prev,
        state_cov_prev,
        state_innovation,
        measurement_error,
    ):
        self.model = model
        self.state_prior = self._get_state_prior(
            state_prev, state_cov_prev, state_innovation
        )
        self.measurement_error = measurement_error

    @staticmethod
    def _get_state_prior(state_prev, state_cov_prev, state_innovation):
        mean = state_prev
        cov = state_cov_prev + state_innovation
        return multivariate_normal(mean=mean.flatten(), cov=cov)

    def get_state_posterior(self, measurement, x):
        prior_mean = self.state_prior.mean.reshape(-1, 1)
        prior_cov = self.state_prior.cov
        H = self.model.jacobian(prior_mean, x)
        S = H @ prior_cov @ H.T + self.measurement_error
        K = prior_cov @ H.T @ jnp.linalg.inv(S)
        mean_post = prior_mean + K @ (measurement - self.model(prior_mean, x))
        cov_post = (
            np.eye(len(self.state_prior.mean)) - K @ H
        ) @ self.state_prior.cov @ (
            np.eye(len(self.state_prior.mean)) - K @ H
        ).T + K @ self.measurement_error @ K.T
        return multivariate_normal(mean=mean_post.flatten(), cov=cov_post)

    def measurement_prior(self, x):
        H = self.model.jacobian(self.state_prior.mean, x)
        mean_meas = self.model(self.state_prior.mean, x)
        cov_meas = H @ self.state_prior.cov @ H.T + self.measurement_error
        return multivariate_normal(mean=mean_meas.flatten(), cov=cov_meas)

    def measurement_posterior(self, x_pred, x_obs, measurement):
        state_post = self.get_state_posterior(measurement, x_obs)
        H = self.model.jacobian(state_post.mean, x_pred)
        mean_meas_post = self.model(state_post.mean, x_pred)
        cov_meas_post = H @ state_post.cov @ H.T + self.measurement_error
        return multivariate_normal(mean=mean_meas_post.flatten(), cov=cov_meas_post)

    def measurement_posterior_cov_estimate(self, x_pred, x_obs):
        meas_prior_pred = self.measurement_prior(x_pred)
        meas_prior_obs = self.measurement_prior(x_obs)
        cov_cross = (
            self.model.jacobian(self.state_prior.mean, x_pred)
            @ self.state_prior.cov
            @ self.model.jacobian(self.state_prior.mean, x_obs).T
        )
        K = cov_cross @ np.linalg.inv(meas_prior_obs.cov)
        cov_meas_pos = meas_prior_pred.cov - K @ cov_cross.T
        return cov_meas_pos

    def calculate_mutual_information(self, x_pred, x_obs):
        cov_pos_estimate = self.measurement_posterior_cov_estimate(x_pred, x_obs)
        cov_prior = self.measurement_prior(x_pred).cov
        prior = multivariate_normal(cov=cov_prior)
        posterior = multivariate_normal(cov=cov_pos_estimate)
        return prior.entropy() - posterior.entropy()

    def measurement_preditive_prior(self, x):
        state_prior = self.state_prior
        H = self.model.jacobian(state_prior.mean, x)
        mean_meas_prior = self.model(state_prior.mean, x)
        cov_meas_prior = H @ state_prior.cov @ H.T + self.measurement_error
        return multivariate_normal(mean=mean_meas_prior.flatten(), cov=cov_meas_prior)
