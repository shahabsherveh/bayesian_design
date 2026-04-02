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
        phi_A = np.trace(jnp.linalg.inv(self.R + info_matrix))
        return phi_A

    def D_opt_criterion(self, eta: np.ndarray) -> float:
        info_matrix = self.calculate_information_matrix(eta)
        phi_D = -jnp.linalg.slogdet(self.R, info_matrix)[1]
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
        phi_A = np.trace(jnp.linalg.inv(self.R + info_matrix))
        return phi_A

    def D_opt_criterion(self, eta: np.ndarray) -> float:
        info_matrix = self.calculate_information_matrix(eta)
        phi_D = -jnp.linalg.slogdet(self.R, info_matrix)[1]
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
        K_obs_inv = jnp.linalg.inv(K_obs)
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
        K_obs_inv = jnp.linalg.inv(K_obs)
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
    def jacobian(self, z, x, argnums=0):
        jac = jax.jacobian(self.__call__, argnums=argnums)
        jac_value = jac(z, x)[0, ..., 0]
        if jac_value.ndim == 1:
            jac_value = jac_value[None, :]
        return jac_value

    def __call__(self, z, x):
        # Placeholder for the model's forward pass
        pass


class LinearModel(Model):
    # def jacobian(self, z, x):
    #     return x

    def __call__(self, z, x):
        return z.T @ x


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
        return mean, cov

    def get_state_posterior(self, measurement, x):
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
        H = self.model.jacobian(self.state_prior[0], x)
        mean_meas = self.model(self.state_prior[0], x)
        cov_meas = H @ self.state_prior[1] @ H.T + self.measurement_error
        return mean_meas, cov_meas

    def measurement_posterior(self, x_pred, x_obs, measurement):
        state_post = self.get_state_posterior(measurement, x_obs)
        H = self.model.jacobian(state_post[0], x_pred)
        mean_meas_post = self.model(state_post[0], x_pred)
        cov_meas_post = H @ state_post[1] @ H.T + self.measurement_error
        return mean_meas_post, cov_meas_post

    def measurement_posterior_cov_estimate(self, x_pred, x_obs):
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
        cov_pos_estimate = self.measurement_posterior_cov_estimate(x_pred, x_obs)
        cov_prior = self.measurement_prior(x_pred)[1]
        prior = multivariate_normal(cov=cov_prior)
        posterior = multivariate_normal(cov=cov_pos_estimate)
        return prior.entropy() - posterior.entropy()

    def measurement_preditive_prior(self, x):
        state_prior = self.state_prior
        H = self.model.jacobian(state_prior[0], x)
        mean_meas_prior = self.model(state_prior[0], x)
        cov_meas_prior = H @ state_prior[1] @ H.T + self.measurement_error
        return mean_meas_prior, cov_meas_prior


class Experiment1:
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
        mean = np.zeros((latent_dim, 1))
        cov = np.eye(latent_dim) * latent_variance
        return mean, cov

    @staticmethod
    def build_design_dist(design_cov, design_mean):
        return multivariate_normal(mean=design_mean, cov=design_cov)

    def build_design_space(self, design_num):
        return self.design_dist.rvs(size=design_num, random_state=0)

    def calculate_epig(self, x_1, x_0=None):
        if x_0 is None:
            x_0 = self.design_space.T
        ekf = self.ekf
        state_prev = ekf.state_prior[0]
        j_0 = ekf.model.jacobian(state_prev.reshape(-1, 1), x_0)
        j_1 = ekf.model.jacobian(state_prev.reshape(-1, 1), x_1)
        sigma_0 = ekf.state_prior[1]
        sigma_1 = ekf.measurement_prior(x_1)[1]
        posterior_covs_deficit = (
            j_0 @ sigma_0 @ (j_1.T @ jnp.linalg.inv(sigma_1) @ j_1) @ sigma_0 @ j_0.T
        )
        cov_0 = j_0 @ sigma_0 @ j_0.T + ekf.measurement_error

        # epig = -jnp.log(1 - (posterior_covs_deficit.diagonal() / cov_0.diagonal())) / 2
        epig = posterior_covs_deficit.diagonal() / cov_0.diagonal()
        return epig.mean()

    def calculate_epig_monte_carlo(self, x_1, x_0, num_samples=1000):
        latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=num_samples)
        y_1_samples = multivariate_normal(
            mean=self.model(latent_samples.T, x_1).flatten(),
            cov=self.measurement_error * np.eye(num_samples),
        ).rvs()
        y_0_samples = multivariate_normal(
            mean=self.model(latent_samples.T, x_0).flatten(),
            cov=self.measurement_error * np.eye(num_samples),
        ).rvs()
        breakpoint()

    def calculate_eig(self, x_1, x_0=None):
        state_cov_prev = self.ekf.state_prior[1]
        measurement_error = self.measurement_error

        eig = x_1.T @ state_cov_prev @ x_1
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
        if x_init is not None:
            x = x_init
        elif x_init_type == "random":
            x = (
                self.design_space[np.random.choice(len(self.design_space))]
                .reshape(1, -1)
                .T
            )
        elif x_init_type == "best_pool":
            pool_values = jnp.apply_along_axis(
                criterion_func, axis=1, arr=self.design_space
            )
            best_index = jnp.argmax(pool_values)
            x = self.design_space[[best_index]].T
        elif x_init_type == "normal":
            x = np.random.normal(loc=0, scale=1, size=(self.design_space.shape[1], 1))
        else:
            x = np.zeros((self.design_space.shape[1], 1))  # Default to zero vector

        crit_value = criterion_func(x, self.design_space.T)  # Initial criterion value
        grads = []
        crit_mean = None
        pbar = tqdm(
            range(max_iters), total=max_iters, desc="Optimizing Criterion", leave=False
        )
        for i in pbar:
            if stochastic:
                sample_index = np.random.choice(len(self.design_space))
                x_0 = self.design_space[[sample_index]].T
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
            pbar.set_postfix(
                {f"{criterion_label}": crit_value, f"Mean {criterion_label}": crit_mean}
            )
            if tol is not None and i > 10:
                if jnp.linalg.norm(crit_mean - crit_value) < tol:
                    break  # Convergence criterion
        return x, crit_value, grads

    def run(
        self,
        criterion_label,
        epochs,
        optimizer_params={
            "x_init": None,
            "learning_rate": 0.01,
            "max_iters": 100,
            "tol": 1e-6,
        },
    ):
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
        pbar = tqdm(
            range(epochs), total=epochs, desc=f"Running {criterion_label} Experiment"
        )
        for i in pbar:
            # epig_pool = jnp.apply_along_axis(
            #     self.calculate_epig, axis=1, arr=self.design_space
            # )
            # x_init = self.design_space[jnp.argmax(epig_pool)]
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
            measurement = self.model(self.latent_true, x_opt) + np.random.normal(
                0, np.sqrt(self.measurement_error)
            )
            ekf.state_prior = ekf.get_state_posterior(measurement, x_opt)
            # latent_estimates = multivariate_normal(
            #     mean=ekf.state_prior[0].flatten(), cov=ekf.state_prior[1]
            # ).rvs(size=1000)
            estimate_mean, estimate_cov = ekf.state_prior
            predictions = self.model(estimate_mean.reshape(-1, 1), self.design_space.T)
            rmse = self.calculate_rmse(predictions, true_measurements)
            rmse_params = self.calculate_rmse_params(estimate_mean, self.latent_true)
            grad_lists.append(grads)
            pbar.set_postfix({"Prediction RMSE": rmse, "Parameter RMSE": rmse_params})
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

    def calculate_rmse(self, predictions, true_measurements):
        return jnp.sqrt(jnp.mean((predictions - true_measurements) ** 2, axis=0)).mean()

    def calculate_rmse_params(self, estimate_mean, latent_true):
        return jnp.sqrt(
            jnp.mean((estimate_mean - latent_true.flatten()) ** 2, axis=0)
        ).mean()

    def run_experiment(self, iterations=10, optimizer_params={}):
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
    def __init__(
        self,
        rmse_params_values,
        rmse_values,
        designs,
        crit_values,
        grad_lists,
        crit_label="EPIG",
    ):
        self.rmse_params_values = rmse_params_values
        self.rmse_values = rmse_values
        self.designs = designs
        self.crit_values = crit_values
        self.grad_lists = grad_lists
        self.crit_label = crit_label

    def plot_results(self):
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
    def __init__(self, experiment_results_list):
        self.experiment_results_list = experiment_results_list

    def plot_comparison(self):
        fig, ax = plt.subplots(figsize=(12, 6))
        for result in self.experiment_results_list:
            ax.plot(result.rmse_values, marker="o", label=result.crit_label)
        ax.set_title("Comparison of RMSE over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE Value")
        ax.legend()
