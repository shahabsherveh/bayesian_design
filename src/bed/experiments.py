from copy import deepcopy
from flax import nnx
import jax
import jax.numpy as jnp
from matplotlib.lines import Line2D
from matplotlib.patches import BoxStyle, Patch
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from mpl_toolkits.mplot3d.axes3d import mpatches
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .ekf import EKF
from .models import LinearModel, Model, NeuralNetworkRegressor
from .data import Data

plt.rcParams.update({"font.size": 11})


class Experiment:
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
        latent_cov,
        latent_innovation,
        measurement_error,
        data: Data,
        model: NeuralNetworkRegressor,
        plot_inter_results=False,
        pre_train_model=False,
        training_kwargs={
            "epochs": 200,
            "learning_rate": 0.01,
            "rngs": nnx.Rngs(0),
        },
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
        self.model = (
            model
            if not pre_train_model
            else model.train(
                data.x_train,
                data.y_train,
                **training_kwargs,
            )
        )
        self.state_init_prior = self.build_prior(
            latent_cov=latent_cov,
            model=model,
        )
        self.data = data
        self.design_space, self.true_measurements = self.build_design_space(data)
        self.measurement_error = (
            measurement_error  # if not pre_train_model else model.mse * jnp.eye(1)
        )
        self.latent_innovation = latent_innovation
        self.ekf = EKF(
            model=model,
            state_prev=self.state_init_prior[0],
            state_cov_prev=self.state_init_prior[1],
            state_innovation=self.latent_innovation,
            measurement_error=self.measurement_error,
        )

        self.plot_3D_inter_results = (
            plot_inter_results if self.design_space.shape[-1] == 2 else False
        )
        self.plot_2D_inter_results = (
            plot_inter_results if self.design_space.shape[-1] == 1 else False
        )

    @staticmethod
    def build_prior(
        latent_cov,
        model: NeuralNetworkRegressor,
    ):
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
        model_state = nnx.state(model.flax_model)
        mean = model.flax_model.state_to_weights(model_state)
        # mean = np.zeros((latent_dim, 1))
        # mean = np.random.normal(loc=0, scale=0.0000001, size=(latent_dim, 1))
        return mean, latent_cov

    @staticmethod
    def build_design_space(data):
        """
        Sample candidate design points from design distribution.

        Args:
            design_num: Number of design candidates to sample

        Returns:
            Array of shape (design_num, design_dim) containing candidate designs

        Note:
            Uses fixed random seed (0) for reproducibility.
        """
        design_space = jnp.concatenate([data.x_train, data.x_test], axis=0)
        true_measurements = jnp.concatenate([data.y_train, data.y_test], axis=0)
        return design_space, true_measurements

    def calculate_epig(self, x, x_1=None, **kwargs):
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
            x_1 = self.design_space
        ekf = self.ekf
        state_prev = ekf.state_prior[0]
        j_1 = ekf.model.jacobian(state_prev.reshape(-1, 1), x_1)[None, ...]
        j_1_T = jnp.matrix_transpose(j_1)
        j_0 = ekf.model.jacobian(state_prev.reshape(-1, 1), x)[:, None, ...]
        j_0_T = jnp.matrix_transpose(j_0)
        sigma = ekf.state_prior[1]
        _, s_x = ekf.measurement_prior(x)
        s_x_inv = jnp.linalg.inv(s_x[:, None, ...])
        posterior_covs_deficit = j_1 @ sigma @ (j_0_T @ s_x_inv @ j_0) @ sigma @ j_1_T
        cov_0 = j_1 @ sigma @ j_1_T + ekf.measurement_error

        # epig = -jnp.log(1 - (posterior_covs_deficit / cov_0)) / 2
        epig = (
            -jnp.log(
                jnp.linalg.det(
                    jnp.eye(self.measurement_error.shape[0])
                    - posterior_covs_deficit @ jnp.linalg.inv(cov_0)
                )
            )
            / 2
        )
        # Get the diagonal to ignore the cross-covariance of the design pool_values
        # Makes sense since in classical case the trace where calculated for the information matrix
        # epig = posterior_covs_deficit.diagonal() / cov_0.diagonal()
        return epig.mean(axis=1)

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
        mean_1 = jax.vmap(lambda theta: self.model(theta.T, x_1))(latent_samples)
        mean_0 = jax.vmap(lambda theta: self.model(theta.T, x_0))(
            latent_samples
        ).swapaxes(1, 2)
        epsilon_1 = mean_1 - y_1
        epsilon_0 = mean_0 - y_0

        def get_normal_likelihood(epsilon):
            cov = self.measurement_error
            return (1 / jnp.sqrt(2 * jnp.pi * cov)) * jnp.exp(-0.5 * (epsilon**2) / cov)

        y_0_pdf_vals = get_normal_likelihood(epsilon_0)
        y_1_pdf_vals = get_normal_likelihood(epsilon_1)
        # __import__("ipdb").set_trace()
        mi = (
            jnp.log((y_0_pdf_vals * y_1_pdf_vals).mean(axis=0))
            - jnp.log(y_0_pdf_vals.mean(axis=0))
            - jnp.log(y_1_pdf_vals.mean(axis=0))
        )
        return jnp.where(~jnp.isnan(mi), mi, -jnp.inf)

    def calculate_epig_mc(self, x, x_1=None, num_latent_samples=1000, **kwargs):
        """
        Calculate EPIG using Monte Carlo sampling (incomplete implementation).
        Intended to compute EPIG by sampling from the joint distribution of
        latent parameters and measurements, avoiding linearization approximations.
        Args:
            x: Proposed observation design
            latent_samples: Number of samples for latent parameters
            design_samples: Number of samples for prediction designs
        """
        if x.ndim == 1:
            x = x[:, None]
        if x_1 is None:
            # x_1 = jax.random.choice(
            #     jax.random.key(102), a=self.design_space, shape=(num_design_samples,)
            # ).T
            # x_1 = self.design_dist.rvs(size=num_design_samples).T
            x_1 = self.design_space
        M = x_1.shape[0]
        N = x.shape[0]
        K = num_latent_samples
        outcome_latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=M)
        y_0_samples = jax.vmap(lambda theta: self.model(theta.T, x))(
            outcome_latent_samples
        )[None, ...].squeeze(-1)
        noise_0 = np.random.normal(
            loc=0,
            scale=np.sqrt(self.measurement_error),
            size=y_0_samples.shape,
        )
        y_0_samples += noise_0
        y_1_samples = jax.vmap(lambda theta: self.model(theta.T, x_1))(
            outcome_latent_samples
        )
        noise_1 = np.random.normal(
            loc=0,
            scale=np.sqrt(self.measurement_error),
            size=y_1_samples.shape,
        )
        y_1_samples += noise_1
        y_1_samples = y_1_samples.diagonal().T.swapaxes(0, 1)[..., None]

        latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=K)
        mi = self.calculate_mutual_information_mc(
            y_1_samples,
            x_1,
            y_0_samples,
            x,
            latent_samples=latent_samples,
        )
        # return mi.mean(where=~jnp.isnan(mi) & ~jnp.isinf(mi), axis=0).squeeze()
        return jnp.atleast_1d(mi.mean(axis=0).squeeze())

    def calculate_eig(self, x, *arg, **kwargs):
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
        state_prior_mean = self.ekf.state_prior[0]
        state_prior_cov = self.ekf.state_prior[1]
        measurement_error = self.measurement_error

        H = self.model.jacobian(state_prior_mean.reshape(-1, 1), x)
        H_T = H.T if H.ndim == 2 else H.swapaxes(1, 2)

        eig = jnp.log((H @ state_prior_cov @ H_T / measurement_error) + 1) / 2
        return jnp.atleast_1d(eig.squeeze())

    def calculate_random(self, x, key, **kwargs):
        val = nnx.vmap(
            lambda k: jax.random.uniform(
                key=k,
            )
        )(key)
        return val

    def optimize(self, criterion_func, method, params={"lr": 1, "max_iters": 50}):
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
        seeds = np.random.randint(
            0, 10000, size=(self.design_space.shape[0],)
        )  # Random keys for randomness in criterion
        keys = jax.vmap(jax.random.key)(seeds)
        if method == "brute_force":
            pool_values = criterion_func(x=self.design_space, key=keys)
            shuffled_indices = jax.random.permutation(
                jax.random.key(0), self.design_space.shape[0]
            )
            pool_values_shuffled = pool_values[shuffled_indices]
            best_index = shuffled_indices[jnp.argmax(pool_values_shuffled)]
            x = self.design_space[best_index]
            crit_value = pool_values_shuffled[best_index]
        elif method == "gradient_ascent":
            x = np.random.normal(
                size=self.design_space[:1].shape,
                scale=0.01,
            )
            grad_func = jax.value_and_grad(lambda x: criterion_func(x, key=keys)[0])
            max_iters = params.get("max_iters")
            lr = params.get("lr")
            pbar = tqdm(range(max_iters), desc="Optimizing design", leave=True)
            for i in pbar:
                if (x > self.design_space.max(axis=0)).any() or (
                    x < self.design_space.min(axis=0)
                ).any():
                    print("Design out of bounds, stopping optimization.")
                    break
                crit_value, grads = grad_func(x)
                x += lr * grads
                pbar.set_postfix({f"{criterion_func.__name__}": crit_value})
            x = x.squeeze(0)
            best_index = x
        elif method == "grid_search":
            num_samples = params.get("num_samples", 200)
            samples_shape = list(self.design_space.shape)
            samples_shape[0] = num_samples
            grid = jax.random.uniform(
                key=jax.random.key(0),
                shape=samples_shape,
                minval=self.design_space.min(axis=0),
                maxval=self.design_space.max(axis=0),
            )
            grid_values = criterion_func(x=grid, key=keys)
            x = grid[jnp.argmax(grid_values)]
            best_index = x
            crit_value = grid_values.max()

        return best_index, x, crit_value

    def run(
        self,
        criterion_label,
        epochs,
        optimizer="grid_search",
        optimizer_params={"lr": 1, "max_iters": 50},
        axes_2D=None,
        figure_parameters={
            "figsize": (190, 190),
            "figsize_comparison": (13, 6),
            "fontsize": 11,
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
        elif criterion_label.upper() == "EPIG-MC":
            criterion_func = self.calculate_epig_mc
        else:
            criterion_func = self.calculate_random
        designs = []
        crit_values = []
        rmse_values = []
        rmse_values_predictions = []
        figsize_mm = figure_parameters.get(
            "figsize",
        )
        figsize_mm = (figsize_mm[0], figsize_mm[1])
        figsize_inches = (figsize_mm[0] / 25.4, figsize_mm[1] / 25.4)
        figsize_comparison_mm = figure_parameters.get(
            "figsize_comparison",
        )
        figsize_comparison_inches = (
            figsize_comparison_mm[0] / 25.4,
            figsize_comparison_mm[1] / 25.4,
        )

        progress_bar = tqdm(
            range(epochs), total=epochs, desc=f"Running {criterion_label} Experiment"
        )
        if self.plot_3D_inter_results:
            fig_3D, axes_3D = plt.subplots(
                figsize=figsize_inches,
                nrows=epochs,
                ncols=3,
                sharex=True,
                sharey=True,
            )
            fig_3D.subplots_adjust(hspace=0.3, wspace=0.3)
            axes_3D[0, 0].set_title("EPIG Surface")
            axes_3D[0, 1].set_title("EPIG-MC Surface")
            axes_3D[0, 2].set_title("EIG Surface")
            for i in range(3):
                axes_3D[-1, i].set_xlabel("x1")
            for i in range(epochs):
                axes_3D[i, 0].set_ylabel("x2")

        for i in progress_bar:
            rmse = self.calculate_rmse()
            estimate_mean, estimate_cov = ekf.state_prior
            predictions = self.model(
                estimate_mean.reshape(-1, 1), self.data.x_val
            ).squeeze()
            rmse_predictions = self.calculate_rmse_predictions(
                predictions, self.data.y_val.squeeze()
            )
            best_index, x_opt, crit_value = self.optimize(
                criterion_func=criterion_func,
                method=optimizer,
                params=optimizer_params,
            )
            measurement = self.data.observe(best_index)
            # + np.random.normal(
            # 0, np.sqrt(self.measurement_error)
            # )
            if self.plot_3D_inter_results:
                self.plot_crit_surface(
                    title=f"{criterion_label} optimization",
                    new_design=x_opt,
                    previous_designs=jnp.array(designs) if designs else None,
                    axes=axes_3D[i],
                )
            if self.plot_2D_inter_results:
                if i % (epochs // 5) == 0:
                    ax = axes_2D[i // (epochs // 5)]
                    crit_ax = self.plot_crit(
                        ax=ax,
                        crit_fn=criterion_func,
                        new_design=x_opt,
                        previous_designs=jnp.array(designs) if designs else None,
                    )
                    crit_ax.set_ylabel(f"{criterion_label}")
                    props = dict(boxstyle="round", facecolor="wheat", alpha=0.2)
                    # ax.set_title(f"Iteration {i + 1}")
                    # iteration_patch = Patch(
                    #     facecolor="none", label=f"Iter {i + 1}", alpha=0.5
                    # )

                    # handles, labels = axes_2D[i].get_legend_handles_labels()
                    # handles.append(observed_patch)
                    ax.text(
                        0.1,
                        0.9,
                        f"Iter {i}",
                        transform=ax.transAxes,
                        color="black",
                        verticalalignment="top",
                        bbox=props,
                    )
                    # ax.legend(
                    #     handles=[iteration_patch],
                    #     loc="upper left",
                    #     draggable=True,
                    # )

            ekf.state_prior = ekf.get_state_posterior(measurement, x_opt[None, ...])
            # latent_estimates = multivariate_normal(
            #     mean=ekf.state_prior[0].flatten(), cov=ekf.state_prior[1]
            # ).rvs(size=1000)
            progress_bar.set_postfix(
                {
                    "Prediction RMSE": rmse,
                    "Frequentist RMSE": rmse_predictions,
                    f"{criterion_label}": crit_value,
                }
            )
            designs.append(x_opt)
            crit_values.append(crit_value)
            rmse_values.append(rmse)
            rmse_values_predictions.append(rmse_predictions)
        return ExperimentResults(
            rmse_values,
            rmse_values_predictions,
            jnp.array(designs),
            crit_values,
            design_space=self.design_space,
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
        param_estimate = self.ekf.state_prior[0].reshape(-1, 1)

        H = self.model.jacobian(param_estimate, self.data.x_val)
        HT = jnp.matrix_transpose(H)
        pred_vars = H @ sigma @ HT + self.measurement_error
        rmse_pred = jnp.sqrt(jnp.mean(pred_vars))
        # if rmse_pred > 10:
        #     breakpoint()
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

    def calculate_rmse_predictions(self, predictions, true_measurements):
        """
        Calculate root mean squared error for predictions.

        Args:
            predictions: Predicted measurement values
            true_measurements: True measurement values

        Returns:
            Average RMSE across all prediction locations (scalar)
        """

        return (
            jnp.mean((predictions.squeeze() - true_measurements.squeeze()) ** 2) ** 0.5
        )

    def run_experiment(
        self,
        experiments=["EPIG", "EIG", "EPIG-MC", "RAND"],
        iterations=10,
        optimizer_method="brute_force",
        optimizer_params={"lr": 1, "max_iters": 50},
    ):
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
        results = []
        instances = [deepcopy(self) for _ in experiments]
        if self.plot_2D_inter_results:
            fig, axes = plt.subplots(
                6, len(experiments), figsize=(165 / 25.4, 165 / 25.4), sharex=True
            )
            fig.subplots_adjust(hspace=0.3, wspace=0.536, right=0.85)
            for i, experiment in enumerate(experiments):
                axes[0, i].set_title(f"{experiment} Strategy")
                axes[-1, i].set_xlabel("Design (x)")
            for i in range(6):
                axes[i, 0].set_ylabel("y")

        for i, experiment in enumerate(experiments):
            self_copy = instances[i]
            try:
                r = self_copy.run(
                    criterion_label=experiment,
                    epochs=iterations,
                    optimizer=optimizer_method,
                    optimizer_params=optimizer_params,
                    axes_2D=axes[:, i]
                    if self.plot_2D_inter_results and len(experiments) > 1
                    else None,
                )
                results.append(r)
            except Exception as e:
                raise (e)
                print(e)

        return MultiExperimentResults(results)

    def plot_crit(
        self,
        ax: plt.Axes,
        crit_fn,
        title="EKF Sequential Design",
        grid_size=40,
        new_design=None,
        previous_designs=None,
    ):
        pool_min = self.design_space[..., 0].min()
        pool_max = self.design_space[..., 0].max()
        distance = pool_max - pool_min
        x_range = jnp.linspace(
            pool_min,  # - 0.2 * distance,
            pool_max,
        )
        x_train = self.data.x_train
        predictions_train = self.model(
            self.ekf.state_prior[0].reshape(-1, 1), x_train
        ).squeeze()
        sorted_desin_space = jnp.sort(self.design_space.squeeze())
        predictions_pool, prediction_variance_pool = self.ekf.measurement_prior(
            sorted_desin_space[:, None, None, None]
        )
        predictions_pool = predictions_pool.squeeze()
        predicion_validate, _ = self.ekf.measurement_prior(self.data.x_val)
        prediction_sigma_pool = jnp.sqrt(prediction_variance_pool.squeeze())
        measurements = self.data.observe(x_range, has_noise=False).squeeze()
        predictions, prediction_variance = self.ekf.measurement_prior(
            x_range[:, None, None, None]
        )
        predictions = predictions.squeeze()
        prediction_sigma = jnp.sqrt(prediction_variance.squeeze())
        ax.plot(
            x_range,
            measurements,
            label="True Measurements",
            color="black",
        )
        ax.scatter(
            sorted_desin_space.squeeze(),
            predictions_pool,
            # label="Design Pool Prdictions",
            marker="o",
            color="black",
            alpha=0.2,
            # s=50,
        )
        # ax.scatter(
        #     self.data.x_val.squeeze(), predicion_validate.squeeze(), color="orange"
        # )
        # ax.scatter(
        #     x_train.squeeze(),
        #     predictions_train,
        #     label="Training Designs",
        #     color="orange",
        #     marker="o",
        #     s=75,
        # )
        ax.plot(
            x_range,
            predictions,
            color="black",
            linestyle="--",
            label="Predictions",
        )

        ax.fill_between(
            x_range,
            (predictions - 1 * prediction_sigma).squeeze(),
            (predictions + 1 * prediction_sigma).squeeze(),
            color="tab:gray",
            alpha=0.2,
            label="90% Confidence Interval",
        )
        if previous_designs is not None:
            previous_measurements = self.data.observe(previous_designs)
            ax.scatter(
                previous_designs[..., 0].squeeze(),
                previous_measurements.squeeze(),
                color="tab:brown",
                marker="o",
                label="Observed Designs",
                alpha=0.7,
                # s=100,
            )

        if new_design is not None:
            new_measurements = self.data.observe(new_design)
            ax.scatter(
                new_design[..., 0].squeeze(),
                new_measurements.squeeze(),
                color="tab:red",
                marker="X",
                label="Optimal Design",
                s=75,
            )
            ax.vlines(
                new_design[..., 0].squeeze(),
                ymin=ax.get_ylim()[0],
                ymax=new_measurements.squeeze(),
                color="tab:red",
                linestyle="--",
            )
        ax_criterion = ax.twinx()
        crit_values = crit_fn(x_range[:, None, None, None]).squeeze()
        crit_values_normalized = (crit_values - crit_values.min()) / (
            crit_values.max() - crit_values.min() + 1e-8
        )
        ax_criterion.plot(
            x_range,
            crit_values,
            color="tab:blue" if crit_fn == self.calculate_epig else "tab:orange",
        )
        # fmt_small = FormatStrFormatter("%0e")
        fmt = FuncFormatter(lambda x, _: f"{x:.0e}" if abs(x) < 1e-2 else f"{x:.2f}")
        ax_criterion.yaxis.set_major_formatter(fmt)
        return ax_criterion

    def plot_crit_surface(
        self,
        title="EPIG, EIG and EPIG-MC Surfaces",
        x_range=None,
        y_range=None,
        grid_size=20,
        new_design=None,
        previous_designs=None,
        axes=None,
        fig_size=(22, 13),
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
        grid_points = jnp.concatenate(
            [xx1.flatten()[:, None, None, None], xx2.flatten()[:, None, None, None]],
            axis=-1,
        )
        figsize_cm = fig_size
        figsize_inches = (figsize_cm[0], figsize_cm[1])
        if axes is None:
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=figsize_inches)
        crit_values_epig = self.calculate_epig(grid_points).reshape(xx1.shape)
        crit_values_eig = self.calculate_eig(grid_points).reshape(xx1.shape)
        crit_values_mc = self.calculate_epig_mc(grid_points).reshape(xx1.shape)
        fmt = FormatStrFormatter("%.2f")
        c = axes[0].contourf(xx1, xx2, crit_values_epig, levels=50, cmap="Blues")
        axes[0].scatter(
            self.design_space[..., 0].squeeze(),
            self.design_space[..., 1].squeeze(),
            c="black",
            alpha=0.2,
            label="Design Pool",
        )
        cbar = plt.colorbar(c, ax=axes[0], format=fmt)
        cbar.locator = MaxNLocator(nbins=3)
        cbar.update_ticks()
        c = axes[2].contourf(xx1, xx2, crit_values_eig, levels=50, cmap="Oranges")
        cbar = plt.colorbar(c, ax=axes[2], format=fmt)
        cbar.locator = MaxNLocator(nbins=3)
        cbar.update_ticks()
        axes[2].scatter(
            self.design_space[..., 0].squeeze(),
            self.design_space[..., 1].squeeze(),
            c="black",
            label="Design Pool",
            marker="o",
            alpha=0.2,
        )
        c = axes[1].contourf(xx1, xx2, crit_values_mc, levels=50, cmap="Greens")
        cbar = plt.colorbar(c, ax=axes[1], format=fmt)
        cbar.locator = MaxNLocator(nbins=3)
        cbar.update_ticks()
        axes[1].scatter(
            self.design_space[..., 0].squeeze(),
            self.design_space[..., 1].squeeze(),
            c="black",
            label="Design Pool",
            marker="o",
            alpha=0.2,
        )

        if previous_designs is not None:
            axes[0].scatter(
                previous_designs[..., 0].squeeze(),
                previous_designs[..., 1].squeeze(),
                c="tab:brown",
                label="Added Designs",
                marker="o",
                # s=100,
                alpha=0.7,
            )
            axes[1].scatter(
                previous_designs[..., 0].squeeze(),
                previous_designs[..., 1].squeeze(),
                c="tab:brown",
                marker="o",
                # s=100,
                alpha=0.7,
            )
            axes[2].scatter(
                previous_designs[..., 0].squeeze(),
                previous_designs[..., 1].squeeze(),
                c="tab:brown",
                marker="o",
                # s=100,
                alpha=0.7,
            )
        if new_design is not None:
            axes[0].scatter(
                new_design[..., 0].squeeze(),
                new_design[..., 1].squeeze(),
                c="tab:red",
                label="New Design",
                marker="X",
                s=70,
            )
            axes[1].scatter(
                new_design[..., 0].squeeze(),
                new_design[..., 1].squeeze(),
                c="tab:red",
                marker="X",
                s=70,
            )
            axes[2].scatter(
                new_design[..., 0].squeeze(),
                new_design[..., 1].squeeze(),
                c="tab:red",
                marker="X",
                s=70,
            )


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
        rmse_values,
        rmse_values_predictions,
        designs,
        crit_values,
        crit_label="EPIG",
        design_space=None,
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
        self.rmse_values = rmse_values
        self.rmse_values_predictions = rmse_values_predictions
        self.designs = designs
        self.crit_values = crit_values
        self.crit_label = crit_label
        self.design_space = design_space

    def plot_results(self):
        """
        Plot experiment results showing criterion and RMSE evolution.

        Creates a 3-panel figure showing:
        - Top: Criterion values over iterations
        - Bottom left: Parameter estimation RMSE over iterations
        - Bottom right: Prediction RMSE over iterations

        Useful for assessing convergence and comparing design strategies.
        """
        fig = plt.figure(figsize=(13, 6))
        gs = GridSpec(2, 2, figure=fig)
        ax_crit = fig.add_subplot(gs[0, :])
        ax_crit.plot(self.crit_values, marker="o")
        ax_crit.set_title(f"{self.crit_label} Values over Iterations")
        ax_crit.set_xlabel("Iteration")
        ax_crit.set_ylabel(self.crit_label)

        ax_rmse = fig.add_subplot(gs[1, :])
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

    def plot_comparison(self, width_mm=90, height_mm=160):
        """
        Plot prediction RMSE comparison across all experiments.

        Creates a single plot with overlaid curves showing how prediction
        RMSE evolves over iterations for each design strategy. Useful for
        visually comparing which criterion leads to faster learning.

        visually comparing which criterion leads to faster learning.

        Note:
            Each curve is labeled with its criterion name (from crit_label).
        """
        fig, axes = plt.subplots(
            3, 1, figsize=(height_mm / 25.4, width_mm / 25.4), sharex=True
        )
        plt.rcParams.update({"font.size": 11})
        fig.subplots_adjust(hspace=0.5)
        crit_values = jnp.array(
            [result.crit_values for result in self.experiment_results_list]
        )
        crit_values_normalized = (
            crit_values - crit_values.min(axis=1, keepdims=True)
        ) / (
            crit_values.max(axis=1, keepdims=True)
            - crit_values.min(axis=1, keepdims=True)
        )
        for i, result in enumerate(self.experiment_results_list):
            axes[0].plot(result.rmse_values, label=result.crit_label)
            axes[1].plot(result.rmse_values_predictions, label=result.crit_label)
            if result.crit_label.upper() == "RAND":
                break
            axes[2].plot(crit_values_normalized[i], label=result.crit_label)

        axes[0].set_title("Estimated Predictive Standard Error")
        # axes[0].set_ylabel("")
        # axes[0].legend()
        axes[1].set_title("Root Mean Squared Error")

        axes[2].set_title("Normalized Criterion Values")
        axes[2].set_xlabel("Iteration")

    def plot_design_distribution(self, width_mm=80, height_mm=160):
        fig, ax = plt.subplots(
            figsize=(height_mm / 25.4, width_mm / 25.4),
        )
        x_range = jnp.linspace(
            self.experiment_results_list[0].design_space.min(),
            self.experiment_results_list[0].design_space.max(),
            100,
        )
        design_space_kde = gaussian_kde(
            self.experiment_results_list[0].design_space.squeeze()
        )
        # ax.set_title("Optimal Designs Distribution")
        ax.plot(
            x_range,
            design_space_kde(x_range),
            color="black",
            label="Design Pool",
        )

        for i, result in enumerate(self.experiment_results_list):
            kde = gaussian_kde(result.designs.squeeze())
            ax.plot(
                x_range,
                kde(x_range),
                label=f"{result.crit_label} Designs",
            )

        ax.set_xlabel("Design Value")
        # ax.legend()
