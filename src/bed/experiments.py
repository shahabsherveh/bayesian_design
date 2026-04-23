from copy import deepcopy
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from .ekf import EKF
from .models import LinearModel


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
        latent_dim,
        latent_var,
        latent_innovation,
        latent_true,
        design_cov,
        design_mean,
        design_pool_num,
        measurement_error,
        model=LinearModel(),
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
        self.model = model
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
        # mean = np.zeros((latent_dim, 1))
        mean = np.ones((latent_dim, 1))
        # mean = np.random.normal(loc=0, scale=0.0000001, size=(latent_dim, 1))
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
        if x.ndim == 1:
            x = x[:, None]
        if x_1 is None:
            x_1 = self.design_space.T
        ekf = self.ekf
        state_prev = ekf.state_prior[0]
        j_1 = ekf.model.jacobian(state_prev.reshape(-1, 1), x_1)
        j_0 = ekf.model.jacobian(state_prev.reshape(-1, 1), x)
        sigma = ekf.state_prior[1]
        s_x = ekf.measurement_prior(x)[1]
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
        mean_1 = jax.vmap(lambda theta: self.model(theta.T, x_1))(
            latent_samples
        ).squeeze()
        mean_0 = jax.vmap(lambda theta: self.model(theta.T, x_0))(
            latent_samples
        ).squeeze()
        epsilon_1 = mean_1 - y_1
        epsilon_0 = (mean_0 - y_0).T

        def get_normal_likelihood(epsilon):
            cov = self.measurement_error
            return (1 / jnp.sqrt(2 * jnp.pi * cov)) * jnp.exp(-0.5 * (epsilon**2) / cov)

        y_0_pdf_vals = get_normal_likelihood(epsilon_0)
        y_1_pdf_vals = get_normal_likelihood(epsilon_1)
        mi = jnp.log(
            (y_0_pdf_vals * y_1_pdf_vals).mean(axis=0)
            / (y_0_pdf_vals.mean(axis=0) * y_1_pdf_vals.mean(axis=0))
        )
        return mi

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
            x_1 = self.design_space.T
        M = x_1.shape[1]
        outcome_latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=M)
        noise_0 = np.random.normal(
            loc=0,
            scale=np.sqrt(self.measurement_error),
            size=(M, x.shape[1]),
        ).squeeze()
        noise_1 = np.random.normal(
            loc=0,
            scale=np.sqrt(self.measurement_error),
            size=(1, M),
        )
        y_0_samples = (
            jax.vmap(lambda theta: self.model(theta.T, x))(
                outcome_latent_samples
            ).squeeze()
            + noise_0
        )[:, None]
        y_1_samples = (
            jax.vmap(lambda theta: self.model(theta.T, x_1))(
                outcome_latent_samples
            ).squeeze()
            + noise_1
        )
        y_1_samples = y_1_samples.diagonal().T

        latent_samples = multivariate_normal(
            mean=self.ekf.state_prior[0].flatten(), cov=self.ekf.state_prior[1]
        ).rvs(size=num_latent_samples)
        mi = self.calculate_mutual_information_mc(
            y_1_samples,
            x_1,
            y_0_samples,
            x,
            latent_samples=latent_samples,
        )
        return mi.mean()

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
        state_prior_cov = self.ekf.state_prior[1]
        measurement_error = self.measurement_error

        def J(x):
            return self.model.jacobian(z=self.ekf.state_prior[0], x=x.reshape(-1, 1))

        eig = jnp.log((J(x) @ state_prior_cov @ J(x).T / measurement_error) + 1) / 2
        return eig[0, 0]

    def calculate_random(self, x, key=jax.random.key(0), **kwargs):
        val = jax.random.uniform(
            key=key,
        )
        return val

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
            criterion_func_vectorized = jax.vmap(criterion_func)
            seeds = np.random.randint(
                0, 10000, size=(self.design_space.shape[0],)
            )  # Random keys for randomness in criterion
            keys = jax.vmap(jax.random.key)(seeds)
            pool_values = criterion_func_vectorized(x=self.design_space, key=keys)
            best_index = jnp.argmax(pool_values)
            x = self.design_space[[best_index]].T
        # Randomly initialize from a normal distribution
        elif x_init_type == "normal":
            x = np.random.normal(loc=0, scale=1, size=(self.design_space.shape[1], 1))

        # Otherwise initialize with zero
        else:
            x = np.zeros((self.design_space.shape[1], 1))  # Default to zero vector

        crit_value = criterion_func(
            x=x, x_1=self.design_space.T
        )  # Initial criterion value
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
        elif criterion_label.upper() == "MC":
            criterion_func = self.calculate_epig_mc
        elif criterion_label.upper() == "RAND":
            criterion_func = self.calculate_random
        designs = []
        crit_values = []
        rmse_values = []
        rmse_params_values = []
        rmse_values_predictions = []
        grad_lists = []
        progress_bar = tqdm(
            range(epochs), total=epochs, desc=f"Running {criterion_label} Experiment"
        )
        fig, axes = plt.subplots(
            figsize=(8, 12), nrows=epochs, ncols=2, sharex=True, sharey=True
        )
        axes[0, 0].set_title("EPIG Surface")
        axes[0, 1].set_title("EIG Surface")
        fig.suptitle(f"{criterion_label} optimization", fontsize=16)
        for i in progress_bar:
            x_opt, crit_value, grads = self.optimize(
                criterion_func=criterion_func,
                criterion_label=criterion_label,
                **optimizer_params,
            )
            measurement = self.model(self.latent_true, x_opt)
            # + np.random.normal(
            # 0, np.sqrt(self.measurement_error)
            # )
            if self.plot_results:
                self.plot_crit_surface(
                    title=f"{criterion_label} optimization",
                    new_design=x_opt,
                    previous_designs=jnp.array(designs) if designs else None,
                    axes=axes[i],
                )
            ekf.state_prior = ekf.get_state_posterior(measurement, x_opt)
            # latent_estimates = multivariate_normal(
            #     mean=ekf.state_prior[0].flatten(), cov=ekf.state_prior[1]
            # ).rvs(size=1000)
            estimate_mean, estimate_cov = ekf.state_prior
            predictions = self.model(estimate_mean.reshape(-1, 1), self.design_space.T)
            true_measurements = self.model(self.latent_true, self.design_space.T)
            rmse = self.calculate_rmse()
            rmse_predictions = self.calculate_rmse_predictions(
                predictions, true_measurements
            )
            rmse_params = self.calculate_rmse_params(estimate_mean, self.latent_true)
            grad_lists.append(grads)
            progress_bar.set_postfix(
                {"Prediction RMSE": rmse, "Parameter RMSE": rmse_params}
            )
            designs.append(x_opt)
            crit_values.append(crit_value)
            rmse_values.append(rmse)
            rmse_values_predictions.append(rmse_predictions)
            rmse_params_values.append(rmse_params)
        return ExperimentResults(
            rmse_params_values,
            rmse_values,
            rmse_values_predictions,
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
        param_estimate = self.ekf.state_prior[0].reshape(-1, 1)

        def j(x):
            return self.model.jacobian(z=param_estimate, x=x[:, None])

        pred_vars = jax.vmap(lambda x: j(x) @ sigma @ j(x).T + self.measurement_error)(
            self.design_space
        )
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
        return jnp.sqrt(jnp.mean((predictions - true_measurements.flatten()) ** 2))

    def run_experiment(
        self,
        experiments=["EPIG", "EIG", "MC", "RAND"],
        iterations=10,
        optimizer_params={},
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
        for experiment in experiments:
            self_copy = deepcopy(self)
            try:
                r = self_copy.run(
                    criterion_label=experiment,
                    epochs=iterations,
                    optimizer_params=optimizer_params,
                )
                results.append(r)
            except Exception as e:
                print(f"Error running {experiment} experiment: {e}")
                pass
        return MultiExperimentResults(results)

    def plot_crit_surface(
        self,
        title="EPIG and EIG Surfaces",
        x_range=None,
        y_range=None,
        grid_size=20,
        new_design=None,
        previous_designs=None,
        axes=None,
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
        if axes is None:
            fig, axes = plt.subplots(1, 2, sharey=True, figsize=(8, 12))
        crit_values_epig = jax.vmap(self.calculate_epig)(grid_points).reshape(xx1.shape)
        crit_values_eig = jax.vmap(self.calculate_eig)(grid_points).reshape(xx1.shape)
        c = axes[0].contourf(xx1, xx2, crit_values_epig, levels=50, cmap="viridis")
        axes[0].scatter(
            self.design_space[:, 0],
            self.design_space[:, 1],
            c="black",
            label="Design Pool",
        )
        plt.colorbar(c, label="EPIG", ax=axes[0])
        axes[0].set_xlabel("Design Dimension 1")
        axes[0].set_ylabel("Design Dimension 2")
        c = axes[1].contourf(xx1, xx2, crit_values_eig, levels=50, cmap="viridis")
        axes[1].scatter(
            self.design_space[:, 0],
            self.design_space[:, 1],
            c="black",
        )
        plt.colorbar(c, label="EIG", ax=axes[1])
        if previous_designs is not None:
            axes[0].scatter(
                previous_designs[:, 0],
                previous_designs[:, 1],
                c="blue",
                label="Added Designs",
                marker="o",
                s=100,
            )
            axes[1].scatter(
                previous_designs[:, 0],
                previous_designs[:, 1],
                c="blue",
                marker="o",
                s=100,
            )
        if new_design is not None:
            axes[0].scatter(
                new_design[0],
                new_design[1],
                c="red",
                label="New Design",
                marker="X",
                s=100,
            )
            axes[1].scatter(
                new_design[0],
                new_design[1],
                c="red",
                marker="X",
                s=100,
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
        rmse_params_values,
        rmse_values,
        rmse_values_predictions,
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
        self.rmse_values_predictions = rmse_values_predictions
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
            ax.plot(result.rmse_values_predictions, marker="o", label=result.crit_label)
        ax.set_title("Comparison of RMSE over Iterations")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RMSE Value")
        ax.legend()
