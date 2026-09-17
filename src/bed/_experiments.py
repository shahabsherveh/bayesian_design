from copy import deepcopy
from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
from scipy.stats import gaussian_kde, multivariate_normal
from tqdm import tqdm
from .ekf import EKF
from .ukf import UKF
from .models import LinearModel, Model, NeuralNetworkRegressor
from .data import Data


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
        ekf: Selected Kalman filter for state estimation
        plot_results: Whether to visualize results during optimization
    """

    def __init__(
        self,
        latent_cov=None,
        latent_innovation=0,
        measurement_error=None,
        data: Data = None,
        model: NeuralNetworkRegressor = None,
        latent_var=None,
        warm_start: int = 10,
        pre_train_model=False,
        training_kwargs=None,
        plot_inter_results=False,
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
            filter_type: State estimator to use, either ``"ekf"`` or ``"ukf"``.
        """
        if latent_cov is None:
            if latent_var is None:
                raise TypeError(
                    "Either latent_cov or latent_var must be provided.")
            latent_cov = latent_var * jnp.eye(model.flax_model.weight_size)
        if measurement_error is None:
            raise TypeError("measurement_error must be provided.")
        if training_kwargs is None:
            training_kwargs = {
                "epochs": 200,
                "learning_rate": 0.01,
                "rngs": nnx.Rngs(0),
            }

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
        self.design_pool, self.design_space, self.true_measurements = self.build_design_space(
            data)
        self.measurement_error = (
            # if not pre_train_model else model.mse * jnp.eye(1)
            measurement_error
        )
        self.latent_innovation = latent_innovation
        self.warm_start = warm_start

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
        design_pool = data.x_test_pool
        design_space = jnp.concatenate([data.x_train], axis=0)
        true_measurements = jnp.concatenate(
            [data.y_train], axis=0)
        return design_pool, design_space, true_measurements

    def calculate_epig(self, x, filtr, x_1=None, **kwargs):
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
            x_1 = self.design_pool
        result = filtr.calculate_epig(x, x_1)
        return result

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
        mean_1 = jax.vmap(lambda theta: self.model(
            theta.T, x_1))(latent_samples)
        mean_0 = jax.vmap(lambda theta: self.model(theta.T, x_0))(
            latent_samples
        ).swapaxes(1, 2)
        epsilon_1 = mean_1 - y_1
        epsilon_0 = mean_0 - y_0

        def get_normal_likelihood(epsilon):
            """Evaluate the scalar Gaussian observation likelihood."""
            cov = self.measurement_error
            return (1 / jnp.sqrt(2 * jnp.pi * cov)) * jnp.exp(-0.5 * (epsilon**2) / cov)

        y_0_pdf_vals = get_normal_likelihood(epsilon_0)
        y_1_pdf_vals = get_normal_likelihood(epsilon_1)
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

    def calculate_eig(self, x, filtr, *arg, **kwargs):
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
        eig = filtr.calculate_eig(x, *arg, **kwargs)
        return jnp.atleast_1d(eig.squeeze())

    def calculate_random(self, x, key, **kwargs):
        """Return one uniformly distributed utility value per candidate."""
        val = nnx.vmap(
            lambda k: jax.random.uniform(
                key=k,
            )
        )(key)
        return val

    def optimize(self, criterion_func, method, filtr, params={"lr": 1, "max_iters": 50}):
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
            pool_values = criterion_func(
                x=self.design_space, filtr=filtr, key=keys)
            shuffled_indices = jax.random.permutation(
                jax.random.key(0), self.design_space.shape[0]
            )
            # shuffled_indices = jnp.arange(self.design_space.shape[0])
            pool_values_shuffled = pool_values[shuffled_indices]
            best_position = jnp.argmax(pool_values_shuffled)
            best_index = shuffled_indices[best_position]
            x = self.design_space[best_index]
            crit_value = pool_values_shuffled[best_position]
        elif method == "gradient_ascent":
            x = np.random.normal(
                size=self.design_space[:1].shape,
                scale=0.01,
            )
            grad_func = jax.value_and_grad(
                lambda x: criterion_func(x, key=keys)[0])
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
        filter_type,
        filter_params,
        filter_instance,
        epochs,
        optimizer="grid_search",
        optimizer_params={"lr": 1, "max_iters": 50},
        trace=False
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
        criterion_dict = {
            "EPIG": self.calculate_epig,
            "EIG": self.calculate_eig,
            "EPIG-MC": self.calculate_epig_mc,
            "RANDOM": self.calculate_random}
        criterion_func = criterion_dict.get(
            criterion_label, self.calculate_random
        )
        selected_designs = []
        crit_values = []
        sd_values_test_pool = []
        rmse_values_test_pool = []
        sd_values_test_glob = []
        rmse_values_test_glob = []
        filters = []

        progress_bar = tqdm(
            range(epochs), total=epochs, desc=f"Running {criterion_label} Experiment"
        )
        filtr = filter_instance
        design_space = jnp.linspace(
            self.design_space.min(), self.design_space.max()
        )
        for i in progress_bar:
            estimate_mean, estimate_cov = filtr.state_prior
            mean_test_pool, cov_test_pool = filtr.measurement_prior(
                self.data.x_test_pool)
            mean_test_glob, cov_test_glob = filtr.measurement_prior(
                self.data.x_test_glob)
            if trace:
                filters.append(deepcopy(filtr))
            rmse_predictions_pool = self.calculate_rmse_predictions(
                mean_test_pool, self.data.y_test_pool.squeeze()
            )
            rmse_predictions_glob = self.calculate_rmse_predictions(
                mean_test_glob, self.data.y_test_glob.squeeze()
            )
            sd_pool = self.calculate_rmse(cov_test_pool)
            sd_glob = self.calculate_rmse(cov_test_glob)

            best_index, x_opt, crit_value = self.optimize(
                criterion_func=criterion_func,
                filtr=filtr,
                method=optimizer,
                params=optimizer_params,
            )
            measurement = self.data.observe(best_index)
            posterior = filtr.get_state_posterior(
                measurement, x_opt[None, ...])

            filtr.state_prior = posterior
            progress_bar.set_postfix(
                {
                    "Global SD": f"{sd_glob.round(2):.2f}",
                    "Pool SD": f"{sd_pool.round(2):.2f}",
                    "Global RMSE": f"{rmse_predictions_glob.round(2):.2f}",
                    "Pool RMSE": f"{rmse_predictions_pool.round(2):.2f}",
                    f"{criterion_label}": f"{crit_value.round(3):.2f}",
                }
            )
            selected_designs.append((x_opt, measurement))
            crit_values.append(crit_value)
            rmse_values_test_glob.append(rmse_predictions_glob)
            rmse_values_test_pool.append(rmse_predictions_pool)
            sd_values_test_glob.append(sd_glob)
            sd_values_test_pool.append(sd_pool)
        return ExperimentResults(
            sd_values_test_glob,
            sd_values_test_pool,
            rmse_values_test_glob,
            rmse_values_test_pool,
            selected_designs,
            crit_values,
            data=self.data,
            design_space=self.design_space,
            crit_label=criterion_label,
            filter_type=filter_type,
            filters=filters,
        )

    def calculate_rmse(self, cov):
        """
        Calculate root mean squared error for predictions.

        Args:
            cov: Predictive covariance matrix or batch of covariance matrices.

        Returns:
            Average RMSE across all prediction locations (scalar)
        """
        rmse_pred = (jnp.linalg.trace(cov)**.5).mean()
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
            jnp.mean((predictions.squeeze() -
                     true_measurements.squeeze()) ** 2) ** 0.5
        )

    def run_experiment(
        self,
        criteria=["EPIG", "EIG", "EPIG-MC", "RAND"],
        filter_types=["ukf", "ekf"],
        filter_params=[{}, {}],
        iterations=10,
        optimizer_method="brute_force",
        optimizer_params={"lr": 1, "max_iters": 50},
        trace=False
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
        for i, filter_type in enumerate(filter_types):
            filter_kwargs = filter_params[i]
            filter_dict = {'ukf': UKF, 'ekf': EKF}
            fclass = filter_dict[filter_type]
            filter_instance = fclass(
                model=self.model,
                state_prev=self.state_init_prior[0],
                state_cov_prev=self.state_init_prior[1],
                state_innovation=self.latent_innovation,
                measurement_error=self.measurement_error,
                **filter_params[i]
            )
            r = self.run(
                criterion_label="RANDOM",
                filter_type=filter_type,
                filter_params=filter_kwargs,
                filter_instance=filter_instance,
                epochs=self.warm_start,
                optimizer=optimizer_method,
                optimizer_params=optimizer_params,
                trace=trace
            )
            results.append(r)
            for j, criterion in enumerate(criteria):
                # Each criterion must start from the same warm-start posterior;
                # otherwise later strategies would inherit earlier observations.
                filter_instance_copy = deepcopy(filter_instance)
                try:
                    r = self.run(
                        criterion_label=criterion,
                        filter_type=filter_type,
                        filter_params=filter_kwargs,
                        filter_instance=filter_instance_copy,
                        epochs=iterations,
                        optimizer=optimizer_method,
                        optimizer_params=optimizer_params,
                        trace=trace
                    )
                    results.append(r)
                except Exception as e:
                    raise (e)
                    print(e)

        return MultiExperimentResults(results)


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
        sd_glob,
        sd_pool,
        rmse_global,
        rmse_pool,
        selected_designs,
        crit_values,
        data,
        crit_label="EPIG",
        filter_type="ekf",
        design_space=None,
        filters=[],
    ):
        """Initialize metric histories and selected designs.

        Args:
            rmse_params_values: Parameter RMSE values over epochs
            rmse_values: Prediction RMSE values over epochs
            designs: Selected designs for each epoch
            crit_values: Criterion values for each selected design
            grad_lists: Gradients from each optimization run
            crit_label: Label for the criterion used
        """
        self.sd_glob = sd_glob
        self.sd_pool = sd_pool
        self.rmse_glob = rmse_global
        self.rmse_pool = rmse_pool
        self.selected_designs = selected_designs
        self.crit_values = crit_values
        self.crit_label = crit_label
        self.filter_type = filter_type
        self.design_space = design_space
        self.filters = filters
        self.data = data


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
        self.experiment_results_dict = self._list_to_dict(
            experiment_results_list)

    @staticmethod
    def _list_to_dict(l):
        d = {}
        for e in l:
            d[e.filter_type] = {}
        for e in l:
            d[e.filter_type][e.crit_label] = e

        return d
