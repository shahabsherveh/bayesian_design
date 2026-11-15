from copy import deepcopy
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from scipy.stats import multivariate_normal
from tqdm import tqdm

from .data import Data
from .ekf import EKF
from .empirical_bayes import empirical_bayes_init
from .models import NeuralNetworkRegressor
from .ukf import UKF


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
        model: Measurement model (a NeuralNetworkRegressor or compatible)
        state_init_prior: Initial prior over latent parameters, (mean, cov)
        data: Dataset container providing train/test pools and observations
        design_pool: Candidate design points for selection
        design_space: Pool of designs combined with the training inputs
        true_measurements: Corresponding observations for the design space
        measurement_error: Observation noise (co)variance
        latent_innovation: Process noise covariance
        warm_start: Number of warm-start design selections
        seed: Random seed supplied to the runner's generator
        init: How the warm-start belief is formed ("filter" or "empirical_bayes")
        empirical_bayes_kwargs: Options passed to the empirical Bayes initialisation
        _rng: NumPy generator reused across (filter, criterion) runs
    """

    def __init__(
        self,
        latent_cov=None,
        latent_innovation=0,
        measurement_error=None,
        data: Data | None = None,
        model: NeuralNetworkRegressor | None = None,
        latent_var=None,
        warm_start: int = 10,
        pre_train_model=False,
        training_kwargs: dict[str, Any] | None = None,
        plot_inter_results=False,
        seed: int = 0,
        init: str = "filter",
        empirical_bayes_kwargs=None,
        latent_mean=None,
    ):
        """
        Initialize sequential experimental design framework.

        Args:
            latent_cov: Prior covariance over latent parameters. If None, derived
                from ``latent_var`` (see below); one of the two is required.
            latent_innovation: Process noise covariance (scalar or matrix).
            measurement_error: Observation noise (co)variance. Required.
            data: Dataset container with train/test pools and observations.
            model: Measurement model exposing a ``flax_model`` (and its
                ``weight_size``). Required unless ``latent_mean`` is given.
            latent_var: Scalar prior variance; used to build ``latent_cov`` when
                ``latent_cov`` is None. Mutually exclusive with ``latent_cov``.
            warm_start: Number of warm-start design selections.
            pre_train_model: If True, train ``model`` on ``data`` before use.
            training_kwargs: Passed to ``model.train`` when ``pre_train_model``.
            plot_inter_results: If True, plot intermediate results during runs.
            seed: Random seed for reproducible (filter, criterion) runs.
            init: How the warm-start belief is formed: ``"filter"`` (process the
                warm-start observations with the filter) or ``"empirical_bayes"``
                (fit the prior scale, and optionally noise, by marginal likelihood).
            empirical_bayes_kwargs: Options for the empirical Bayes initialisation.
            latent_mean: Prior mean for latent parameters. Required for models
                without a ``flax_model``; otherwise inferred from the model weights.
        """
        if data is None:
            raise TypeError("data must be provided.")
        if model is None:
            raise TypeError("model must be provided.")
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
            latent_mean=latent_mean,
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
        self.seed = seed
        # How the belief after the warm start is formed: "filter" processes the warm-start
        # observations with the filter itself; "empirical_bayes" fits the prior scale (and
        # optionally the noise scale) by marginal likelihood on those observations and starts
        # every criterion from the Laplace belief at the MAP (see bed.empirical_bayes).
        if init not in ("filter", "empirical_bayes"):
            raise ValueError(
                f"Unknown init '{init}'; choose 'filter' or 'empirical_bayes'.")
        self.init = init
        self.empirical_bayes_kwargs = {} if empirical_bayes_kwargs is None else dict(
            empirical_bayes_kwargs)
        # Random source for the RANDOM criterion and Monte Carlo EPIG; `run`
        # re-seeds it so every (filter, criterion) run is reproducible.
        self._rng = np.random.default_rng(seed)

    @staticmethod
    def build_prior(
        latent_cov,
        model: NeuralNetworkRegressor,
        latent_mean=None,
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
        if latent_mean is not None:
            return jnp.asarray(latent_mean).reshape(-1, 1), latent_cov
        if not hasattr(model, "flax_model"):
            raise TypeError(
                "latent_mean is required for models without Flax parameters.")
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
        mean_1 = jnp.expand_dims(jax.vmap(lambda theta: self.model(
            theta.T, x_1))(latent_samples), axis=2)
        mean_0 = jnp.expand_dims(jax.vmap(lambda theta: self.model(theta.T, x_0))(
            latent_samples), axis=1
        )
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

    def _sample_latent(self, filtr, size, rng=None):
        """Draw parameter samples from the filter's Gaussian belief with the run's generator."""
        local_rng = rng if rng is not None else self._rng
        state_mean, state_cov = filtr.state_prior
        return jnp.asarray(multivariate_normal(
            mean=np.asarray(state_mean).flatten(), cov=np.asarray(state_cov)
        ).rvs(size=size, random_state=local_rng)).reshape(size, -1)

    def calculate_epig_mc(self, x, filtr, x_1=None, num_latent_samples=1000, rng=None, **kwargs):
        """Nested Monte Carlo estimate of EPIG (Bickford Smith et al., 2023) for any output
        dimension with a diagonal observation covariance: one outer sample (theta_j, y_j, y'_j)
        per pool point, ``num_latent_samples`` inner samples for the marginal and joint densities,
        log-sum-exp throughout. Strongly biased downward for small inner sample sizes in more than
        one output dimension; see tests/mechanistic_test.py for its convergence check."""
        if x.ndim == 3:
            x = x[None, ...]
        if x_1 is None:
            x_1 = self.design_pool
        local_rng = rng if rng is not None else self._rng

        M, B, K = x_1.shape[0], x.shape[0], num_latent_samples
        R = jnp.atleast_2d(jnp.asarray(self.measurement_error))

        state_mean, state_cov = filtr.state_prior

        # diagonal observation noise
        r = jnp.diagonal(R)
        d_y = r.size
        # one outer sample (theta_j, y_j, y'_j) per pool point, K inner samples for the densities
        theta_outer = self._sample_latent(filtr, M, rng=local_rng)
        theta_inner = self._sample_latent(filtr, K, rng=local_rng)
        f_out_x = jax.vmap(lambda th: self.model(th, x))(
            theta_outer).reshape(M, B, d_y)
        f_out_x1 = jax.vmap(lambda th: self.model(th, x_1))(
            theta_outer).reshape(M, M, d_y)
        # pair pool point j with theta_j: (M, d_y)
        f_out_x1 = f_out_x1[jnp.arange(M), jnp.arange(M)]
        y_0 = f_out_x + jnp.sqrt(r) * \
            jnp.asarray(
            local_rng.normal(size=(M, B, d_y)))
        y_1 = f_out_x1 + jnp.sqrt(r) * \
            jnp.asarray(local_rng.normal(size=(M, d_y)))
        f_in_x = jax.vmap(lambda th: self.model(th, x))(
            theta_inner).reshape(K, B, d_y)
        f_in_x1 = jax.vmap(lambda th: self.model(th, x_1))(
            theta_inner).reshape(K, M, d_y)
        # (K, M, B)
        logp_0 = -0.5 * \
            jnp.sum((y_0[None] - f_in_x[:, None]) ** 2 / r, axis=-1)
        # (K, M, 1)
        logp_1 = -0.5 * jnp.sum((y_1[None] - f_in_x1)
                                ** 2 / r, axis=-1)[..., None]
        log_k = jnp.log(K)
        log_joint = jax.scipy.special.logsumexp(
            logp_0 + logp_1, axis=0) - log_k
        log_marg_0 = jax.scipy.special.logsumexp(logp_0, axis=0) - log_k
        log_marg_1 = jax.scipy.special.logsumexp(logp_1, axis=0) - log_k
        # (M, B)
        mi = log_joint - log_marg_0 - log_marg_1
        return jnp.atleast_1d(mi.mean(axis=0))

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

    def optimize(self, criterion_func, method, filtr, params=None, exclude=None, rng=None):
        """
        Select the design that maximizes an information criterion.

        Supports three methods: exhaustive pool search (``brute_force``), gradient
        ascent via JAX automatic differentiation (``gradient_ascent``), and a
        reduced-budget pool search (``grid_search``).

        Args:
            criterion_func: Criterion to maximise, with signature
                ``func(x, filtr=..., key=..., rng=...)`` returning values over ``x``.
            method: One of ``"brute_force"``, ``"gradient_ascent"``, or
                ``"grid_search"``.
            filtr: Filter whose belief defines the criterion.
            params: Method-specific options (e.g. ``lr``, ``max_iters``,
                ``num_samples``). Defaults to ``{"lr": 1, "max_iters": 50}``.
            exclude: Set of pool indices that may not be selected. Applies to all
                methods: ``brute_force``/``grid_search`` mask them before the
                argmax, ``gradient_ascent`` snaps to the nearest non-excluded
                pool point.
            rng: NumPy generator for criterion keys and randomness. Defaults to
                ``self._rng``.

        Returns:
            Tuple ``(best_index, x, crit_value)`` where:
                - ``best_index``: integer index into ``self.design_space``
                - ``x``: the selected design, ``self.design_space[best_index]``
                - ``crit_value``: criterion value at the selected design
        """
        if params is None:
            params = {"lr": 1, "max_iters": 50}
        local_rng = rng if rng is not None else self._rng
        keys = self._criterion_keys(self.design_space.shape[0], rng=local_rng)
        if method == "brute_force":
            pool_values = jnp.asarray(criterion_func(
                x=self.design_space, filtr=filtr, key=keys, rng=local_rng)).reshape(-1)
            if exclude:
                # Pool-based selection is without replacement: a design whose
                # label is already in the filter cannot be selected again.
                if len(exclude) >= pool_values.shape[0]:
                    raise ValueError(
                        "Every design in the pool has been observed.")
                pool_values = pool_values.at[jnp.asarray(
                    sorted(exclude))].set(-jnp.inf)
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
            x = local_rng.normal(
                size=self.design_space[:1].shape,
                scale=0.01,
            )
            grad_func = jax.value_and_grad(
                lambda x: criterion_func(x, filtr=filtr, key=keys, rng=local_rng)[0])
            max_iters = int(params.get("max_iters", 50))
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
            # gradient ascent explores the continuous design box; snap the final
            # iterate to the nearest allowed pool point so the result is a real,
            # observable design with a pool index like the other methods.
            best_index = self._nearest_pool_index(x.squeeze(0), exclude)
            x = self.design_space[best_index]
            crit_value = jnp.asarray(criterion_func(
                x=x[None, ...], filtr=filtr, key=keys[:1], rng=local_rng)).reshape(-1)[0]
        elif method == "grid_search":
            num_samples = params.get("num_samples", 200)
            # Sample candidate designs from the pool with replacement, so the
            # result is a real (observable) design and has a pool index.
            best_index, crit_value = self._pool_sample_search(
                criterion_func, filtr, num_samples, exclude, rng=local_rng)
            x = self.design_space[best_index]

        return best_index, x, crit_value

    def _nearest_pool_index(self, x, exclude=None):
        """Return the pool index closest to design ``x``, skipping excluded ones."""
        dist = jnp.sum((self.design_space - x) ** 2, axis=tuple(range(1, self.design_space.ndim)))
        if exclude:
            if len(exclude) >= dist.shape[0]:
                raise ValueError("Every design in the pool has been observed.")
            dist = dist.at[jnp.asarray(sorted(exclude))].set(jnp.inf)
        return jnp.argmin(dist)

    def _pool_sample_search(self, criterion_func, filtr, num_samples, exclude=None, rng=None):
        """Evaluate a criterion on ``num_samples`` pool points drawn with replacement.

        Returns the winning pool index and its criterion value. Excluded indices
        are masked before the argmax; if every drawn point is excluded, fall back
        to evaluating the full non-excluded pool. Raises only when the whole pool
        is excluded.
        """
        local_rng = rng if rng is not None else self._rng
        n = self.design_space.shape[0]
        if exclude and len(exclude) >= n:
            raise ValueError("Every design in the pool has been observed.")
        idx = jax.random.randint(
            jax.random.key(0), (num_samples,), 0, n)
        # The criterion consumes one key per candidate design.
        sample_keys = self._criterion_keys(num_samples, rng=local_rng)
        values = jnp.asarray(criterion_func(
            x=self.design_space[idx], filtr=filtr, key=sample_keys,
            rng=local_rng)).reshape(-1)
        if exclude:
            allowed = ~jnp.isin(idx, jnp.asarray(sorted(exclude)))
            if not bool(jnp.any(allowed)):
                # Every drawn point was excluded: evaluate the full allowed pool.
                keep = [i for i in range(n) if i not in set(int(e) for e in exclude)]
                allowed_idx = jnp.asarray(keep)
                pool_values = jnp.asarray(criterion_func(
                    x=self.design_space[allowed_idx],
                    filtr=filtr,
                    key=self._criterion_keys(len(keep), rng=local_rng),
                    rng=local_rng,
                )).reshape(-1)
                return allowed_idx[jnp.argmax(pool_values)], pool_values.max()
            values = values.at[~allowed].set(-jnp.inf)
        best_pos = jnp.argmax(values)
        return idx[best_pos], values[best_pos]

    def _criterion_keys(self, count, rng=None):
        """Return ``count`` per-candidate random keys from ``rng`` (default ``self._rng``)."""
        local_rng = rng if rng is not None else self._rng
        seeds = local_rng.integers(0, 10000, size=(count,))
        return jax.vmap(jax.random.key)(seeds)

    def run(
        self,
        criterion_label,
        filter_type,
        filter_params,
        filter_instance,
        epochs,
        optimizer="grid_search",
        optimizer_params=None,
        trace=False,
        exclude=None,
        rng=None,
        label=None,
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
        if criterion_label not in criterion_dict:
            raise ValueError(
                f"Unknown criterion '{criterion_label}'; choose from {sorted(criterion_dict)}")
        criterion_func = criterion_dict[criterion_label]
        if optimizer_params is None:
            optimizer_params = {"lr": 1, "max_iters": 50}
        # Use a per-call generator instead of rebinding self._rng: an explicit rng
        # keeps runs reproducible and leaves the instance's default stream intact;
        # a bare call consumes (advances) the default stream rather than replaying
        # it from the seed.
        local_rng = rng if rng is not None else self._rng
        used_indices = set() if exclude is None else {int(i) for i in exclude}
        selected_indices = []
        selected_designs = []
        crit_values = []
        sd_values_test_pool = []
        rmse_values_test_pool = []
        rmse_outputs_test_pool = []
        rmse_outputs_test_glob = []
        sd_values_test_glob = []
        rmse_values_test_glob = []
        filters = []

        progress_bar = tqdm(
            range(epochs), total=epochs, desc=f"Running {criterion_label} Experiment"
        )
        filtr = filter_instance
        for i in progress_bar:
            estimate_mean, estimate_cov = filtr.state_prior
            mean_test_pool, cov_test_pool = filtr.measurement_prior(
                self.data.x_test_pool)
            mean_test_glob, cov_test_glob = filtr.measurement_prior(
                self.data.x_test_glob)
            if trace:
                # Snapshot the belief (sharing the immutable model) rather than
                # deep-copying the whole filter, which would clone the model graph
                # every epoch. Fall back to deepcopy for filters without snapshot().
                snap = getattr(filtr, "snapshot", None)
                filters.append(snap() if callable(snap) else deepcopy(filtr))
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
                exclude=used_indices,
                rng=local_rng,
            )
            measurement = self.data.observe(best_index)
            used_indices.add(int(best_index))
            selected_indices.append(int(best_index))
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
            rmse_outputs_test_pool.append(self.calculate_rmse_per_output(
                mean_test_pool, self.data.y_test_pool))
            rmse_outputs_test_glob.append(self.calculate_rmse_per_output(
                mean_test_glob, self.data.y_test_glob))
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
            crit_label=criterion_label if label is None else label,
            filter_type=filter_type,
            filters=filters,
            selected_indices=selected_indices,
            rmse_pool_outputs=rmse_outputs_test_pool,
            rmse_glob_outputs=rmse_outputs_test_glob,
        )

    def calculate_rmse(self, cov):
        """
        Mean predictive uncertainty over a batch of covariance matrices.

        Args:
            cov: Batch of predictive covariance matrices, shape ``(n, d_y, d_y)``.
                A single ``(d_y, d_y)`` matrix yields one value.

        Returns:
            The mean over the batch of ``sqrt(tr(cov))``, a scalar.
        """
        return (jnp.linalg.trace(cov) ** 0.5).mean()

    def calculate_rmse_per_output(self, predictions, true_measurements):
        """RMSE over the design points, separately for each output dimension (shape ``(d_y,)``)."""
        n = jnp.asarray(true_measurements).shape[0]
        pred = jnp.asarray(predictions).reshape(n, -1)
        true = jnp.asarray(true_measurements).reshape(n, -1)
        return jnp.sqrt(jnp.mean((pred - true) ** 2, axis=0))

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
        criteria=None,
        filter_types=None,
        filter_params=None,
        iterations=10,
        optimizer_method="brute_force",
        optimizer_params=None,
        trace=False
    ):
        """
        Compare EPIG and EIG design strategies side-by-side.

        Runs both EPIG and EIG sequential experiments with identical settings
        to enable direct comparison of the two design criteria.

        Args:
            criteria: Strategies to compare. Defaults to EPIG, EIG and RANDOM.
                EPIG-MC is supported but excluded from the default because its
                nested Monte Carlo estimate costs ``num_latent_samples`` model
                evaluations per pool point; pass it explicitly to include it.
            filter_types: Filters to evaluate, one per entry in ``filter_params``.
                Defaults to ``["ukf", "ekf"]``.
            filter_params: Constructor overrides per filter, parallel to
                ``filter_types``. Defaults to empty overrides for each filter.
            iterations: Number of sequential experiments for each method
            optimizer_params: Optimization settings (passed to run())

        Returns:
            MultiExperimentResults containing results from both methods

        Note:
            Creates a deepcopy of self for the second experiment to ensure
            independent EKF state evolution.
        """
        if criteria is None:
            criteria = ["EPIG", "EIG", "RANDOM"]
        if filter_types is None:
            filter_types = ["ukf", "ekf"]
        if filter_params is None:
            filter_params = [{} for _ in filter_types]
        if optimizer_params is None:
            optimizer_params = {"lr": 1, "max_iters": 50}
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
            # The same seed for every filter gives them the same warm-start designs.
            warm = self.run(
                criterion_label="RANDOM",
                filter_type=filter_type,
                filter_params=filter_kwargs,
                filter_instance=filter_instance,
                epochs=self.warm_start,
                optimizer=optimizer_method,
                optimizer_params=optimizer_params,
                trace=trace,
                rng=np.random.default_rng(self.seed),
                label="WARM-START",
            )
            if self.init == "empirical_bayes":
                idx = jnp.asarray(warm.selected_indices, dtype=int)
                eb = empirical_bayes_init(
                    self.model, self.data.x_train[idx], self.data.y_train[idx],
                    prior_mean=self.state_init_prior[0], prior_cov=self.state_init_prior[1],
                    z_init=filter_instance.state_prior[0],
                    noise_cov=self.measurement_error, **self.empirical_bayes_kwargs,
                )
                # The warm-start observations are used once, inside the Laplace belief; the
                # filter starts from it with the fitted observation covariance.
                filter_instance = fclass(
                    model=self.model,
                    state_prev=eb.mean,
                    state_cov_prev=eb.cov,
                    state_innovation=self.latent_innovation,
                    measurement_error=eb.noise_cov,
                    **filter_params[i]
                )
                warm.init_info = eb
            results.append(warm)
            for j, criterion in enumerate(criteria):
                # Each criterion starts from a copy of the same warm-start posterior
                # and may not re-select the warm-start designs.
                filter_instance_copy = deepcopy(filter_instance)
                r = self.run(
                    criterion_label=criterion,
                    filter_type=filter_type,
                    filter_params=filter_kwargs,
                    filter_instance=filter_instance_copy,
                    epochs=iterations,
                    optimizer=optimizer_method,
                    optimizer_params=optimizer_params,
                    trace=trace,
                    exclude=warm.selected_indices,
                    rng=np.random.default_rng(self.seed + 1),
                )
                results.append(r)

        return MultiExperimentResults(results)


class ExperimentResults:
    """
    Container for results from a sequential experimental design run.

    Stores all metrics tracked during an experiment including RMSE values,
    selected designs, criterion values, and optimization gradients.

    Attributes:
        sd_glob: List of predictive standard deviations on the global test set per epoch
        sd_pool: List of predictive standard deviations on the test pool per epoch
        rmse_glob: List of prediction RMSEs on the global test set per epoch
        rmse_pool: List of prediction RMSEs on the test pool per epoch
        selected_designs: List of ``(design, measurement)`` tuples per epoch
        crit_values: List of criterion values for the selected designs per epoch
        crit_label: Name of the criterion used (e.g. "EPIG" or "EIG")
        filter_type: State estimator used ("ekf" or "ukf")
        design_space: The pool of candidate designs (may be None)
        filters: List of per-epoch filter snapshots when ``trace=True``
        data: Dataset container the run operated on
        selected_indices: List of pool indices selected across epochs
        rmse_pool_outputs: Per-output RMSE on the test pool per epoch
        rmse_glob_outputs: Per-output RMSE on the global test set per epoch
        init_info: Empirical Bayes result when the warm start was followed by
            empirical Bayes initialisation (otherwise None)
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
        selected_indices=None,
        rmse_pool_outputs=None,
        rmse_glob_outputs=None,
    ):
        """Initialize metric histories and selected designs.

        Args:
            sd_glob: Predictive standard deviations on the global test set per epoch.
            sd_pool: Predictive standard deviations on the test pool per epoch.
            rmse_global: Prediction RMSEs on the global test set per epoch.
            rmse_pool: Prediction RMSEs on the test pool per epoch.
            selected_designs: ``(design, measurement)`` tuples per epoch.
            crit_values: Criterion values for the selected designs per epoch.
            data: Dataset container the run operated on.
            crit_label: Name of the criterion used (default "EPIG").
            filter_type: State estimator used (default "ekf").
            design_space: The pool of candidate designs (default None).
            filters: Per-epoch filter snapshots when ``trace=True`` (default []).
            selected_indices: Pool indices selected across epochs (default None).
            rmse_pool_outputs: Per-output RMSE on the test pool (default None).
            rmse_glob_outputs: Per-output RMSE on the global test set (default None).
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
        self.selected_indices = [] if selected_indices is None else list(
            selected_indices)
        self.rmse_pool_outputs = [] if rmse_pool_outputs is None else list(
            rmse_pool_outputs)
        self.rmse_glob_outputs = [] if rmse_glob_outputs is None else list(
            rmse_glob_outputs)
        # EmpiricalBayesResult when the warm start was followed by empirical Bayes
        self.init_info = None


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
