import jax.numpy as jnp
import jax

import matplotlib.pyplot as plt
from bed.models import EKF, LinearModel, Experiment1, ExperimentResults
from .utils import linear_model_epig
import scipy.stats as stats


class TestLinearModel:
    def test_jacobian(self):
        X = jnp.array([[1.0, 0.5], [0.0, 1.0]])
        z = jnp.array([[0.5], [1.0]])
        model = LinearModel()
        assert model.jacobian(z, X)


class TestEKF:
    def test_pos(self):
        # Define a simple linear model
        model = LinearModel()
        state = jnp.array([[0.0], [0.0]])  # Initial state
        state_cov = jnp.eye(2)  # Initial covariance
        state_innovation_cov = jnp.zeros((2, 2))
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[1.0]])
        X = jnp.array([[1.0, 0.5]])
        ekf = EKF(
            model,
            state,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )
        posterior = ekf.get_state_posterior(measurement=measurement, x=X)
        cov_prior = state_cov
        mean_pior = state
        measurement_precision = 1.0 / measurement_cov[0, 0]
        S = jnp.linalg.inv(cov_prior) + measurement_precision * X.T @ X
        cov_pos = jnp.linalg.inv(S)
        mean_pos = cov_pos @ (
            jnp.linalg.inv(cov_prior) @ mean_pior
            + measurement_precision * X.T * measurement[0, 0]
        )

        assert jnp.allclose(posterior[0], mean_pos)
        assert jnp.allclose(posterior[1], cov_pos)

    def test_pred_pos_estimate(self):
        # Define a simple linear model
        model = LinearModel()
        state = jnp.array([[0.0], [0.0]])  # Initial state
        state_cov = jnp.eye(2)  # Initial covariance
        state_innovation_cov = jnp.zeros((2, 2))
        measurement_cov = jnp.eye(1)  # Measurement covariance
        measurement = jnp.array([[10.0]])
        X = jnp.array([[1.0, 0.5]])
        ekf = EKF(
            model,
            state,
            state_cov,
            state_innovation_cov,
            measurement_cov,
        )
        posterior = ekf.measurement_posterior(
            measurement=measurement, x_obs=X, x_pred=X + 1
        )

        cov_est = ekf.measurement_posterior_cov_estimate(
            X + 1,
            X,
        )
        assert jnp.allclose(posterior[1], cov_est)


class TestExperimentalEKF:
    jax.config.update("jax_enable_x64", True)
    latent_dim = 2
    latent_true = 1 * jax.random.normal(jax.random.PRNGKey(0), (latent_dim, 1))
    latent_var = 0.01
    latent_innovation = 0
    measurement_cov = 1 * jnp.eye(1)
    design_pool_num = 100
    design_mean = jnp.array([0 * (-1) ** (i // 4) for i in range(latent_dim)])
    random_key = jax.random.PRNGKey(0)
    eigs = jnp.full(latent_dim, fill_value=0.0000001, dtype="float64")
    eigs = eigs.at[:1].set(1)
    # eigs = 1 / (0.1 * jax.random.laplace(key=random_key, shape=(latent_dim,)))
    # eigs = jnp.abs(eigs)
    gd_params = {
        "learning_rate": 1,
        "max_iters": 0,
        "tol": 1e-3,
        "stochastic": False,
        "x_init_type": "best_pool",
    }
    # epochs = int(10 * latent_dim)
    epochs = 3
    eigs = latent_dim * eigs / eigs.sum()
    design_cov = stats.random_correlation.rvs(
        eigs=eigs,
        random_state=1,
        tol=1e-6,
        diag_tol=1e-6,
    )
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    plot_results = True

    def test_epig(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=0,
        )
        # x_0 = jnp.ones_like(experiment.design_space[[0]].T)
        # x_1 = jnp.ones_like(experiment.design_space[[0]].T)
        x_0 = jnp.array([[1.0], [0]])
        x_1 = jnp.array([[0], [1.0]])
        epig = experiment.calculate_epig(x_1, x_0)
        linear_epig = linear_model_epig(
            x_1,
            x_0,
            experiment.state_init_prior[1],
            experiment.measurement_error,
        )
        assert jnp.isclose(epig, linear_epig)

    def test_epig_monte_carlo(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
        )
        x_0 = experiment.design_space[[0]].T
        x_1 = experiment.design_space[[1]].T
        epig = experiment.calculate_epig_monte_carlo(x_1, x_0)

    def test_eig(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
        )
        x = experiment.design_space[[0]].T
        eig = experiment.calculate_eig(x)

    def test_optimization_epig(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
        )
        # x, epig, grads = experiment.optimize_epig_gd(learning_rate=0.5, max_iters=1000)
        results = experiment.run(
            criterion_label="EPIG",
            epochs=30,
            optimizer_params={"learning_rate": 1, "max_iters": 100, "x_init": None},
        )

    def test_optimization_eig(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
        )
        x, eig = experiment.optimize_eig_gd(learning_rate=1, max_iters=10)

    def test_run_epig(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
            plot_results=self.plot_results,
        )
        experiment_results = experiment.run_epig(
            iterations=50,
            gd_params={"learning_rate": 10, "max_iters": 30},
        )
        experiment_results.plot_results()
        plt.show()

    def test_run_eig(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
            plot_results=self.plot_results,
        )
        experiment_results = experiment.run_eig(
            x_init=None,
            iterations=50,
            gd_params={
                "learning_rate": 1,
                "max_iters": 30,
            },
        )
        experiment_results.plot_results()
        plt.show()

    def test_run(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
            plot_results=self.plot_results,
        )
        results = experiment.run_experiment(
            iterations=self.epochs,
            optimizer_params=self.gd_params,
        )
        results.plot_comparison()
        plt.show()

    def test_plot_crit_surface(self):
        experiment = Experiment1(
            latent_true=self.latent_true,
            latent_dim=self.latent_dim,
            latent_var=self.latent_var,
            latent_innovation=self.latent_innovation,
            design_pool_num=self.design_pool_num,
            design_mean=self.design_mean,
            design_cov=self.design_cov,
            measurement_error=self.measurement_cov,
        )
        experiment.plot_crit_surface()
