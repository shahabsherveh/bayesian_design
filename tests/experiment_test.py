import jax
import jax.numpy as jnp
from bed.experiments import Experiment1
from scipy import stats
from .utils import linear_model_epig
from matplotlib import pyplot as plt
from tqdm import trange


class TestExperimentalEKF:
    jax.config.update("jax_enable_x64", True)
    latent_dim = 20
    latent_true = 1 * jax.random.normal(jax.random.PRNGKey(0), (latent_dim, 1)) + 0
    latent_var = 0.1
    latent_innovation = 0
    measurement_cov = 1 * jnp.eye(1)
    design_pool_num = 100
    design_mean = jnp.zeros(latent_dim)
    design_mean = design_mean.at[4:].set(10)
    random_key = jax.random.PRNGKey(0)
    eigs = jnp.full(latent_dim, fill_value=0.01, dtype="float64")
    eigs = eigs.at[:10].set(10)
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
    epochs = 100
    eigs = latent_dim * eigs / eigs.sum()
    design_cov = 10 * stats.random_correlation.rvs(
        eigs=eigs,
        random_state=1,
        tol=1e-6,
        diag_tol=1e-6,
    )
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    plot_results = False

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
        epig_mc_list = []
        pbar = trange(100, 10100, 100, desc="EPIG MC Samples", leave=True)
        epig = experiment.calculate_epig(x_0)
        for i in pbar:
            epig_mc_inner = []
            pbar_inner = trange(100, desc="EPIG MC Repeats", leave=False)
            for _ in trange(100, desc="EPIG MC Repeats", leave=False):
                epig_mc = experiment.calculate_epig_mc(
                    x_0, num_latent_samples=i, num_design_samples=100
                )
                pbar_inner.set_description(f"EPIG MC: {epig_mc:.4f} EPIG: {epig:.4f}")
                epig_mc_inner.append(epig_mc)
            epig_mc_list.append(epig_mc_inner)
        epig_samples = jnp.array(epig_mc_list)
        q = [0.0, 0.25, 0.5, 0.75, 1.0]
        for i, q_i in enumerate(q):
            quantile = jnp.quantile(epig_samples, q=q_i, axis=1)
            plt.plot(range(100, 10100, 100), quantile, label=f"Quantile {q_i}")
        plt.axhline(epig, color="red", linestyle="--", label="True EPIG")
        plt.xlabel("K")
        plt.ylabel("EPIG Estimate")
        plt.title("EPIG Monte Carlo Estimates vs True EPIG")
        plt.legend()
        plt.show()

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
