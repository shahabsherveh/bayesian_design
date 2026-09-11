from copy import deepcopy
from flax import nnx
import jax
import jax.numpy as jnp
from bed.data import create_synthetic_data, get_mnist_data
from bed.experiments import Experiment
from scipy import stats

from bed.models import (
    CNN,
    LinearNN,
    NeuralNetworkClassifier,
    NeuralNetworkRegressor,
    DenseNN,
)
from .utils import linear_model_epig
from matplotlib import pyplot as plt
from tqdm import trange


class TestExperiment0:
    jax.config.update("jax_enable_x64", True)
    design_dim = 2
    latent_dim = design_dim + 1
    latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
    latent_var = 0.1
    latent_innovation = 0
    measurement_cov = 0.25 * jnp.eye(1)
    # epochs = int(10 * latent_dim)
    epochs = 5
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    random_key = jax.random.PRNGKey(0)
    model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
    model_true = deepcopy(model)
    state_true = model_true.weights_to_state(latent_true)
    nnx.update(model_true, state_true)
    plot_results = True
    num_train = 20
    num_test = 80
    training_kwargs = {"learning_rate": 0.01, "epochs": 500, "rngs": nnx.Rngs(0)}
    data = create_synthetic_data(
        model_true,
        num_train,
        num_test,
        design_dim,
        embedding_dim=1,
        embedding_noise_std=0.01,
        measurement_noise_std=jnp.sqrt(measurement_cov[0, 0]),
    )
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=plot_results,
        pre_train_model=False,
        training_kwargs=training_kwargs,
    )

    def test_epig(self):
        experiment = self.experiment
        x_0 = jnp.array([[1.0], [0], [0], [0], [0]])
        x_1 = jnp.array([[0], [1.0], [0], [0], [0]])
        epig = experiment.calculate_epig(x_1, x_0)

    def test_epig_monte_carlo(self):
        experiment = self.experiment
        x_0 = experiment.design_space[jnp.array([0])]
        epig_mc_list = []
        pbar = trange(100, 10100, 100, desc="EPIG MC Samples", leave=True)
        epig = experiment.calculate_epig(x_0)
        try:
            for i in pbar:
                epig_mc_inner = []
                pbar_inner = trange(100, desc="EPIG MC Repeats", leave=False)
                for _ in trange(100, desc="EPIG MC Repeats", leave=False):
                    epig_mc = experiment.calculate_epig_mc(
                        x_0, num_latent_samples=i, num_design_samples=100
                    )
                    pbar_inner.set_description(
                        f"EPIG MC: {epig_mc.squeeze():.4f} EPIG: {epig.squeeze():.4f}"
                    )
                    epig_mc_inner.append(epig_mc)
                epig_mc_list.append(epig_mc_inner)
        except KeyboardInterrupt:
            pass
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
        experiment = self.experiment
        x = experiment.design_space[[0]].T
        eig = experiment.calculate_eig(x)

    def test_optimization_epig(self):
        experiment = self.experiment
        # x, epig, grads = experiment.optimize_epig_gd(learning_rate=0.5, max_iters=1000)
        results = experiment.run(
            criterion_label="EPIG",
            epochs=10,
        )
        plt.show()

    def test_run(self):
        experiment = self.experiment
        results = experiment.run_experiment(
            experiments=[
                "EPIG",
                "EIG",
                "MC",
                "RAND",
            ],
            iterations=self.epochs,
            optimizer_method="gradient_ascent",
            optimizer_params={"lr": 2, "max_iters": 100},
        )
        results.plot_comparison()
        plt.show()


class TestExperiment1:
    jax.config.update("jax_enable_x64", True)
    design_dim = 50
    latent_dim = design_dim + 1
    latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
    latent_var = 0.1
    latent_innovation = 0
    measurement_cov = 0.25 * jnp.eye(1)
    # epochs = int(10 * latent_dim)
    epochs = 50
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    random_key = jax.random.PRNGKey(0)
    model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
    model_true = deepcopy(model)
    state_true = model_true.weights_to_state(latent_true)
    nnx.update(model_true, state_true)
    plot_results = False
    num_train = 100
    num_test = 200
    training_kwargs = {"learning_rate": 0.01, "epochs": 50, "rngs": nnx.Rngs(0)}
    data = create_synthetic_data(
        model_true,
        num_train,
        num_test,
        design_dim,
        embedding_dim=20,
        embedding_noise_std=0.0001,
        measurement_noise_std=jnp.sqrt(measurement_cov[0, 0]),
    )
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=plot_results,
        pre_train_model=False,
        training_kwargs=training_kwargs,
    )

    def test_epig(self):
        experiment = self.experiment
        x_0 = jnp.array([[1.0], [0], [0], [0], [0]])
        x_1 = jnp.array([[0], [1.0], [0], [0], [0]])
        epig = experiment.calculate_epig(x_1, x_0)

    def test_epig_monte_carlo(self):
        experiment = self.experiment
        x_0 = experiment.design_space[jnp.array([0])]
        epig_mc_list = []
        pbar = trange(100, 10100, 100, desc="EPIG MC Samples", leave=True)
        epig = experiment.calculate_epig(x_0)
        try:
            for i in pbar:
                epig_mc_inner = []
                pbar_inner = trange(100, desc="EPIG MC Repeats", leave=False)
                for _ in trange(100, desc="EPIG MC Repeats", leave=False):
                    epig_mc = experiment.calculate_epig_mc(
                        x_0, num_latent_samples=i, num_design_samples=100
                    )
                    pbar_inner.set_description(
                        f"EPIG MC: {epig_mc.squeeze():.4f} EPIG: {epig.squeeze():.4f}"
                    )
                    epig_mc_inner.append(epig_mc)
                epig_mc_list.append(epig_mc_inner)
        except KeyboardInterrupt:
            pass
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
        experiment = self.experiment
        x = experiment.design_space[[0]].T
        eig = experiment.calculate_eig(x)

    def test_optimization_epig(self):
        experiment = self.experiment
        # x, epig, grads = experiment.optimize_epig_gd(learning_rate=0.5, max_iters=1000)
        results = experiment.run(
            criterion_label="EPIG",
            epochs=10,
        )
        plt.show()

    def test_run(self):
        experiment = self.experiment
        results = experiment.run_experiment(
            experiments=[
                "EPIG",
                "EIG",
                "MC",
                "RAND",
            ],
            iterations=self.epochs,
        )
        results.plot_comparison()
        plt.show()


class TestExperiment2:
    jax.config.update("jax_enable_x64", True)
    hidden_dim_0 = 16
    hidden_dim_1 = 16
    design_dim = 10
    latent_dim = (
        design_dim * hidden_dim_0
        + hidden_dim_0
        + hidden_dim_0 * hidden_dim_1
        + hidden_dim_1
        + hidden_dim_1 * 1
        + 1
    )
    latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
    latent_var = 0.05
    latent_innovation = 0
    measurement_cov = 100 * jnp.eye(1)
    # epochs = int(10 * latent_dim)
    epochs = 100
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    training_kwargs = {"learning_rate": 0.01, "epochs": 50, "rngs": nnx.Rngs(0)}
    random_key = jax.random.PRNGKey(0)
    model = DenseNN(
        input_dim=design_dim,
        hidden_dims=[hidden_dim_0, hidden_dim_1],
        output_dim=1,
        rngs=nnx.Rngs(6),
    )
    model_true = DenseNN(
        input_dim=design_dim,
        hidden_dims=[hidden_dim_0, hidden_dim_1],
        output_dim=1,
        rngs=nnx.Rngs(0),
    )
    state_true = model_true.weights_to_state(latent_true)
    nnx.update(model_true, state_true)
    plot_results = False
    num_train = 20
    num_test = 150
    data = create_synthetic_data(
        model_true,
        num_train,
        num_test,
        design_dim,
        embedding_dim=5,
        embedding_noise_std=0.010,
        measurement_noise_std=jnp.sqrt(measurement_cov[0, 0]),
    )
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=plot_results,
        pre_train_model=True,
        training_kwargs=training_kwargs,
    )

    def test_epig(self):
        experiment = self.experiment
        x_0 = jnp.array([[1.0], [0], [0], [0], [0]])
        x_1 = jnp.array([[0], [1.0], [0], [0], [0]])
        epig = experiment.calculate_epig(x_1, x_0)

    def test_epig_monte_carlo(self):
        experiment = self.experiment
        x_0 = experiment.design_space[jnp.array([0])]
        epig_mc_list = []
        pbar = trange(500, 10500, 500, desc="EPIG MC Samples", leave=True)
        epig = experiment.calculate_epig(x_0)
        try:
            for i in pbar:
                epig_mc_inner = []
                pbar_inner = trange(100, desc="EPIG MC Repeats", leave=False)
                for _ in trange(100, desc="EPIG MC Repeats", leave=False):
                    epig_mc = experiment.calculate_epig_mc(
                        x_0, num_latent_samples=i, num_design_samples=100
                    )
                    pbar_inner.set_description(
                        f"EPIG MC: {epig_mc.squeeze():.4f} EPIG: {epig.squeeze():.4f}"
                    )
                    epig_mc_inner.append(epig_mc)
                epig_mc_list.append(epig_mc_inner)
        except KeyboardInterrupt:
            pass
        epig_samples = jnp.array(epig_mc_list)
        q = [0.25, 0.5, 0.75]
        breakpoint()
        for i, q_i in enumerate(q):
            quantile = jnp.quantile(epig_samples, q=q_i, axis=1)
            plt.plot(range(500, 10500, 500), quantile, label=f"Quantile {q_i}")
        plt.axhline(epig, color="red", linestyle="--", label="True EPIG")
        plt.xlabel("K")
        plt.ylabel("EPIG Estimate")
        plt.title("EPIG Monte Carlo Estimates vs True EPIG")
        plt.legend()
        plt.show()

    def test_eig(self):
        experiment = self.experiment
        x = experiment.design_space[[0]].T
        eig = experiment.calculate_eig(x)

    def test_optimization_epig(self):
        experiment = self.experiment
        # x, epig, grads = experiment.optimize_epig_gd(learning_rate=0.5, max_iters=1000)
        results = experiment.run(
            criterion_label="EPIG",
            epochs=10,
        )
        plt.show()

    def test_run(self):
        experiment = self.experiment
        results = experiment.run_experiment(
            experiments=[
                "EPIG",
                "EIG",
                "MC",
                "RAND",
            ],
            iterations=self.epochs,
        )
        results.plot_comparison()
        plt.show()


class TestExperiment3:
    jax.config.update("jax_enable_x64", True)
    latent_dim = 24 + 20 + 5
    latent_true = 2 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
    design_dim = 5
    latent_var = 0.1
    latent_innovation = 0
    measurement_cov = 0.25 * jnp.eye(10)
    # epochs = int(10 * latent_dim)
    epochs = 100
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    random_key = jax.random.PRNGKey(0)
    model = CNN(rngs=nnx.Rngs(0))
    plot_results = False
    num_train = 20
    num_test = 80
    data = get_mnist_data(num_train=num_train, num_test=num_test)
    experiment = Experiment(
        model=NeuralNetworkClassifier(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=plot_results,
        pre_train_model=True,
    )

    def test_epig(self):
        experiment = self.experiment
        x_0 = jnp.array([[1.0], [0], [0], [0], [0]])
        x_1 = jnp.array([[0], [1.0], [0], [0], [0]])
        epig = experiment.calculate_epig(x_1, x_0)

    def test_epig_monte_carlo(self):
        experiment = self.experiment
        x_0 = experiment.design_space[0]
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
                pbar_inner.set_description(
                    f"EPIG MC: {epig_mc:.4f} EPIG: {epig.squeeze():.4f}"
                )
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
        experiment = self.experiment
        x = experiment.design_space[[0]].T
        eig = experiment.calculate_eig(x)

    def test_optimization_epig(self):
        experiment = self.experiment
        # x, epig, grads = experiment.optimize_epig_gd(learning_rate=0.5, max_iters=1000)
        results = experiment.run(
            criterion_label="EPIG",
            epochs=10,
        )
        plt.show()

    def test_run(self):
        experiment = self.experiment
        results = experiment.run_experiment(
            experiments=[
                "EPIG",
                "EIG",
                # "MC",
                "RAND",
            ],
            iterations=self.epochs,
        )
        results.plot_comparison()
        plt.show()
