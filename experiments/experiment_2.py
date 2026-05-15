from bed.data import create_synthetic_data
from bed.models import DenseNN, NeuralNetworkRegressor
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx


jax.config.update("jax_enable_x64", True)
hidden_dim_0 = 16
hidden_dim_1 = 16
design_dim = 20
latent_dim = (
    design_dim * hidden_dim_0
    + hidden_dim_0
    + hidden_dim_0 * hidden_dim_1
    + hidden_dim_1
    + hidden_dim_1 * 1
    + 1
)
latent_true = 2 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 100
latent_innovation = 0
measurement_cov = 100 * jnp.eye(1)
# epochs = int(10 * latent_dim)
epochs = 100
# design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
training_kwargs = {"learning_rate": 0.05, "epochs": 300, "rngs": nnx.Rngs(0)}
random_key = jax.random.PRNGKey(0)
model = DenseNN(
    input_dim=design_dim,
    hidden_dim_0=hidden_dim_0,
    hidden_dim_1=hidden_dim_1,
    rngs=nnx.Rngs(6),
)
model_true = DenseNN(
    input_dim=design_dim,
    hidden_dim_0=hidden_dim_0,
    hidden_dim_1=hidden_dim_1,
    rngs=nnx.Rngs(0),
)
state_true = model_true.weights_to_state(latent_true)
nnx.update(model_true, state_true)
plot_results = False
num_train = 499
num_test = 1
data = create_synthetic_data(
    model_true,
    num_train,
    num_test,
    design_dim,
    embedding_dim=10,
    embedding_noise_std=0.10,
    measurement_noise_std=0 * jnp.sqrt(measurement_cov[0, 0]),
    # measurement_noise_std=0,
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
results = experiment.run_experiment(
    experiments=[
        "EPIG",
        "EIG",
        "MC",
        "RAND",
    ],
    iterations=epochs,
    # optimizer_method="grid_search",
    optimizer_method="grid_search",
    optimizer_params={
        "lr": 10,
        "max_iters": 10,
        "num_samples": 300,
    },
)
results.plot_comparison()
plt.show()
