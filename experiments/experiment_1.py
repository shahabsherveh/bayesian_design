from bed.data import create_synthetic_data
from bed.models import LinearNN, NeuralNetworkRegressor
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from copy import deepcopy


jax.config.update("jax_enable_x64", True)
design_dim = 400
latent_dim = design_dim + 1
latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 0.1
latent_innovation = 0
measurement_cov = 1 * jnp.eye(1)
# epochs = int(10 * latent_dim)
epochs = 100
# design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
random_key = jax.random.PRNGKey(0)
model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
model_true = deepcopy(model)
state_true = model_true.weights_to_state(latent_true)
nnx.update(model_true, state_true)
plot_results = True
num_train = 90
num_test = 10
training_kwargs = {"learning_rate": 0.01, "epochs": 50, "rngs": nnx.Rngs(0)}
data = create_synthetic_data(
    model_true,
    num_train,
    num_test,
    design_dim,
    embedding_dim=50,
    embedding_noise_std=0.01,
    measurement_noise_std=jnp.sqrt(measurement_cov[0, 0]),
    var=2,
)
experiment = Experiment(
    model=NeuralNetworkRegressor(model),
    data=data,
    latent_cov=latent_var * jnp.eye(latent_dim),
    latent_innovation=latent_innovation,
    measurement_error=measurement_cov,
    plot_inter_results=plot_results,
    pre_train_model=False,
    training_kwargs=training_kwargs,
)
results = experiment.run_experiment(
    experiments=[
        "EPIG",
        "EIG",
        "EPIG-MC",
        "RAND",
    ],
    iterations=epochs,
    optimizer_method="brute_force",
    # optimizer_method="grid_search",
    optimizer_params={"num_samples": 400},
)
results.plot_comparison()
plt.show()
