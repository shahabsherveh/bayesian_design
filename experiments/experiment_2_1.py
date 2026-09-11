from bed.data import create_synthetic_data, create_synthetic_normal_mixture_data_1D
from bed.models import DenseNN, LinearNN, NeuralNetworkRegressor, Sinus
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx


jax.config.update("jax_enable_x64", True)
hidden_dim_0 = 16
hidden_dim_1 = 16
hidden_dim_2 = 16
design_dim = 1
latent_var = 2
latent_innovation = 0
measurement_cov = 0.0001 * jnp.eye(1)
# epochs = int(10 * latent_dim)
# design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
training_kwargs = {"learning_rate": 0.01, "epochs": 30, "rngs": nnx.Rngs(0)}
random_key = jax.random.PRNGKey(0)
model = DenseNN(
    input_dim=design_dim,
    hidden_dims=[hidden_dim_0, hidden_dim_1, hidden_dim_2],
    output_dim=1,
    rngs=nnx.Rngs(6),
)
latent_dim = design_dim + 1
latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
model_true = Sinus(
    input_dim=design_dim,
    freq=0.5,
    amp=3,
)
# state_true = model_true.weights_to_state(latent_true)
# nnx.update(model_true, state_true)
plot_results = True
num_train = 20
num_test = 10
num_val = 0
epochs = 6
data = create_synthetic_normal_mixture_data_1D(
    model_true,
    jnp.array(
        [
            0.16,
        ]
    ),
    jnp.array(
        [
            0.5,
        ]
    ),
    num_train,
    num_test,
    num_val=num_val,
    measurement_noise_std=0,
    # extra_points=jnp.array([-1.5, 2.5]),
    extra_points=jnp.array([]),
    key=jax.random.PRNGKey(1234),
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
        # "EPIG-MC",
        # "RAND",
    ],
    iterations=epochs,
    # optimizer_method="grid_search",
    optimizer_method="brute_force",
    optimizer_params={
        "lr": 10,
        "max_iters": 10,
        "num_samples": 1000,
    },
)
results.plot_comparison()
plt.show()
