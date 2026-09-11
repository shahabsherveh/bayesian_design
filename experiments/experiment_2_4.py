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
latent_innovation = 0.05
measurement_cov = 0.1 * jnp.eye(1)
# epochs = int(10 * latent_dim)
# design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
training_kwargs = {"learning_rate": 0.01, "epochs": 100, "rngs": nnx.Rngs(0)}
random_key = jax.random.PRNGKey(0)
model = DenseNN(
    input_dim=design_dim,
    hidden_dims=[hidden_dim_0, hidden_dim_1, hidden_dim_2],
    output_dim=1,
    rngs=nnx.Rngs(9),
)
latent_dim = design_dim + 1
latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
model_true = Sinus(
    input_dim=design_dim,
    freq=2,
    amp=3,
)
# state_true = model_true.weights_to_state(latent_true)
# nnx.update(model_true, state_true)
plot_results = True
num_train = 20
num_test = 80
epochs = 51
data = create_synthetic_normal_mixture_data_1D(
    model_true,
    jnp.array(
        [
            0.13,
        ]
    ),
    jnp.array(
        [
            0,
        ]
    ),
    num_train,
    num_test,
    measurement_noise_std=measurement_cov[0, 0] ** 0.5,
    # extra_points=jnp.array([-1.25]),
    extra_points=jnp.array([]),
    key=jax.random.key(9),
)

model_state = nnx.state(model)
mean = model.state_to_weights(model_state)
latent_dim = mean.shape[0]
latent_bias_var = 2
latent_kernel_var = 2.0
latent_cov = jnp.zeros((latent_dim,))
for layer, layer_meta in model.weight_mapping.items():
    s = layer_meta["slice"]
    shape = layer_meta["shape"]
    var = latent_kernel_var / shape[0]
    fan_in = shape[0]
    fan_out = shape[1] if len(shape) > 1 else 1
    fan_avg = (fan_in + fan_out) / 2
    var = 2 / fan_in
    var = latent_bias_var
    # if layer[1] == "bias":
    #     var = latent_bias_var
    latent_cov = latent_cov.at[s[0] : s[1]].set(var)
latent_cov = jnp.diag(latent_cov)


experiment = Experiment(
    model=NeuralNetworkRegressor(model),
    data=data,
    latent_cov=latent_cov,
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
results.plot_design_distribution()
plt.show()
