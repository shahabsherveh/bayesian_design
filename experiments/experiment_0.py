from bed.data import create_synthetic_data
from bed.models import LinearNN, NeuralNetworkRegressor
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from copy import deepcopy


jax.config.update("jax_enable_x64", True)
design_dim = 2
latent_dim = design_dim + 1
latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 1
latent_innovation = 0
measurement_cov = 0.16 * jnp.eye(1)
# epochs = int(10 * latent_dim)
epochs = 5
design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
random_key = jax.random.PRNGKey(0)
model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
model_true = deepcopy(model)
state_true = model_true.weights_to_state(latent_true)
nnx.update(model_true, state_true)
plot_results = True
num_train = 20
num_test = 80
training_kwargs = {"learning_rate": 0.01, "epochs": 50, "rngs": nnx.Rngs(0)}
data = create_synthetic_data(
    model_true,
    num_train,
    num_test,
    design_dim,
    embedding_dim=1,
    embedding_noise_std=0.001,
    measurement_noise_std=jnp.sqrt(measurement_cov[0, 0]),
)
model_state = nnx.state(model)
mean = model.state_to_weights(model_state)
latent_dim = mean.shape[0]
latent_bias_var = 1
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
    if layer[1] == "bias":
        var = latent_bias_var
    latent_cov = latent_cov.at[s[0] : s[1]].set(var)
latent_cov = jnp.diag(latent_cov)


experiment = Experiment(
    model=NeuralNetworkRegressor(model),
    data=data,
    latent_cov=latent_cov,
    latent_innovation=latent_innovation,
    measurement_error=measurement_cov,
    plot_inter_results=True,
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
    optimizer_method="grid_search",
    optimizer_params={"lr": 2, "max_iters": 100, "num_samples": 200},
)
results.plot_comparison()
plt.show()
