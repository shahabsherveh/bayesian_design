from tqdm import trange
from bed.data import create_synthetic_data
from bed.models import LinearNN, NeuralNetworkRegressor
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from copy import deepcopy
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

jax.config.update("jax_enable_x64", True)
design_dim = 10
latent_dim = design_dim + 1
latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 0.1
latent_innovation = 0
measurement_cov = 0.01 * jnp.eye(1)
# epochs = int(10 * latent_dim)
epochs = 50
# design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
random_key = jax.random.PRNGKey(0)
model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
model_true = deepcopy(model)
state_true = model_true.weights_to_state(latent_true)
nnx.update(model_true, state_true)
plot_results = False
num_train = 50
num_test = 400
training_kwargs = {"learning_rate": 0.01, "epochs": 100, "rngs": nnx.Rngs(0)}
data = create_synthetic_data(
    model_true,
    num_train,
    num_test,
    design_dim,
    embedding_dim=5,
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
fig, ax = plt.subplots(figsize=(13, 6))
for i, q_i in enumerate(q):
    quantile = jnp.quantile(epig_samples, q=q_i, axis=1)
    ax.plot(range(100, 10100, 100), quantile, label=f"Quantile {q_i}")
ax.axhline(epig, color="red", linestyle="--", label="True EPIG")
ax.set_xlabel("Number of Latent Samples")
plt.ylabel("EPIG Estimate")
plt.title("EPIG Monte Carlo Estimates vs True EPIG")
plt.legend()
plt.show()
