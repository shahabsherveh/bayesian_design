#!.venv/bin/python
from tqdm import trange
from bed.data import create_synthetic_data
from bed.ekf import EKF
from bed.models import LinearNN, NeuralNetworkRegressor
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from copy import deepcopy
import matplotlib.pyplot as plt

from bed.ukf import UKF

plt.rcParams.update({"font.size": 11})

jax.config.update("jax_enable_x64", True)
design_dim = 10
latent_dim = design_dim + 1
latent_true = 1 * \
    jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 0.1 * jnp.eye(latent_dim)
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
num_test = 1000
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
    latent_cov=latent_var,
    latent_innovation=latent_innovation,
    measurement_error=measurement_cov,
    plot_inter_results=plot_results,
    pre_train_model=False,
    training_kwargs=training_kwargs,
)
x_0 = experiment.design_space[jnp.array([0])]
epig_mc_list = []
pbar = trange(100, 10100, 100, desc="EPIG MC Samples", leave=True)
ekf = UKF(
    model=experiment.model,
    state_prev=experiment.state_init_prior[0],
    state_cov_prev=experiment.state_init_prior[1],
    state_innovation=experiment.measurement_error,
    measurement_error=experiment.measurement_error
)

epig = experiment.calculate_epig(x_0, filtr=ekf)
try:
    for i in pbar:
        epig_mc_inner = []
        pbar_inner = trange(30, desc="EPIG MC Repeats", leave=False)
        for _ in pbar_inner:
            epig_mc = experiment.calculate_epig_mc(
                x_0, filtr=ekf, num_latent_samples=i, num_design_samples=500
            )
            pbar_inner.set_description(
                f"EPIG MC: {epig_mc.squeeze():.4f} EPIG: {epig.squeeze():.4f}"
            )
            epig_mc_inner.append(epig_mc)
        pbar.set_description(
            f"EPIG MC: {jnp.mean(jnp.array(epig_mc_inner)):.4f} EPIG: {epig.squeeze():.4f}"
        )
        epig_mc_list.append(epig_mc_inner)
except KeyboardInterrupt:
    pass
epig_samples = jnp.array(epig_mc_list)
q = jnp.array([0.01, 0.5, 0.99])
fig, ax = plt.subplots(figsize=(90 / 25.4, 100 / 25.4))
fig.subplots_adjust(bottom=0.15)
quantile = jnp.quantile(epig_samples, q=q, axis=1)
# ax.plot(range(100, 10100, 100), quantile[1], color="black", label="median")
ax.plot(
    range(100, 10100, 100),
    epig_samples.mean(axis=1),
    color="tab:green",
    label="mean",
    # linestyle="--",
)
ax.fill_between(
    range(100, 10100, 100),
    quantile[0].squeeze(),
    quantile[2].squeeze(),
    color="tab:green",
    alpha=0.3,
    label="25th-75th Percentile",
)
ax.axhline(epig, color="tab:blue", linestyle="-", label="EPIG")
ax.set_xlabel("Number of Latent Samples")
plt.ylabel("EPIG Estimate")
ax.set_ylim(-0.5, None)
# plt.title("EPIG Monte Carlo Estimates vs True EPIG")
plt.legend()
plt.show()
