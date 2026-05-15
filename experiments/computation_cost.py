from numpy.random import normal
from tqdm import trange
from bed.data import create_synthetic_data
from bed.models import LinearNN, NeuralNetworkRegressor
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from copy import deepcopy
import timeit


jax.config.update("jax_enable_x64", True)
design_dim = 10
latent_dim = design_dim + 1
latent_true = 1 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 0.1
latent_innovation = 0
measurement_cov = 0.25 * jnp.eye(1)
# epochs = int(10 * latent_dim)
# design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
random_key = jax.random.PRNGKey(0)
model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
model_true = deepcopy(model)
state_true = model_true.weights_to_state(latent_true)
nnx.update(model_true, state_true)
plot_results = False
pbar = trange(1000, 14000, 1000)
epig_times = []
eig_times = []
mc_times = []
rand_times = []
for num_designs in pbar:
    num_test = num_designs // 2
    num_train = num_designs // 2
    data = create_synthetic_data(
        model_true,
        num_train,
        num_test,
        design_dim,
        embedding_dim=1,
        embedding_noise_std=0.0010,
        measurement_noise_std=jnp.sqrt(measurement_cov[0, 0]),
        # measurement_noise_std=0,
    )
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    x = experiment.design_space[:1]
    epig = timeit.timeit(lambda: experiment.calculate_epig(x), number=1)
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    eig = timeit.timeit(lambda: experiment.calculate_eig(x), number=1)
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    mc = timeit.timeit(lambda: experiment.calculate_epig_mc(x), number=1)
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_var=latent_var,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    rand = timeit.timeit(lambda: normal(), number=1)
    epig_times.append(epig)
    eig_times.append(eig)
    mc_times.append(mc)
    rand_times.append(rand)

fig, ax = plt.subplots()
ax.set_title("Computation Performance")
ax.plot(pbar.iterable, epig_times, label="EPIG")
ax.plot(pbar.iterable, eig_times, label="EIG")
ax.plot(pbar.iterable, mc_times, label="MC")
ax.plot(pbar.iterable, rand_times, label="RAND")
ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("Computation Time (s)")
plt.legend()
plt.show()
