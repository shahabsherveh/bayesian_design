import argparse
from numpy import number
from numpy.random import normal
from tqdm import trange
from bed.data import create_synthetic_data
from bed.models import LinearNN, NeuralNetworkRegressor
from bed.experiments import Experiment
from bed.ekf import EKF
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
from copy import deepcopy
import timeit
from tabulate import tabulate

parser = argparse.ArgumentParser(
    prog="Bayesian Active Learning",
    description="Computing EPIG, EIG and EPIG-MC exectution performance in miliseconds",
)
# parser.add_argument('-n', '--number_of_runs', default=10, type=int)
args = parser.parse_args()


plt.rcParams.update({"font.size": 11})
jax.config.update("jax_enable_x64", True)
design_dim = 10
latent_dim = design_dim + 1
latent_true = 1 * \
    jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
latent_var = 0.1
latent_cov = latent_var * jnp.eye(latent_dim)
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
epig_times = ['EPIG']
eig_times = ['EIG']
mc_times = ['EPIG-MC']
rand_times = ['RAndom']
number_of_runs = 1
designs = ['', 'K=10', 'K=50', 'K=100', 'K=500', 'K=1000', 'K=5000']
pbar = trange(1, len(designs))
for i in pbar:
    num_designs = int(designs[i].split('=')[1])
    num_test = num_designs - 1
    num_train = 1
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
    model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_cov=latent_cov,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    ekf = EKF(
        model=experiment.model,
        state_prev=experiment.state_init_prior[0],
        state_cov_prev=experiment.state_init_prior[1],
        state_innovation=experiment.latent_innovation,
        measurement_error=experiment.measurement_error,
    )

    x = experiment.design_space[:1]
    epig = timeit.timeit(lambda: experiment.calculate_epig(x, ekf),
                         number=number_of_runs)
    model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_cov=latent_cov,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    ekf = EKF(
        model=experiment.model,
        state_prev=experiment.state_init_prior[0],
        state_cov_prev=experiment.state_init_prior[1],
        state_innovation=experiment.latent_innovation,
        measurement_error=experiment.measurement_error,
    )
    eig = timeit.timeit(lambda: experiment.calculate_eig(x, ekf),
                        number=number_of_runs)
    model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_cov=latent_cov,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    ekf = EKF(
        model=experiment.model,
        state_prev=experiment.state_init_prior[0],
        state_cov_prev=experiment.state_init_prior[1],
        state_innovation=experiment.latent_innovation,
        measurement_error=experiment.measurement_error,
    )
    mc = timeit.timeit(lambda: experiment.calculate_epig_mc(x, ekf),
                       number=number_of_runs)
    model = LinearNN(input_dim=design_dim, rngs=nnx.Rngs(0))
    experiment = Experiment(
        model=NeuralNetworkRegressor(model),
        data=data,
        latent_cov=latent_cov,
        latent_innovation=latent_innovation,
        measurement_error=measurement_cov,
        plot_inter_results=False,
        pre_train_model=False,
    )
    ekf = EKF(
        model=experiment.model,
        state_prev=experiment.state_init_prior[0],
        state_cov_prev=experiment.state_init_prior[1],
        state_innovation=experiment.latent_innovation,
        measurement_error=experiment.measurement_error,
    )
    if i == 1:
        continue
    rand = timeit.timeit(lambda: normal(), number=number_of_runs)
    epig_times.append(f"{epig*1000/number_of_runs:.1f}")
    eig_times.append(f"{eig*1000/number_of_runs:.1f}")
    mc_times.append(f"{mc*1000/number_of_runs:.1f}")
    rand_times.append(f"{rand*1000/number_of_runs:.1f}")

table = [
    designs,
    epig_times,
    eig_times,
    mc_times,
    # rand_times
]
print(tabulate(table, headers='firstrow', tablefmt='simple_grid'))
# fig, ax = plt.subplots(figsize=(80 / 25.4, 80 / 25.4))
# x_axis = 10 ** (jnp.array(pbar.iterable))
# # ax.set_title("Computation Performance")
# ax.loglog(
#     x_axis,
#     epig_times,
#     label="EPIG",
#     marker="o",
# )
# ax.loglog(
#     x_axis,
#     eig_times,
#     label="EIG",
#     marker="o",
# )
# ax.loglog(
#     x_axis,
#     mc_times,
#     label="EPIG-MC",
#     marker="o",
# )
# ax.loglog(
#     x_axis,
#     rand_times,
#     label="RAND",
#     marker="o",
# )
# ax.set_xlabel("Pool Size")
# ax.set_ylabel("Computation Time (s)")
# plt.legend()
# plt.show()
