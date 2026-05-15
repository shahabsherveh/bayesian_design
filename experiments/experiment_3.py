from bed.data import get_mnist_data
from bed.models import CNN, NeuralNetworkClassifier
from bed.experiments import Experiment
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx


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
results = experiment.run_experiment(
    experiments=[
        "EPIG",
        "EIG",
        "MC",
        "RAND",
    ],
    iterations=epochs,
)
results.plot_comparison()
plt.show()
