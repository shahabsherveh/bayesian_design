import typer
import jax
import jax.numpy as jnp
from scipy import stats

app = typer.Typer()


@app.callback()
def callback():
    """
    Awesome Portal Gun
    """
    print("Welcome to the Bayesian Experimental Design CLI!")


@app.command()
def experiment(
    model: str = typer.Option("linear", help="The number of the experiment to run"),
    design_dimension: int = typer.Option(
        2, help="The dimensionality of the design space"
    ),
    etrained_model: bool = typer.Option(
        False, help="Whether to use a pretrained model"
    ),
    x_train_path: str = typer.Option(
        "data/x_train.csv", help="Path to the training data"
    ),
    y_train_path: str = typer.Option(
        "data/y_train.csv", help="Path to the training labels"
    ),
    x_test_path: str = typer.Option("data/x_test.csv", help="Path to the test data"),
    y_test_path: str = typer.Option("data/y_test.csv", help="Path to the test labels"),
):
    jax.config.update("jax_enable_x64", True)
    latent_dim = 24 + 20 + 5
    design_dim = 5
    latent_true = 2 * jax.random.normal(jax.random.PRNGKey(1234), (latent_dim, 1)) + 0
    latent_var = 0.1
    latent_innovation = 0
    measurement_cov = 0.1 * jnp.eye(1)
    design_pool_num = 100
    design_mean = jnp.zeros(design_dim)
    eigs = jnp.full(design_dim, fill_value=0.01, dtype="float64")
    eigs = eigs.at[:2].set(1.99)
    eigs = design_dim * eigs / eigs.sum()
    gd_params = {
        "learning_rate": 1,
        "max_iters": 0,
        "tol": 1e-3,
        "stochastic": False,
        "x_init_type": "best_pool",
    }
    # epochs = int(10 * latent_dim)
    epochs = 100
    design_cov = 10 * stats.random_correlation.rvs(
        eigs=eigs,
        random_state=1,
        tol=1e-6,
        diag_tol=1e-6,
    )
    # design_cov = jnp.array([[1.0, 0.99], [0.99, 1.0]])
    random_key = jax.random.PRNGKey(0)
    model = NeuralNetworkModel(
        input_dim=design_dim,
        hidden_dim_0=4,
        hidden_dim_1=4,
        key=random_key,
    )
    x_train = stats.multivariate_normal(mean=design_mean, cov=design_cov).rvs(size=100)
    y_train = model(latent_true, x_train.T)
    loss_values = model.train(x_train, y_train, epochs=200, learning_rate=0.01)
    plot_results = False
    print("Running experiment...")
