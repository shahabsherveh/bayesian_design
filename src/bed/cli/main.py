from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import typer
import matplotlib.pyplot as plt

from bed.data import get_uci_data


app = typer.Typer()
CONFIG_DIR = Path(__file__).resolve().parents[3] / "experiments" / "configs"


def _load_config(config: str) -> dict:
    import tomllib
    config_path = Path(config)
    if not config_path.exists():
        config_path = CONFIG_DIR / f"{config}.toml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config '{config}' not found. Use a path or one of: "
            + ", ".join(sorted(p.stem for p in CONFIG_DIR.glob("*.toml")))
        )
    with config_path.open("rb") as f:
        cfg = tomllib.load(f)
    cfg["_path"] = str(config_path)
    return cfg


def _build_model(model_cfg: dict):
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from bed.models import CNN, DenseNN, LinearNN, NeuralNetworkClassifier, NeuralNetworkRegressor
    model_type = model_cfg["type"]
    seed = model_cfg.get("seed", 0)
    if model_type == "LinearNN":
        model = LinearNN(input_dim=model_cfg["input_dim"], rngs=nnx.Rngs(seed))
        wrapper = NeuralNetworkRegressor(model)
    elif model_type == "DNN":
        model = DenseNN(
            input_dim=model_cfg["input_dim"],
            hidden_dims=model_cfg["hidden_dims"],
            output_dim=model_cfg["output_dim"],
            rngs=nnx.Rngs(seed),
        )
        wrapper = NeuralNetworkRegressor(model)
    elif model_type == "CNN":
        model = CNN(rngs=nnx.Rngs(seed))
        wrapper = NeuralNetworkClassifier(model)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    return model, wrapper


def _build_latent_cov(model, latent_cfg: dict):
    import jax.numpy as jnp
    cov_type = latent_cfg["type"]
    if cov_type == "scaled_identity":
        return latent_cfg["scale"] * jnp.eye(model.weight_size)
    if cov_type == "he_diagonal":
        bias_var = latent_cfg.get("bias_var", 1.0)
        kernel_scale = latent_cfg.get("kernel_scale", 2.0)
        diag = jnp.zeros((model.weight_size,))
        for (_, param_name), meta in model.weight_mapping.items():
            s0, s1 = meta["slice"]
            shape = meta["shape"]
            fan_in = shape[0]
            var = bias_var if param_name == "bias" else (kernel_scale / fan_in)
            diag = diag.at[s0:s1].set(var)
        return jnp.diag(diag)
    raise ValueError(f"Unsupported latent covariance type: {cov_type}")


def _build_data(model, data_cfg: dict, truth_cfg: dict | None):
    import jax
    import jax.numpy as jnp
    from flax import nnx

    from bed.data import (
        create_synthetic_data,
        create_synthetic_normal_mixture_data_1D,
        create_synthetic_skewnormal_mixture_data_1D,
        get_mnist_data,
    )
    kind = data_cfg["kind"]
    if kind == "mnist":
        return get_mnist_data(
            num_train=data_cfg["num_train"],
            num_test=data_cfg["num_test"],
            batch_size=data_cfg.get("batch_size", 32),
        )
    if kind == "uci":
        dataset= data_cfg["name"]
        test_size=data_cfg['test_size']
        return get_uci_data(dataset=dataset, test_size=test_size)

    model_true = deepcopy(model)
    if truth_cfg and truth_cfg.get("enabled", True):
        key = jax.random.PRNGKey(truth_cfg.get("seed", 1234))
        scale = truth_cfg.get("scale", 1.0)
        bias = truth_cfg.get("bias", 0.0)
        latent_true = scale * jax.random.normal(key, (model.weight_size, 1)) + bias
        state_true = model_true.weights_to_state(latent_true)
        nnx.update(model_true, state_true)

    if kind == "synthetic":
        return create_synthetic_data(
            model_true,
            num_train=data_cfg["num_train"],
            num_test=data_cfg["num_test"],
            input_dim=data_cfg["input_dim"],
            embedding_dim=data_cfg["embedding_dim"],
            embedding_noise_std=data_cfg.get("embedding_noise_std", 0.1),
            measurement_noise_std=data_cfg.get("measurement_noise_std", 0.1),
            output_dim=data_cfg.get("output_dim", 1),
            key=jax.random.PRNGKey(data_cfg.get("seed", 0)),
            var=data_cfg.get("var", 1.0),
        )
    if kind == "normal_mixture_1d":
        return create_synthetic_normal_mixture_data_1D(
            model_true,
            vars=jnp.array(data_cfg["vars"]),
            means=jnp.array(data_cfg["means"]),
            num_train=data_cfg["num_train"],
            num_test=data_cfg["num_test"],
            num_val=data_cfg.get("num_val", 0),
            measurement_noise_std=data_cfg.get("measurement_noise_std", 0.1),
            extra_points=(
                jnp.array(data_cfg["extra_points"])
                if "extra_points" in data_cfg
                else None
            ),
            key=jax.random.PRNGKey(data_cfg.get("seed", 0)),
            skew=data_cfg.get("skew", 0.0),
        )
    if kind == "skewnormal_mixture_1d":
        return create_synthetic_skewnormal_mixture_data_1D(
            model_true,
            vars=jnp.array(data_cfg["vars"]),
            means=jnp.array(data_cfg["means"]),
            num_train=data_cfg["num_train"],
            num_test=data_cfg["num_test"],
            num_val=data_cfg.get("num_val", 0),
            measurement_noise_std=data_cfg.get("measurement_noise_std", 0.1),
            extra_points=(
                jnp.array(data_cfg["extra_points"])
                if "extra_points" in data_cfg
                else None
            ),
            key=jax.random.PRNGKey(data_cfg.get("seed", 0)),
            skews=data_cfg.get("skews"),
        )
    raise ValueError(f"Unsupported data kind: {kind}")


def run_experiment_from_config(
    config: str,
    iterations: int | None = None,
    strategies: list[str] | None = None,
    show_plot: bool = True,
    dry_run: bool = False,
):
    import jax
    from jax import numpy as jnp
    from flax import nnx
    from bed.experiments import Experiment
    cfg = _load_config(config)
    jax.config.update("jax_enable_x64", cfg.get("runtime", {}).get("x64", True))

    model, wrapped_model = _build_model(cfg["model"])
    latent_cov = _build_latent_cov(model, cfg["latent_cov"])
    data = _build_data(model, cfg["data"], cfg.get("truth"))
    training_cfg = cfg.get("training", {})
    training_kwargs = {
        "learning_rate": training_cfg.get("learning_rate", 0.01),
        "epochs": training_cfg.get("epochs", 200),
        "rngs": nnx.Rngs(training_cfg.get("seed", 0)),
    }

    measurement_cfg = cfg["measurement_error"]
    measurement_error = measurement_cfg["variance"] * jnp.eye(measurement_cfg["dim"])

    if dry_run:
        return {
            "config": cfg["_path"],
            "model_type": cfg["model"]["type"],
            "data_kind": cfg["data"]["kind"],
            "design_pool_size": int(data.x_train.shape[0] + data.x_test.shape[0]),
            "strategies": strategies or cfg["run"]["strategies"],
            "iterations": iterations or cfg["run"]["iterations"],
        }

    experiment = Experiment(
        model=wrapped_model,
        data=data,
        latent_cov=latent_cov,
        latent_innovation=cfg["experiment"].get("latent_innovation", 0),
        measurement_error=measurement_error,
        plot_inter_results=(
            cfg["experiment"].get("plot_inter_results", False) and show_plot
        ),
        pre_train_model=cfg["experiment"].get("pre_train_model", False),
        training_kwargs=training_kwargs,
        filter_type=cfg["experiment"].get("filter_type", "ekf"),
    )

    run_cfg = cfg["run"]
    results = experiment.run_experiment(
        experiments=strategies or run_cfg["strategies"],
        iterations=iterations or run_cfg["iterations"],
        optimizer_method=run_cfg.get("optimizer_method", "brute_force"),
        optimizer_params=run_cfg.get("optimizer_params", {"lr": 1, "max_iters": 50}),
    )
    if show_plot:
        results.plot_comparison()
        plt.show()
    return results


@app.callback()
def callback():
    """
    Awesome Portal Gun
    """
    print("Welcome to the Bayesian Experimental Design CLI!")


@app.command("list-configs")
def list_configs():
    """List bundled experiment config names."""
    names = sorted(p.stem for p in CONFIG_DIR.glob("*.toml"))
    for name in names:
        print(name)


@app.command("experiment")
def experiment(
    config: str = typer.Option(
        "experiment_0",
        "--config",
        "-c",
        help="Config name in experiments/configs or a path to a .toml file.",
    ),
    iterations: int | None = typer.Option(
        None,
        "--iterations",
        "-n",
        help="Override iteration count from config.",
    ),
    strategies: str | None = typer.Option(
        None,
        "--strategies",
        help="Comma-separated strategy list (e.g. EPIG,EIG,EPIG-MC,RAND).",
    ),
    no_plot: bool = typer.Option(
        False, "--no-plot", help="Skip plotting and interactive windows."
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Validate config and print execution summary."
    ),
):
    strategy_list = (
        [s.strip() for s in strategies.split(",") if s.strip()]
        if strategies is not None
        else None
    )
    result = run_experiment_from_config(
        config=config,
        iterations=iterations,
        strategies=strategy_list,
        show_plot=not no_plot,
        dry_run=dry_run,
    )
    if dry_run:
        print(result)
if __name__ == "__main__":
   results = run_experiment_from_config(
       config="experiment_4",
   )