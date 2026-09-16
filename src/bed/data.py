"""Data containers and dataset-generation utilities for Bayesian design."""

from jax import numpy as jnp
import jax
from scipy import stats

from bed.models import FlaxModel


def _evaluate_model(model, x):
    """Evaluate Flax models and plain callable test doubles uniformly."""
    if isinstance(model, FlaxModel):
        return model(x, rngs=None)
    return model(x)


def _reshape_outputs(values, count):
    """Normalize model outputs to one row per generated design."""
    values = jnp.asarray(values)
    if values.shape[0] == count:
        return values.reshape(count, -1)
    if values.shape[-1] == count:
        return jnp.moveaxis(values, -1, 0).reshape(count, -1)
    return values.reshape(count, -1)


class Data:
    """Hold training, candidate-pool, and optional global test data.

    Inputs use the batched shapes expected by the measurement models.  The
    ``observe`` method maps an integer index to a stored training observation
    and a design value to a synthetic observation from ``underlying_model``.
    """

    def __init__(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test_pool: jnp.ndarray,
        y_test_pool: jnp.ndarray,
        x_test_glob: jnp.ndarray | None = None,
        y_test_glob: jnp.ndarray | None = None,
        underlying_model: None | FlaxModel = None,
        measurement_noise_std: float | None = None,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test_pool = x_test_pool
        self.y_test_pool = y_test_pool
        self.x_test_glob = x_test_pool if x_test_glob is None else x_test_glob
        self.y_test_glob = y_test_pool if y_test_glob is None else y_test_glob
        self.underlying_model = underlying_model
        self.measurement_noise_std = (
            measurement_noise_std if measurement_noise_std is not None else 0.0
        )
        self._observation_key = jax.random.PRNGKey(0)

    @property
    def x_test(self):
        """Backward-compatible alias for the candidate test pool."""
        return self.x_test_pool

    @property
    def y_test(self):
        """Backward-compatible alias for the candidate test measurements."""
        return self.y_test_pool

    @classmethod
    def from_npy(
        cls,
        x_train_path: str,
        y_train_path: str,
        x_test_path: str,
        y_test_path: str,
    ) -> "Data":
        """Load the four required arrays from NumPy ``.npy`` files."""
        x_train = jnp.load(x_train_path)
        y_train = jnp.load(y_train_path)
        x_test = jnp.load(x_test_path)
        y_test = jnp.load(y_test_path)
        return cls(x_train, y_train, x_test, y_test)

    def to_npy(
        self,
        x_train_path: str,
        y_train_path: str,
        x_test_path: str,
        y_test_path: str,
    ):
        """Serialize the training and test arrays as NumPy ``.npy`` files."""
        jnp.save(x_train_path, self.x_train)
        jnp.save(y_train_path, self.y_train)
        jnp.save(x_test_path, self.x_test)
        jnp.save(y_test_path, self.y_test)

    def observe(self, index, has_noise: bool = True) -> jnp.ndarray:
        """Return the observation associated with a training index or design.

        Integer indices address ``y_train``.  Non-integer inputs are evaluated
        by ``underlying_model`` and optionally receive measurement noise.
        """
        index = jnp.asarray(index)
        if index.ndim == 0 and jnp.issubdtype(index.dtype, jnp.integer):
            obs = self.y_train[index]
        else:
            if self.underlying_model is None:
                train_matches = jnp.all(self.x_train == index, axis=-1)
                test_matches = jnp.all(self.x_test_pool == index, axis=-1)
                if jnp.any(train_matches):
                    return self.y_train[jnp.argmax(train_matches)]
                if jnp.any(test_matches):
                    return self.y_test_pool[jnp.argmax(test_matches)]
                raise ValueError(
                    "A design requires an underlying_model unless it is stored "
                    "in the training or test pool."
                )
            obs = _evaluate_model(self.underlying_model, index).squeeze()
            if has_noise:
                self._observation_key, noise_key = jax.random.split(
                    self._observation_key
                )
                noise = self.measurement_noise_std * jax.random.normal(
                    shape=obs.shape, key=noise_key
                )
            else:
                noise = 0.0
            obs = (obs + noise).squeeze()
        return obs


def create_synthetic_data(
    model,
    num_train: int,
    num_test: int,
    input_dim: int,
    embedding_dim: int,
    embedding_noise_std: float = 0.1,
    measurement_noise_std: float = 0.1,
    output_dim: int = 1,
    key=jax.random.PRNGKey(0),
    var=1.0,
) -> Data:
    """Generate Gaussian design points with an optional informative subspace."""
    eigs = jnp.full(fill_value=embedding_noise_std, shape=input_dim)
    eigs = eigs.at[:embedding_dim].set(
        (input_dim / embedding_dim)
        - (embedding_noise_std * (input_dim - embedding_dim) / embedding_dim)
    )
    eigs = eigs.at[embedding_dim:].set(embedding_noise_std)
    cov = jnp.ones_like(eigs)
    cov = cov.at[:embedding_dim].set(var**0.5)
    cov = jnp.diag(cov)

    if input_dim != 1:
        design_cov = (
            cov
            @ stats.random_correlation.rvs(
                eigs=eigs,
                random_state=1,
                tol=1e-6,
                diag_tol=1e-6,
            )
            @ cov
        )
    else:
        design_cov = jnp.array([[eigs[0]]])

    design_mean = jnp.zeros(input_dim)

    key_train, key_test, key_noise_train, key_noise_test = jax.random.split(
        key, 4)
    x_train = jax.random.multivariate_normal(
        mean=design_mean,
        cov=design_cov,
        shape=(num_train,),
        key=key_train,
        method="svd",
    )[:, None, None, :]
    x_test = jax.random.multivariate_normal(
        mean=design_mean,
        cov=design_cov,
        shape=(num_test,),
        key=key_test,
        method="svd",
    )[:, None, None, :]
    noise_train = measurement_noise_std * jax.random.normal(
        shape=(num_train, output_dim), key=key_noise_train
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(num_test, output_dim), key=key_noise_test
    )

    y_train = _reshape_outputs(
        _evaluate_model(model, x_train), num_train
    )
    y_test = _reshape_outputs(
        _evaluate_model(model, x_test), num_test
    )
    y_train = y_train + noise_train
    y_test = y_test + noise_test
    return Data(x_train, y_train, x_test, y_test, underlying_model=model)


def create_synthetic_normal_with_outliers_data_1D(
    model,
    var,
    mean,
    outliers,
    num_train: int,
    num_test: int,
    key=jax.random.PRNGKey(0),
) -> Data:
    """Generate one-dimensional normal data with explicit outlier designs."""
    x_train = jax.random.normal(shape=(num_train,), key=key) * var + mean
    x_test = jax.random.normal(shape=(num_test,), key=key) * var + mean
    x_test = jnp.append(x_test, outliers)
    y_train = _evaluate_model(model, x_train)
    y_test = _evaluate_model(model, x_test)
    return Data(
        x_train=x_train[:, None, None, None],
        x_test=x_test[:, None, None, None],
        y_train=y_train,
        y_test=y_test,
    )


def create_synthetic_normal_mixture_data_1D(
    model,
    vars: jnp.ndarray,
    means: jnp.ndarray,
    num_train: int,
    num_test: int,
    num_val: int = 0,
    measurement_noise_std: float = 0.1,
    extra_points: jnp.ndarray | None = None,
    key=jax.random.key(0),
    skew: float = 0.0,
) -> Data:
    """Generate a one-dimensional mixture of normal design distributions."""

    design_cov = jnp.diag(vars)
    keys = iter(jax.random.split(key, 2 * len(vars) + 4))
    x_train = jax.random.normal(
        shape=(num_train,), key=next(keys)) * vars[0] + means[0]
    for i in range(1, len(vars)):
        x_train = jnp.append(
            x_train,
            jax.random.normal(shape=(num_train,), key=next(keys)) *
            vars[i] + means[i],
        )
    x_train = x_train[:, None, None, None]
    x_test = jax.random.normal(
        shape=(num_test,), key=next(keys)
    ) * vars[0] + means[0]
    for i in range(1, len(vars)):
        x_test = jnp.append(
            x_test,
            jax.random.normal(
                shape=(num_test,), key=next(keys)
            ) * vars[i] + means[i],
        )
    x_test = x_test[:, None, None, None]
    if extra_points is not None:
        x_train = jnp.append(
            x_train, extra_points[:, None, None, None], axis=0)
    if num_val > 0:
        x_val = jax.random.multivariate_normal(
            mean=means,
            cov=design_cov,
            shape=(num_val,),
            key=next(keys),
            method="svd",
        ).flatten()[:, None, None, None]
        y_val = _evaluate_model(model, x_val)
    else:
        x_val = None
        y_val = None

    extra_count = 0 if extra_points is None else len(extra_points)
    noise_train = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_train + extra_count, 1, 1, 1), key=next(keys)
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_test, 1, 1, 1), key=next(keys)
    )
    y_train = _evaluate_model(model, x_train) + noise_train
    y_test = _evaluate_model(model, x_test) + noise_test
    return Data(
        x_train,
        y_train,
        x_test,
        y_test,
        x_val,
        y_val,
        underlying_model=model,
        measurement_noise_std=measurement_noise_std,
    )


def create_synthetic_skewnormal_mixture_data_1D(
    model,
    vars: jnp.ndarray,
    means: jnp.ndarray,
    num_train: int,
    num_test: int,
    num_val: int = 0,
    measurement_noise_std: float = 0.1,
    extra_points: jnp.ndarray | None = None,
    key=jax.random.PRNGKey(0),
    skews: list[float] | None = None,
) -> Data:
    """Generate a one-dimensional mixture of skew-normal distributions."""

    design_cov = jnp.diag(vars)
    skews = skews if skews is not None else [0] * len(vars)
    x_train = stats.skewnorm.rvs(
        size=num_train, loc=means[0], scale=vars[0] ** 0.5, a=skews[0]
    )
    for i in range(1, len(vars)):
        x_train = jnp.append(
            x_train,
            stats.skewnorm.rvs(
                size=num_train,
                loc=means[i],
                scale=vars[i] ** 0.5,
                a=skews[i],
            ),
        )
    x_train = x_train[:, None, None, None]
    x_test = stats.skewnorm.rvs(
        size=num_test, loc=means[0], scale=vars[0] ** 0.5, a=skews[0]
    )
    for i in range(1, len(vars)):
        x_test = jnp.append(
            x_test,
            stats.skewnorm.rvs(
                size=num_test,
                loc=means[i],
                scale=vars[i] ** 0.5,
                a=skews[i],
            ),
        )
    x_test = x_test[:, None, None, None]
    if extra_points is not None:
        x_train = jnp.append(
            x_train, extra_points[:, None, None, None], axis=0)
    if num_val > 0:
        x_val = jax.random.multivariate_normal(
            mean=means,
            cov=design_cov,
            shape=(num_val,),
            key=key,
            method="svd",
        ).flatten()[:, None, None, None]
        y_val = _evaluate_model(model, x_val)
    else:
        x_val = None
        y_val = None

    extra_count = 0 if extra_points is None else len(extra_points)
    noise_train = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_train + extra_count, 1, 1, 1), key=key
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_test, 1, 1, 1), key=key
    )
    y_train = _evaluate_model(model, x_train) + noise_train
    y_test = _evaluate_model(model, x_test) + noise_test
    return Data(
        x_train,
        y_train,
        x_test,
        y_test,
        x_val,
        y_val,
        underlying_model=model,
        measurement_noise_std=measurement_noise_std,
    )


def create_synthetic_fatailed_data_1D(
    model,
    df,
    mean,
    num_train: int,
    num_test: int,
    measurement_noise_std: float = 0.1,
    key=jax.random.PRNGKey(0),
) -> Data:
    """Generate one-dimensional heavy-tailed data from Student's t samples."""
    x_train = (
        jax.random.t(df=df, shape=(num_train,), key=key)[
            :, None, None, None] - mean
    )
    x_test = jax.random.t(df=df, shape=(num_test,), key=key)[
        :, None, None, None] - mean

    noise_train = measurement_noise_std * jax.random.normal(
        shape=(num_train, 1, 1, 1), key=key
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(num_test, 1, 1, 1), key=key
    )
    y_train = _evaluate_model(model, x_train) + noise_train
    y_test = _evaluate_model(model, x_test) + noise_test
    return Data(x_train, y_train, x_test, y_test, underlying_model=model)


def get_mnist_data(num_train: int, num_test: int, batch_size: int = 32) -> Data:
    """Load and normalize MNIST images, returning one-hot encoded labels."""
    import tensorflow as tf
    import tensorflow_datasets as tfds
    train_ds: tf.data.Dataset = tfds.load("mnist", split="train")
    test_ds: tf.data.Dataset = tfds.load("mnist", split="test")
    train_ds = train_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # normalize train set
    test_ds = test_ds.map(
        lambda sample: {
            "image": tf.cast(sample["image"], tf.float32) / 255,
            "label": sample["label"],
        }
    )  # Normalize the test set.
    # Create a shuffled dataset by allocating a buffer size of 1024 to randomly draw elements from.
    # Group into batches of `batch_size` and skip incomplete batches, prefetch the next sample to improve latency.
    x_train = []
    x_test = []
    y_train = []
    y_test = []
    for i, sample in enumerate(train_ds.as_numpy_iterator()):
        x_train.append(sample["image"])
        y_train.append(sample["label"])
        if i >= num_train - 1:
            break
    for i, sample in enumerate(test_ds.as_numpy_iterator()):
        x_test.append(sample["image"])
        y_test.append(sample["label"])
        if i >= num_test - 1:
            break

    y_train = jax.nn.one_hot(jnp.array(y_train), num_classes=10)
    y_test = jax.nn.one_hot(jnp.array(y_test), num_classes=10)
    return Data(
        x_train=jnp.array(x_train),
        y_train=y_train,
        x_test_pool=jnp.array(x_test),
        y_test_pool=y_test,
    )


def get_uci_data(
    dataset: str,
    test_size: int | float,
    test_pool_quantile: float = .75,
    random_state: int = 0,
    **kwargs
):
    """Load a UCI dataset and split its test set into pool and global subsets."""
    # fetch dataset
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    uci_data = fetch_ucirepo(dataset)

    x = uci_data.data.features
    scaler = StandardScaler()
    x = jnp.array(scaler.fit_transform(x))[:, None, None, :]
    y = jnp.atleast_2d(uci_data.data.targets.values)[:, None, None, :]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state)
    pca = PCA(1)
    projections = pca.fit_transform(x_test.squeeze())
    q = jnp.quantile(projections, q=test_pool_quantile)
    test_glob_mask = (projections < q).squeeze()
    x_test_glob = x_test[test_glob_mask]
    y_test_glob = y_test[test_glob_mask]
    x_test_pool = x_test[~test_glob_mask]
    y_test_pool = y_test[~test_glob_mask]
    return Data(x_train, y_train, x_test_pool, y_test_pool, x_test_glob, y_test_glob)


def get_1d_regression(model, train_size=200, test_pool_size=50, test_glob_size=50, interval_train=[-5, 5], interval_test_pool=[2, 4], key=jax.random.key(0)):
    keys = jax.random.split(key, 3)
    x_train = jax.random.uniform(shape=(train_size, 1, 1, 1),
                                 minval=interval_train[0], maxval=interval_train[1], key=keys[0])
    y_train = _evaluate_model(model, x_train)
    x_test_glob = jax.random.uniform(shape=(test_glob_size, 1, 1, 1),
                                     minval=interval_train[0], maxval=interval_train[1], key=keys[1])
    y_test_glob = _evaluate_model(model, x_test_glob)
    x_test_pool = jax.random.uniform(shape=(
        test_pool_size, 1, 1, 1), minval=interval_test_pool[0], maxval=interval_test_pool[1], key=keys[2])
    y_test_pool = _evaluate_model(model, x_test_pool)
    return Data(
        x_train,
        y_train,
        x_test_pool,
        y_test_pool,
        x_test_glob,
        y_test_glob
    )
