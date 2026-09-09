from jax import numpy as jnp
import jax
from scipy import stats

from bed.models import FlaxModel


class Data:
    def __init__(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        x_val: jnp.ndarray | None = None,
        y_val: jnp.ndarray | None = None,
        underlying_model: None | FlaxModel = None,
        measurement_noise_std: float | None = None,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.x_val = (
            x_val if x_val is not None else jnp.concatenate([x_train, x_test], axis=0)
        )
        self.y_val = (
            y_val if x_val is not None else jnp.concatenate([y_train, y_test], axis=0)
        )
        self.underlying_model = underlying_model
        self.measurement_noise_std = (
            measurement_noise_std if measurement_noise_std is not None else 0.0
        )

    @classmethod
    def from_npy(
        cls,
        x_train_path: str,
        y_train_path: str,
        x_test_path: str,
        y_test_path: str,
    ) -> "Data":
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
        jnp.save(x_train_path, self.x_train)
        jnp.save(y_train_path, self.y_train)
        jnp.save(x_test_path, self.x_test)
        jnp.save(y_test_path, self.y_test)

    def observe(self, index, has_noise: bool = True) -> jnp.ndarray:
        if jnp.isdtype(index.dtype, "int"):
            if index < len(self.x_train):
                obs = self.y_train[index]
            else:
                obs = self.y_test[index - len(self.x_train)]
        else:
            obs = self.underlying_model(index, rngs=None).squeeze()
            if has_noise:
                noise = self.measurement_noise_std * jax.random.normal(
                    shape=obs.shape, key=jax.random.PRNGKey(0)
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

    x_train = jax.random.multivariate_normal(
        mean=design_mean,
        cov=design_cov,
        shape=(num_train,),
        key=key,
        method="svd",
    )[:, None, None, :]
    x_test = jax.random.multivariate_normal(
        mean=design_mean,
        cov=design_cov,
        shape=(num_test,),
        key=key,
        method="svd",
    )[:, None, None, :]
    noise_train = measurement_noise_std * jax.random.normal(
        shape=(num_train, 1, 1, output_dim), key=key
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(num_test, 1, 1, output_dim), key=key
    )

    y_train = model(x_train, rngs=None)
    y_test = model(x_test, rngs=None)
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
    x_train = jax.random.normal(shape=(num_train,), key=key) * var + mean
    x_test = jax.random.normal(shape=(num_test,), key=key) * var + mean
    x_test = jnp.append(x_test, outliers)
    y_train = model(x_train, rngs=None)
    y_test = model(x_test, rngs=None)
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

    design_cov = jnp.diag(vars)
    x_train = jax.random.normal(shape=(num_train,), key=key) * vars[0] + means[0]
    for i in range(1, len(vars)):
        x_train = jnp.append(
            x_train,
            jax.random.normal(shape=(num_train,), key=key) * vars[i] + means[i],
        )
    x_train = x_train[:, None, None, None]
    x_test = jax.random.normal(shape=(num_test,), key=key) * vars[0] + means[0]
    for i in range(1, len(vars)):
        x_test = jnp.append(
            x_test,
            jax.random.normal(shape=(num_test,), key=key) * vars[i] + means[i],
        )
    x_test = x_test[:, None, None, None]
    if extra_points is not None:
        x_train = jnp.append(x_train, extra_points[:, None, None, None], axis=0)
    if num_val > 0:
        x_val = jax.random.multivariate_normal(
            mean=means,
            cov=design_cov,
            shape=(num_val,),
            key=key,
            method="svd",
        ).flatten()[:, None, None, None]
        y_val = model(x_val, rngs=None)
    else:
        x_val = None
        y_val = None

    noise_train = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_train + len(extra_points), 1, 1, 1), key=key
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_test, 1, 1, 1), key=key
    )
    y_train = model(x_train, rngs=None) + noise_train
    y_test = model(x_test, rngs=None) + noise_test
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
        x_train = jnp.append(x_train, extra_points[:, None, None, None], axis=0)
    if num_val > 0:
        x_val = jax.random.multivariate_normal(
            mean=means,
            cov=design_cov,
            shape=(num_val,),
            key=key,
            method="svd",
        ).flatten()[:, None, None, None]
        y_val = model(x_val, rngs=None)
    else:
        x_val = None
        y_val = None

    noise_train = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_train + len(extra_points), 1, 1, 1), key=key
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(len(vars) * num_test, 1, 1, 1), key=key
    )
    y_train = model(x_train, rngs=None) + noise_train
    y_test = model(x_test, rngs=None) + noise_test
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
    x_train = (
        jax.random.t(df=df, shape=(num_train,), key=key)[:, None, None, None] - mean
    )
    x_test = jax.random.t(df=df, shape=(num_test,), key=key)[:, None, None, None] - mean

    noise_train = measurement_noise_std * jax.random.normal(
        shape=(num_train, 1, 1, 1), key=key
    )
    noise_test = measurement_noise_std * jax.random.normal(
        shape=(num_test, 1, 1, 1), key=key
    )
    y_train = model(x_train, rngs=None) + noise_train
    y_test = model(x_test, rngs=None) + noise_test
    return Data(x_train, y_train, x_test, y_test, underlying_model=model)


def get_mnist_data(num_train: int, num_test: int, batch_size: int = 32) -> Data:
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
        x_test=jnp.array(x_test),
        y_test=y_test,
    )

def get_uci_data(dataset:str,test_size:int|float):
    # fetch dataset 
    from ucimlrepo import fetch_ucirepo
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    uci_data = fetch_ucirepo(dataset) 
      
    x = uci_data.data.features
    scaler = StandardScaler()
    x = jnp.array(scaler.fit_transform(x)[:,None,None,:])
    y = jnp.atleast_2d(uci_data.data.targets.values)[:,None,None,:]
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=test_size)
    return Data(x_train,y_train,x_test,y_test)
