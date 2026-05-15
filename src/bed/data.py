from jax import numpy as jnp
import jax
from scipy import stats
import tensorflow_datasets as tfds
import tensorflow as tf

from bed.models import FlaxModel


class Data:
    def __init__(
        self,
        x_train: jnp.ndarray,
        y_train: jnp.ndarray,
        x_test: jnp.ndarray,
        y_test: jnp.ndarray,
        underlying_model: None | FlaxModel = None,
    ):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.underlying_model = underlying_model

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

    def observe(self, index):
        if jnp.isdtype(index.dtype, "int"):
            if index < len(self.x_train):
                obs = self.y_train[index]
            else:
                obs = self.y_test[index - len(self.x_train)]
        else:
            obs = self.underlying_model(index, rngs=None).squeeze()

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
) -> Data:
    key = jax.random.PRNGKey(0)
    eigs = jnp.full(fill_value=embedding_noise_std, shape=input_dim)
    eigs = eigs.at[:embedding_dim].set(
        (input_dim / embedding_dim)
        - (embedding_noise_std * (input_dim - embedding_dim) / embedding_dim)
    )
    eigs = eigs.at[embedding_dim:].set(embedding_noise_std)
    design_cov = stats.random_correlation.rvs(
        eigs=eigs,
        random_state=1,
        tol=1e-6,
        diag_tol=1e-6,
    )
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


def get_mnist_data(num_train: int, num_test: int, batch_size: int = 32) -> Data:
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
