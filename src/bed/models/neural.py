"""Flax/NNX models and latent-parameter adapters."""

from copy import deepcopy
from functools import partial

import jax.numpy as jnp
import optax
from flax import nnx
from flax.nnx.nn import initializers as nnx_initializers
from tqdm import tqdm

from ._base import Model


class FlaxModel(nnx.Module):
    """Base class for NNX models with flattened latent parameters.

    Subclasses create their layers in ``__init__`` and then call
    ``self.weight_mapping, self.weight_size = self._create_weight_mapping()``.
    The flattened latent state and everything derived from it assume those two
    attributes (plus ``output_dim``) exist, so they are declared here.
    """

    # Populated by the subclass in ``__init__``; see ``_create_weight_mapping``.
    weight_mapping: dict
    weight_size: int
    output_dim: int

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        """Evaluate the network at a batch of designs."""
        raise NotImplementedError

    def _validate_parameters(self, z):
        """Validate the shape of a flattened parameter vector."""
        raise NotImplementedError

    def _validate_input(self, x):
        """Validate the shape of a model input."""
        raise NotImplementedError

    def _create_weight_mapping(self):
        """Map every parameter path of the NNX state to a slice of the flat vector.

        Works for any nesting depth (e.g. ``self.output`` in ``LinearNN``,
        ``self.layers[i]`` in ``DenseNN``, ``self.conv1`` in ``CNN``).
        Keys are the full parameter paths as tuples; the last element is the
        parameter name (``"kernel"`` or ``"bias"``).
        """
        mapping = {}
        idx = 0
        for path, param in nnx.to_flat_state(nnx.state(self)):
            size = param.get_value().size
            shape = param.shape
            mapping[tuple(path)] = {
                "slice": (idx, idx + size),
                "shape": shape,
            }
            idx += size
        return mapping, idx

    def weights_to_state(self, z):
        """Convert a flat latent vector into an NNX parameter state."""
        self._validate_parameters(z)
        flat = []
        for path, meta in self.weight_mapping.items():
            slice = meta["slice"]
            shape = meta["shape"]
            w = z[slice[0]: slice[1]].reshape(shape)
            flat.append((path, nnx.Param(w)))
        return nnx.from_flat_state(flat)

    def state_to_weights(self, state: nnx.State):
        """Flatten an NNX parameter state using the model's stable layout."""
        weights = jnp.zeros((self.weight_size, 1))
        flat = dict(nnx.to_flat_state(state))
        for path, meta in self.weight_mapping.items():
            w = flat[path].get_value().flatten()[:, None]
            weights = weights.at[meta["slice"][0]: meta["slice"][1]].set(w)
        return weights

    def _validate_parameters(self, z):
        """Ensure that ``z`` contains one value for every model parameter."""
        assert z.shape[0] == (self.weight_size), (
            "Parameter vector z has incorrect size."
        )

    def loss(self, x, y, rngs: nnx.Rngs):
        """Compute the task-specific training loss."""
        raise NotImplementedError("Subclasses must implement the loss method.")


class DenseNN(FlaxModel):
    """Fully connected network with configurable hidden activations."""

    def __init__(
            self,
            input_dim,
            hidden_dims,
            output_dim,
            rngs: nnx.Rngs,
            activation="gelu",
    ):
        """Create a dense network with the requested hidden activations.

        ``activation`` may be one activation name/callable applied to every
        hidden layer, or a sequence with one activation per hidden layer.
        """
        nodes = [input_dim] + hidden_dims + [output_dim]
        self.activations = self._resolve_activations(activation, len(hidden_dims))
        self.layers = nnx.List()
        for i in range(len(nodes)-1):
            self.layers.append(nnx.Linear(nodes[i], nodes[i + 1], rngs=rngs, kernel_init=nnx_initializers.variance_scaling(
                scale=2.0, mode="fan_in", distribution="truncated_normal")))
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.weight_mapping, self.weight_size = self._create_weight_mapping()

    @staticmethod
    def _resolve_activations(activation, hidden_count):
        available = {
            "gelu": nnx.gelu,
            "relu": nnx.relu,
            "sigmoid": nnx.sigmoid,
            "tanh": jnp.tanh,
            "silu": nnx.silu,
            "swish": nnx.silu,
            "elu": nnx.elu,
            "leaky_relu": nnx.leaky_relu,
            "identity": lambda value: value,
            "linear": lambda value: value,
        }
        requested = (
            [activation] * hidden_count
            if isinstance(activation, str) or callable(activation)
            else list(activation)
        )
        if len(requested) != hidden_count:
            raise ValueError(
                "activation must be a single function/name or one per hidden layer."
            )
        resolved = []
        for function in requested:
            if isinstance(function, str):
                try:
                    function = available[function.lower()]
                except KeyError as error:
                    raise ValueError(
                        f"Unsupported activation '{function}'. "
                        f"Choose from: {', '.join(sorted(available))}."
                    ) from error
            if not callable(function):
                raise TypeError("Each activation must be a callable or supported name.")
            resolved.append(function)
        return tuple(resolved)

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        """Apply hidden activations followed by a linear output layer."""
        output = x
        for layer, activation in zip(self.layers[:-1], self.activations):
            output = activation(layer(output))
        output = self.layers[-1](output)
        return output

    def _validate_input(self, x):
        """Check that the leading input dimension matches ``input_dim``."""
        assert x.shape[0] == self.input_dim, "Input x has incorrect dimensionality."


class NNFLax(DenseNN):
    """Backward-compatible dense network constructor."""

    def __init__(
        self,
        input_dim,
        hidden_dim_0,
        hidden_dim_1,
        rngs: nnx.Rngs,
        activation="gelu",
    ):
        super().__init__(
            input_dim=input_dim,
            hidden_dims=[hidden_dim_0, hidden_dim_1],
            output_dim=1,
            rngs=rngs,
            activation=activation,
        )


class LinearNN(FlaxModel):
    """Single linear layer used as a differentiable baseline."""

    def __init__(self, input_dim, rngs: nnx.Rngs):
        """Create a one-output linear network."""
        self.output = nnx.Linear(input_dim, 1, rngs=rngs)
        self.input_dim = input_dim
        self.output_dim = 1
        self.weight_mapping, self.weight_size = self._create_weight_mapping()

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        """Evaluate the linear layer."""
        output = self.output(x)
        return output


class Sinus(FlaxModel):
    """Fixed sinusoidal forward model."""

    def __init__(self, input_dim, freq, amp=1, noise_std=0.1):
        """Create ``amp * sin(2 pi freq x)``; extra arguments are compatibility parameters."""
        self.output = lambda x: amp * jnp.sin(2 * jnp.pi * freq * x)

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        """Evaluate the sinusoidal response."""
        output = self.output(x)
        return output


class SinusInverse(FlaxModel):
    """Fixed inverse-coordinate sinusoidal forward model."""

    def __init__(self, input_dim, freq_init, amp=1, noise_std=0.1):
        """Create ``amp * sin(2 pi freq_init / x)``."""
        self.output = lambda x: amp * (jnp.sin(2 * jnp.pi * freq_init / x))

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        """Evaluate the inverse sinusoidal response."""
        output = self.output(x)
        return output


class CNN(FlaxModel):
    """Small convolutional classifier for image-shaped designs."""

    def __init__(self, *, rngs: nnx.Rngs):
        """Build the convolution, pooling, dropout, and output layers."""
        self.conv1 = nnx.Conv(1, 2, kernel_size=(3, 3), rngs=rngs)
        # self.batch_norm1 = nnx.BatchNorm(8, rngs=rngs)
        self.dropout1 = nnx.Dropout(rate=0.025)
        self.conv2 = nnx.Conv(2, 2, kernel_size=(3, 3), rngs=rngs)
        # self.batch_norm2 = nnx.BatchNorm(16, rngs=rngs)
        self.avg_pool = partial(
            nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
        self.output = nnx.Linear(7 * 7 * 2, 10, rngs=rngs)
        # self.dropout2 = nnx.Dropout(rate=0.025)
        # self.linear2 = nnx.Linear(128, 10, rngs=rngs)
        self.output_dim = 10
        self.weight_mapping, self.weight_size = self._create_weight_mapping()

    def __call__(self, x, rngs: nnx.Rngs | None = None):
        """Apply convolutional feature extraction and classify the image."""
        if x.ndim == 3:
            flattened_shape = 1
        else:
            flattened_shape = x.shape[0]
        conv1 = self.avg_pool(
            nnx.relu(self.dropout1(self.conv1(x), rngs=rngs)))
        conv2 = self.avg_pool(nnx.relu(self.conv2(conv1)))
        conv2_flatten = conv2.reshape(flattened_shape, 1, 1, -1)  # flatten
        output = self.output(conv2_flatten)
        return output


class NeuralNetworkBase(Model):
    """
    Nonlinear measurement model implemented as a simple feedforward neural network.
    This model captures complex relationships between latent states and designs,
    making it suitable for testing the Extended Kalman Filter's ability to
    handle nonlinear dynamics.
    Attributes:
        W1: Weight matrix for first layer
        b1: Bias vector for first layer
        W2: Weight matrix for second layer
        b2: Bias vector for second layer
    """

    def __init__(self, model: FlaxModel):
        """
        Wrap an NNX model as a measurement model.

        Args:
            model: NNX model whose parameters form the latent state.
        """
        self.flax_model = model
        self.graphdef, _ = nnx.split(self.flax_model)

    def build_loss_fn(self, x_train, y_train):
        """Return the task-specific loss closure used by :meth:`train`."""
        return lambda x: None

    def __call__(self, z, x, **kwargs):
        """
        Compute nonlinear measurement using a feedforward neural network.
        Args:
            z: Latent state vector of shape (d, 1)
            x: Design vector/matrix of shape (d, n)
        Returns:
            Measurements of shape (1, n) after passing through the network
        """
        model = deepcopy(self.flax_model)
        if z is not None:
            state = model.weights_to_state(z)
            # model = nnx.merge(self.graphdef, state)
            nnx.update(model, state)
        output = model(x)  # Transpose x to match expected input shape
        return output

    def jacobian(self, z, x):
        """Return derivatives of each output with respect to flat parameters."""
        model = deepcopy(self.flax_model)
        state = model.weights_to_state(z)
        nnx.update(model, state)
        grad_fn = [
            nnx.vmap(
                nnx.grad(lambda model, x, i=i: model(x)[0, 0, i]),
                in_axes=(None, 0),
            )
            for i in range(model.output_dim)
        ]
        grad = [g_fn(model, x) for g_fn in grad_fn]
        grad_weights = [
            nnx.vmap(lambda gr: model.state_to_weights(gr).T)(g) for g in grad
        ]

        jac_value = jnp.concatenate(grad_weights, axis=1)
        return jac_value

    def train(
            self, x_train, y_train, rngs: nnx.Rngs, epochs=200, learning_rate=0.1, **kwargs
    ):
        """
        Train the neural network model using mean squared error loss.

        Args:
            x_train: Training design matrix of shape (n_samples, n_features)
            y_train: Training observations of shape (n_samples,)

        Note:
            This is a simple training loop for demonstration purposes. In practice,
            you may want to use more sophisticated optimization techniques and
            regularization.
        """
        model = self.flax_model
        loss_fn = self.build_loss_fn(x_train, y_train)

        optimizer = nnx.optimizer.Optimizer(
            model, tx=optax.adamw(learning_rate=learning_rate), wrt=nnx.Param
        )
        pbar = tqdm(range(epochs), desc="Training Neural Network Model")
        for epoch in pbar:
            grad_fn = nnx.value_and_grad(loss_fn)
            loss, grads = grad_fn(model, rngs)
            optimizer.update(model, grads)
            pbar.set_postfix({"loss": loss})

        self.mse = loss

        return self


class NeuralNetworkClassifier(NeuralNetworkBase):
    """NNX measurement model trained with softmax cross-entropy."""

    def build_loss_fn(self, x_train, y_train):
        """Build a mean softmax cross-entropy loss for one training batch."""
        def loss_fn(model, rngs: nnx.Rngs):
            """Compute cross-entropy for the current model parameters."""
            logits = model(x_train, rngs)
            loss = optax.softmax_cross_entropy(
                logits=logits.squeeze(), labels=y_train.squeeze()
            ).mean()
            return loss

        return loss_fn


class NeuralNetworkRegressor(NeuralNetworkBase):
    """NNX measurement model trained with mean squared error."""

    def build_loss_fn(self, x_train, y_train):
        """Build a mean squared error loss for one training batch."""
        def loss_fn(model, rngs: nnx.Rngs):
            """Compute regression error for the current model parameters."""
            predictions = model(x_train, rngs).squeeze()
            loss = jnp.mean((predictions - y_train.squeeze()) ** 2)
            return loss

        return loss_fn
