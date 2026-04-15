from flax import nnx
from jax import numpy as jnp


def weights_to_state(weights, model: nnx.Module):
    """
    Convert flat weights vector to model state dictionary.

    Args:
        weights: Flat vector of model parameters (shape: [num_params, 1])
        model: Flax model instance to determine parameter shapes

    Returns:
        state_dict: Dictionary mapping parameter names to reshaped arrays
    """
    state_dict = {}
    model_state = nnx.state(model)
    params = nnx.filter_state(model_state, nnx.Param)
    idx = 0
    for layer_name, layer_params in params.items():
        state_dict[layer_name] = {}
        for weight_name, weight_array in layer_params.items():
            num_elements = weight_array.size
            state_dict[layer_name][weight_name] = weights[
                idx : idx + num_elements
            ].reshape(weight_array.shape)
            idx += num_elements
    state = nnx.State(state_dict)
    return state


def state_to_weights(state: nnx.State):
    """
        Convert model state dictionary to flat weights vector.
    Args:
        state: Flax model state containing parameters
    Returns:
        weights: Flat vector of model parameters (shape: [num_params, 1])
    """
    params = nnx.filter_state(state, nnx.Param)
    weights_list = []
    for layer_name, layer_params in params.items():
        for weight_name, weight_array in layer_params.items():
            weights_list.append(weight_array.flatten())

    weights = jnp.concatenate(weights_list)[:, None]
    return weights
