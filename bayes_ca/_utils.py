from typing import Union

import jax.numpy as jnp
from jaxtyping import Array, Float, Int


def _safe_handling_params(param: Union[Float, Float[Array, "num_features"]], num_features: Int):
    """
    Coerces supplied parameter to have shape (num_features, ).

    Parameters
    ----------
    param: float or array
    num_features: int
    """
    if isinstance(param, float):
        return jnp.asarray([param]) * jnp.ones(num_features)
    elif isinstance(param, Array):
        coerced = jnp.squeeze(param)  # drop any trailing dimensions
        if (num_features > 1) and (coerced.shape[0] != num_features):
            raise ValueError(f"Array of shape {coerced.shape} does not match num_features.")
        else:
            return coerced
    else:
        raise TypeError(
            f"Param of type {param.dtype} not understood. Supported types are float or Array."
        )
