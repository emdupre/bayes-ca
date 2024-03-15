import warnings
from typing import Optional, Tuple

from jax import vmap
import jax.numpy as jnp
from jax.lax import while_loop
from jaxtyping import Array, Float, Int

from bayes_ca import inference as core


def _calculate_f_g(
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu0: Float,
    sigmasq0: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Use exponential family magic to compute gradient of the
    smooth part of the objective (not including the CP prior)/
    """
    log_normalizer, _, _, expected_subj_means = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu0, sigmasq0, sigmasq_subj
    )
    f = -0.5 / sigmasq_subj * jnp.sum((subj_means - global_means) ** 2) - log_normalizer
    g = 1 / sigmasq_subj * (subj_means - expected_subj_means)
    return f, g


def _prox(
    x: Float[Array, "num_timesteps num_features"],
    mu0: Float,
    sigmasq0: Float,
    hazard_rates: Float[Array, "max_duration"],
    step_size: Float,
):
    """
    Compute the proximal update by taking a step in the direction of the gradient
    and using the posterior mode to find the new global states/
    """
    x_next = core.gaussian_cp_posterior_mode(x, hazard_rates, mu0, sigmasq0, step_size)[1]
    return x_next


def nesterov_acceleration(
    step_size: Float,
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu0: Float,
    sigmasq0: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
    tol: Optional[Float] = 1e-6,
    max_iter: Optional[Int] = 500,
    max_iter_backtracking: Optional[Int] = 1000,
    backtracking_factor: Optional[Float] = 0.6,
) -> Tuple[Array, Array]:
    """
    Parameters
    ----------
    global_means: ndarray
        Estimated global means of shape (T x N)
    subj_means: ndarray
        Estaimted subject means of shape (S x T x N)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    sigmasq_subj: float or ndarray
        Hierarchical variance
    hazard_rates: ndarray
        of shape (K,)
    tol: float, optional
        iteration stops when the gradient mapping is below this tolerance.
    max_iter: int, optional
         Maximum number of iterations
    max_iter_backtracking: int, optional
        Maximum number of iterations for backtracking line search
    backtracking_factor: float, optional
        Value by which to shrink stepsize on each backtracking iteration

    Returns
    -------
    new_global_means : ndarray
        Updated estimate of global means, shape (T x N)
    g : ndarray
        Estimated gradient, shape (T x N)
    """
    # _, num_features = global_means.shape

    # Nesterov acceleration for effective emissions
    # adapted from openopt/copt/proximal_gradient.py
    n_iterations = 0
    tk = 1
    yk = global_means

    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    while True:
        f, g = vmap(_calculate_f_g, in_axes=(None, 0, None, None, None, None))(
            yk, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates
        )
        current_step_size = step_size
        x_next = _prox(yk + current_step_size * g, mu0, sigmasq0, hazard_rates, step_size)
        for _ in range(max_iter_backtracking):
            update_direction = x_next - yk
            f_next, g_next = vmap(_calculate_f_g, in_axes=(None, 0, None, None, None, None))(
                x_next, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates
            )
            if f_next <= f + jnp.sum(g * update_direction) + jnp.sum(
                update_direction * update_direction
            ) / (2.0 * current_step_size):
                # .. step size found ..
                break
            else:
                # .. backtracking, reduce step size ..
                current_step_size *= backtracking_factor
                x_next = _prox(yk + current_step_size * g, mu0, sigmasq0, hazard_rates, step_size)
        else:
            warnings.warn("Maxium number of line-search iterations reached")
        t_next = (1 + jnp.sqrt(1 + 4 * tk * tk)) / 2
        yk = x_next + ((tk - 1.0) / t_next) * (x_next - global_means)

        x_prox = _prox(
            x_next + current_step_size * g_next, mu0, sigmasq0, hazard_rates, step_size
        )
        certificate = jnp.linalg.norm((global_means - x_prox) / current_step_size)

        tk = t_next
        global_means = x_next

        if certificate < tol:
            break

        if n_iterations >= max_iter:
            break
        else:
            n_iterations += 1

    if n_iterations >= max_iter:
        warnings.warn(
            "did not reach desired tolerance level",
            RuntimeWarning,
        )
    pass


def line_search(
    step_size: Float,
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu0: Float,
    sigmasq0: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
    tol: Optional[Float] = 1e-6,
    max_iter: Optional[Int] = 500,
    max_iter_backtracking: Optional[Int] = 1000,
    backtracking_factor: Optional[Float] = 0.6,
) -> Tuple[Array, Array]:
    """
    Parameters
    ----------
    global_means: ndarray
        Estimated global means of shape (T x N)
    subj_means: ndarray
        Estaimted subject means of shape (S x T x N)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    sigmasq_subj: float or ndarray
        Hierarchical variance
    hazard_rates: ndarray
        of shape (K,)
    tol: float, optional
        iteration stops when the gradient mapping is below this tolerance.
    max_iter: int, optional
         Maximum number of iterations
    max_iter_backtracking: int, optional
        Maximum number of iterations for backtracking line search
    backtracking_factor: float, optional
        Value by which to shrink stepsize on each backtracking iteration

    Returns
    -------
    new_global_means : ndarray
        Updated estimate of global means, shape (T x N)
    g : ndarray
        Estimated gradient, shape (T x N)
    """
    # _, num_features = global_means.shape

    # backtracking line search for effective emissions
    # adapted from openopt/copt/proximal_gradient.py
    n_iterations = 0
    # .. compute gradient and step size
    f, g = vmap(_calculate_f_g, in_axes=(None, 0, None, None, None, None))(
        global_means, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates
    )
    f = f.sum(axis=0)
    g = g.sum(axis=0)

    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    while True:
        global_means_next = _prox(
            global_means + step_size * g, mu0, sigmasq0, hazard_rates, step_size
        )
        update_direction = global_means_next - global_means
        step_size *= 1.1

        for _ in range(max_iter_backtracking):
            f_next, g_next = vmap(_calculate_f_g, in_axes=(None, 0, None, None, None, None))(
                global_means_next, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates
            )
            f_next = f_next.sum(axis=0)
            rhs = (
                f
                + jnp.sum(g * update_direction)
                + jnp.sum(update_direction * update_direction) / (2.0 * step_size)
            )
            if f_next <= rhs:
                # .. step size found ..
                break
            else:
                # .. backtracking, reduce step size ..
                step_size *= backtracking_factor
                global_means_next = _prox(
                    global_means + step_size * g, mu0, sigmasq0, hazard_rates, step_size
                )
                update_direction = global_means_next - global_means
        else:
            warnings.warn("Maxium number of line-search iterations reached")

        certificate = jnp.linalg.norm((global_means - global_means_next) / step_size)
        global_means = global_means_next
        g = g_next.sum(axis=0)

        if certificate < tol:
            break

        if n_iterations >= max_iter:
            break
        else:
            n_iterations += 1  # increase iter count and repeat

    if n_iterations >= max_iter:
        warnings.warn(
            "Did not reach desired tolerance level",
            RuntimeWarning,
        )

    return global_means, g
