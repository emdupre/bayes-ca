import warnings
from typing import Optional, Tuple

import jax
from scipy import sparse
from jax import vmap
import jax.numpy as jnp
from jax.lax import while_loop
from jaxtyping import Array, Float, Int

from bayes_ca import inference as core


def init_lipschitz(f_grad, x0):
    """
    Lipschitz constant initialization for step size,
    adapted from openopt/copt/utils.py
    """
    L0 = 1e-3
    f0, grad0 = f_grad(x0)
    if sparse.issparse(grad0) and not sparse.issparse(x0):
        x0 = sparse.csc_matrix(x0).T

    elif sparse.issparse(x0) and not sparse.issparse(grad0):
        grad0 = sparse.csc_matrix(grad0).T

    x_tilde = x0 - (1.0 / L0) * grad0
    f_tilde = f_grad(x_tilde)[0]

    for _ in range(100):
        if f_tilde <= f0:
            break
        L0 *= 10
        x_tilde = x0 - (1.0 / L0) * grad0
        f_tilde = f_grad(x_tilde)[0]

    return L0


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
    Nesterov acceleration for effective emissions
    adapted from openopt/copt/proximal_gradient.py

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

    def _step_cond(state):
        itr, certificate, _ = state
        return (itr < max_iter) & (certificate > tol)

    def _step_body(state):
        itr, certificate, params = state
        step_size, tk, yk, global_means, _ = params

        f, g = vmap(_calculate_f_g, in_axes=(None, 0, None, None, None, None))(
            yk, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates
        )
        # TODO : Can these sums be done more elegantly ?
        f = -1 * f.sum()
        g = -1 * g.sum(axis=0)

        current_step_size = step_size
        x_next = _prox(yk - current_step_size * g, mu0, sigmasq0, hazard_rates, step_size)

        def _backtrack_cond(backtrack_state):
            """ """
            backtrack_itr, current_step_size, _back_params = backtrack_state
            update_direction, f_next, _ = _back_params

            lower_bound = (
                f
                + jnp.sum(g * update_direction)
                + jnp.sum(update_direction * update_direction) / (2.0 * current_step_size)
            )

            return (f_next > lower_bound) & (backtrack_itr <= max_iter_backtracking)

        def _backtrack_body(backtrack_state):
            """ """
            backtrack_itr, current_step_size, _ = backtrack_state
            # .. backtracking, reduce step size ..
            current_step_size *= backtracking_factor

            update_direction = x_next - yk
            f_next, g_next = vmap(_calculate_f_g, in_axes=(None, 0, None, None, None, None))(
                x_next, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates
            )

            # TODO : Can these sums be done more elegantly ?
            f_next = -1 * f_next.sum()
            g_next = -1 * g_next.sum(axis=0)

            _back_params = (update_direction, f_next, g_next)

            return backtrack_itr + 1, current_step_size, _back_params

        init_vals = (0, step_size, (x_next, 0, x_next))
        _, step_size, _back_params = while_loop(_backtrack_cond, _backtrack_body, init_vals)
        (_, _, g_next) = _back_params

        t_next = (1 + jnp.sqrt(1 + 4 * tk * tk)) / 2
        yk = x_next + ((tk - 1.0) / t_next) * (x_next - global_means)

        x_prox = _prox(x_next - step_size * g_next, mu0, sigmasq0, hazard_rates, step_size)
        certificate = jnp.linalg.norm((global_means - x_prox) / step_size)

        # itr, certificate, (tk, yk, global_means, g)
        return itr + 1, certificate, (step_size, t_next, yk, x_next, g)

    init_vals = (
        0,
        jnp.inf,
        (step_size, 1, global_means, global_means, global_means),
    )
    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    itr, _, params = while_loop(_step_cond, _step_body, init_vals)
    (_, _, _, global_means, g) = params

    if itr > max_iter:
        warnings.warn(
            "Did not reach desired tolerance level",
            RuntimeWarning,
        )
    return global_means, g


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
    backtracking line search for effective emissions
    adapted from openopt/copt/proximal_gradient.py

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
