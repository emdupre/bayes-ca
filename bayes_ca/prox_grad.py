from functools import partial
from typing import Callable, Union

import jax.numpy as jnp
from jax import jit, vmap
from jaxopt import ProximalGradient
from jax.flatten_util import ravel_pytree
from copt import minimize_proximal_gradient
from jaxtyping import Bool, Array, Float, Int

from bayes_ca import inference as core


def _debug_objective(
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Compute the function using the 2d params and all the hypers
    Note that we return the negative for each value, since we are maximizing rather
    than minimizing the function, as COPT is designed to do.
    """
    log_normalizer, _, _, _ = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu_pri, sigmasq_pri, sigmasq_subj
    )

    def _single_objective(subj_mean):
        return -0.5 / sigmasq_subj * jnp.sum((subj_mean - global_means) ** 2) - log_normalizer

    return log_normalizer, -1 * vmap(_single_objective)(subj_means).sum()


def _objective(
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Compute the function using the 2d params and all the hypers
    Note that we return the negative for each value, since we are maximizing rather
    than minimizing the function, as COPT is designed to do.
    """
    log_normalizer, _, _, _ = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu_pri, sigmasq_pri, sigmasq_subj
    )

    def _single_objective(subj_mean):
        return -0.5 / sigmasq_subj * jnp.sum((subj_mean - global_means) ** 2) - log_normalizer

    return -1 * vmap(_single_objective)(subj_means).sum()


def _grad_objective(
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Compute the gradient of the function using the 2d params and all the hypers
    Note that we return the negative for each value, since we are maximizing rather
    than minimizing the function, as COPT is designed to do.
    """
    _, _, _, expected_subj_means = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu_pri, sigmasq_pri, sigmasq_subj
    )

    def _single_grad(subj_mean):
        return 1 / sigmasq_subj * (subj_mean - expected_subj_means)

    return -1 * vmap(_single_grad)(subj_means).sum(axis=0)


def _fun_grad_objective(
    x: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Parameters
    ----------
    x : ndarray
        Initial estimate of global means, shape (T x N)
    subj_means: ndarray
        Estimated subject means of shape (S x T x N)
    mu_pri: float or ndarray
        Prior mean
    sigmasq_pri: float or ndarray
        Prior variance
    sigmasq_subj: float or ndarray
        Hierarchical variance
    hazard_rates: ndarray
        of shape (K,)

    Returns
    -------
    f : float
    grad : ndarray
    """
    log_normalizer, _, _, expected_subj_means = core.gaussian_cp_smoother(
        x, hazard_rates, mu_pri, sigmasq_pri, sigmasq_subj
    )

    def _single_fun_grad(subj_mean):
        f = -0.5 / sigmasq_subj * jnp.sum((subj_mean - x) ** 2) - log_normalizer
        g = 1 / sigmasq_subj * (subj_mean - expected_subj_means)
        return f, g

    fun, grad = vmap(_single_fun_grad)(subj_means)
    return -1 * fun.sum(), -1 * grad.sum(axis=0)


def _prox(
    x: Float[Array, "num_timesteps num_features"],
    hyperparams,
    stepsize: Float = 1.0,
):
    """
    Parameters
    ----------
    x : ndarray
        Initial estimate of global means, shape (T x N)
    hyperparams : tuple
    scale : float
        Ignored, required for compatibility with ProximalGradient
    """
    mu_pri, sigmasq_pri, hazard_rates = hyperparams
    _, num_features = x.shape
    _, x_next = core.gaussian_cp_posterior_mode(
        x, hazard_rates, mu_pri, sigmasq_pri, jnp.repeat(stepsize, num_features)
    )
    return x_next


@partial(jit, static_argnames=["unravel"])
def _flat_objective(
    _flat_global_means: Array,
    unravel: Callable,
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Flatten the objective to work with copt
    """
    global_means = unravel(_flat_global_means)
    return _objective(global_means, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)


@partial(jit, static_argnames=["unravel"])
def _flat_grad_objective(
    _flat_global_means,
    unravel,
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Flatten the gradient to work with copt
    """
    global_means = unravel(_flat_global_means)
    g = _grad_objective(global_means, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)
    return ravel_pytree(g)[0]


@partial(jit, static_argnames=["unravel"])
def _flat_prox(
    _flat_global_means,
    unravel,
    step_size: Float,
    mu_pri: Float,
    sigmasq_pri: Float,
    hazard_rates: Float[Array, "max_duration"],
):
    """
    Compute the proximal update by taking a step in the direction of the gradient
    and using the posterior mode to find the new global
    """
    global_means = unravel(_flat_global_means)
    _, num_features = global_means.shape
    x_next = core.gaussian_cp_posterior_mode(
        global_means, hazard_rates, mu_pri, sigmasq_pri, jnp.repeat(step_size, num_features)
    )[1]
    return ravel_pytree(x_next)[0]


def pgd(
    x0: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
    tol: Float = 1e-06,
    max_iter: Int = 500,
    accelerated: Bool = False,
    max_iter_backtracking: Int = 1000,
    backtracking_factor: Float = 0.6,
    trace_certificate: Bool = False,
):
    """
    Parameters
    ----------
    x0 : ndarray
        Initial estimate of global means, shape (T x N)
    subj_means: ndarray
        Estimated subject means of shape (S x T x N)
    mu_pri: float or ndarray
        Prior mean
    sigmasq_pri: float or ndarray
        Prior variance
    sigmasq_subj: float or ndarray
        Hierarchical variance
    hazard_rates: ndarray
        of shape (K,)
    tol: float, optional
        iteration stops when the gradient mapping is below this tolerance.
    max_iter: int, optional
         Maximum number of iterations
    accelerated : bool, optional
        Whether or not to use the accelerated variant of the Backtracking
        line search algorithm
    max_iter_backtracking: int, optional
        Maximum number of iterations for backtracking line search
    backtracking_factor: float, optional
        Value by which to shrink stepsize on each backtracking iteration
    trace_certificate : bool, optional
        Whether or not to track certificates over fitting, default False

    Returns
    -------
    res : ``scipy.optimize.OptimizeResult`` object.
        Important attributes are:
            ``x``: the solution array
            ``success``: a Boolean flag indicating if the optimizer exited successfully
            ``message``: describes the cause of the termination.
        See `scipy.optimize.OptimizeResult` for a description of other attributes.
    """
    # The optimization library COPT operates on parameter vectors.
    # Wrap the objective and gradient above to take take flattened vectors
    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)

    # Close over the global hyperparameters:
    f = lambda x: _flat_objective(
        x, unravel, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    g = lambda x: _flat_grad_objective(
        x, unravel, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    prox = lambda x, stepsize: _flat_prox(x, unravel, stepsize, mu_pri, sigmasq_pri, hazard_rates)

    results = minimize_proximal_gradient(
        fun=f,
        x0=x0_flat,
        prox=prox,
        jac=g,
        tol=tol,
        max_iter=max_iter,
        accelerated=accelerated,
        max_iter_backtracking=max_iter_backtracking,
        backtracking_factor=backtracking_factor,
        trace_certificate=trace_certificate,
    )
    # pack the output back into a PyTree
    init_lognorm, init_loss = _debug_objective(
        x0, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    lognorm, loss = _debug_objective(
        unravel(results["x"]), subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    results["x"] = unravel(results["x"])
    results["loss"] = loss
    results["init_loss"] = init_loss
    results["lognorm"] = lognorm
    results["init_lognorm"] = init_lognorm
    return results


def pgd_jaxopt(
    x0: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
    maxiter: Int = 500,
    maxls: Int = 15,
    tol: Float = 1e-06,
    acceleration: Bool = False,
    decrease_factor: Float = 0.5,
    jit: Union[str, Bool] = "auto",
):
    """
    jaxopt.ProximalGradient requires two functions:

    - a smooth function of the form fun(x, *args, **kwargs), returning
      scalar value and gradient
    - a proximity operator associated with the non_smooth part of the
      function. It should be of the form prox(params, hyperparams_prox, scale=1.0).
      See jaxopt.prox for examples.

    Parameters
    ----------
    x0 : ndarray
        Initial estimate of global means, shape (T x N)
    subj_means: ndarray
        Estimated subject means of shape (S x T x N)
    mu_pri: float or ndarray
        Prior mean
    sigmasq_pri: float or ndarray
        Prior variance
    sigmasq_subj: float or ndarray
        Hierarchical variance
    hazard_rates: ndarray
        of shape (K,)
    max_iter: int, optional
         Maximum number of iterations
    maxls: int, optional
        Maximum number of iterations for backtracking line search
    tol: float, optional
        iteration stops when the gradient mapping is below this tolerance.
    acceleration : bool, optional
        Whether or not to use the accelerated variant of the Backtracking
        line search algorithm (FISTA)
    decrease_factor: float, optional
        Value by which to shrink stepsize on each backtracking iteration
    jit : bool, optional
        Whether or not to jit-compile the optimization loop (default: “auto”).

    Returns
    -------
    OptStep
    """

    pg = ProximalGradient(
        fun=_fun_grad_objective,
        prox=_prox,
        value_and_grad=True,
        stepsize=-1,
        maxiter=maxiter,
        maxls=maxls,
        tol=tol,
        acceleration=acceleration,
        decrease_factor=decrease_factor,
        jit=jit,
    )

    # import jax
    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    result = pg.run(
        x0,  # init params
        (mu_pri, sigmasq_pri, hazard_rates),  # hyperparams_prox
        subj_means,  # begin args passed to fun
        mu_pri,
        sigmasq_pri,
        sigmasq_subj,
        hazard_rates,
    )
    return result
