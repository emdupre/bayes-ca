import jax.numpy as jnp
from jax import jit, vmap
from jaxtyping import Array, Float
from jax.flatten_util import ravel_pytree
from copt import minimize_proximal_gradient

from bayes_ca import inference as core


def pgd(
    x0: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    mu_pri: Float,
    sigmasq_pri: Float,
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
    tol=1e-06,
    max_iter=500,
    accelerated=False,
    max_iter_backtracking=1000,
    backtracking_factor=0.6,
    trace_certificate=False,
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

    # Compute the function and its gradient, using the 2d params and all the hypers
    # Note that we return the negative for each value, since we are maximizing rather
    # than minimizing the function, as COPT is designed to do.
    def _objective(
        global_means: Float[Array, "num_timesteps num_features"],
        subj_means: Float[Array, "num_subjects num_timesteps num_features"],
        mu0: Float,
        sigmasq0: Float,
        sigmasq_subj: Float,
        hazard_rates: Float[Array, "max_duration"],
    ):
        """
        Compute the log probability as a function of the global means
        """
        log_normalizer, _, _, _ = core.gaussian_cp_smoother(
            global_means, hazard_rates, mu0, sigmasq0, sigmasq_subj
        )

        def _single_objective(subj_mean):
            return -0.5 / sigmasq_subj * jnp.sum((subj_mean - global_means) ** 2) - log_normalizer

        return -1 * vmap(_single_objective)(subj_means).sum()

    def _grad_objective(
        global_means: Float[Array, "num_timesteps num_features"],
        subj_means: Float[Array, "num_subjects num_timesteps num_features"],
        mu0: Float,
        sigmasq0: Float,
        sigmasq_subj: Float,
        hazard_rates: Float[Array, "max_duration"],
    ):
        _, _, _, expected_subj_means = core.gaussian_cp_smoother(
            global_means, hazard_rates, mu0, sigmasq0, sigmasq_subj
        )

        def _single_grad(subj_mean):
            return 1 / sigmasq_subj * (subj_mean - expected_subj_means)

        return -1 * vmap(_single_grad)(subj_means).sum(axis=0)

    # Flatten the objective and gradient to work with copt
    def _flat_objective(
        _flat_global_means,
        subj_means: Float[Array, "num_subjects num_timesteps num_features"],
        mu0: Float,
        sigmasq0: Float,
        sigmasq_subj: Float,
        hazard_rates: Float[Array, "max_duration"],
    ):

        global_means = unravel(_flat_global_means)
        return _objective(global_means, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates)

    def _flat_grad_objective(
        _flat_global_means,
        subj_means: Float[Array, "num_subjects num_timesteps num_features"],
        mu0: Float,
        sigmasq0: Float,
        sigmasq_subj: Float,
        hazard_rates: Float[Array, "max_duration"],
    ):

        global_means = unravel(_flat_global_means)
        g = _grad_objective(global_means, subj_means, mu0, sigmasq0, sigmasq_subj, hazard_rates)
        return ravel_pytree(g)[0]

    # Write a flat prox operator
    def _flat_prox(
        _flat_global_means,
        step_size: Float,
        mu0: Float,
        sigmasq0: Float,
        hazard_rates: Float[Array, "max_duration"],
    ):
        """
        Compute the proximal update by taking a step in the direction of the gradient
        and using the posterior mode to find the new global
        """
        global_means = unravel(_flat_global_means)
        x_next = core.gaussian_cp_posterior_mode(
            global_means, hazard_rates, mu0, sigmasq0, step_size
        )[1]
        return ravel_pytree(x_next)[0]

    # Finally, close over the global hyperparameters
    f = lambda x: _flat_objective(x, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)
    g = lambda x: _flat_grad_objective(
        x, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    prox = lambda x, stepsize: _flat_prox(x, stepsize, mu_pri, sigmasq_pri, hazard_rates)

    # The optimization library COPT operates on parameter vectors.
    # Wrap the objective and gradient above to take take flattened vectors
    # Use tree flatten and unflatten to convert params x0 from PyTrees to flat arrays
    x0_flat, unravel = ravel_pytree(x0)

    results = minimize_proximal_gradient(
        fun=jit(f),
        x0=x0_flat,
        prox=jit(prox),
        jac=jit(g),
        tol=tol,
        max_iter=max_iter,
        accelerated=accelerated,
        max_iter_backtracking=max_iter_backtracking,
        backtracking_factor=backtracking_factor,
        trace_certificate=trace_certificate,
    )
    # pack the output back into a PyTree
    results["x"] = unravel(results["x"])
    return results
