import warnings
from typing import Optional, Tuple, Union

from jax import vmap
import jax.numpy as jnp
import jax.random as jr
from jax.lax import conv, scan
from dynamax.types import PRNGKey
from jaxtyping import Array, Bool, Float, Int
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


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


def _normalize(
    u: Float[Array, "num_timesteps max_duration"],
    axis: Optional[Int] = 0,
    eps: Optional[Float] = 1e-15,
):
    """Normalizes the values within the axis in a way that they sum up to 1.

    Args:
        u: Input array to normalize.
        axis: Axis over which to normalize.
        eps: Minimum value threshold for numerical stability.

    Returns:
        Tuple of the normalized values, and the normalizing denominator.
    """
    u = jnp.where(u == 0, 0, jnp.where(u < eps, eps, u))
    c = u.sum(axis=axis)
    c = jnp.where(c == 0, 1, c)
    return u / c, c


def _condition_on(
    probs: Float[Array, "max_duration"], ll: Float[Array, "num_timesteps max_duration"]
):
    """Condition on new emissions, given in the form of log likelihoods
    for each discrete state, while avoiding numerical underflow.

    Args:
        probs(k): prior for state k
        ll(k): log likelihood for state k

    Returns:
        probs(k): posterior for state k
    """
    ll_max = ll.max()
    new_probs = probs * jnp.exp(ll - ll_max)
    new_probs, norm = _normalize(new_probs)
    log_norm = jnp.log(norm) + ll_max
    return new_probs, log_norm


def _predict(
    probs: Float[Array, "max_duration"],
    hazard_rates: Float[Array, "max_duration"],
):
    return jnp.concatenate(
        [jnp.array([probs @ hazard_rates]), (1 - hazard_rates[:-1]) * probs[:-1]]
    )


def _backward_predict(
    probs: Float[Array, "max_duration"],
    hazard_rates: Float[Array, "max_duration"],
):
    p = hazard_rates * probs[0]
    p = p.at[:-1].add((1 - hazard_rates[:-1]) * probs[1:])
    return p


###
# Generic functions for changepoint inference (over run lengths)
###
def cp_filter(
    hazard_rates: Float[Array, "max_duration"],
    pred_log_likes: Float[Array, "num_timesteps max_duration"],
) -> Tuple[Array, Array, Array]:
    """
    hazard_rates: (K+1,)
    pred_log_probs: (T, K+1)
    """
    K = pred_log_likes.shape[1] - 1

    def _step(carry, ll):
        log_normalizer, predicted_probs = carry

        filtered_probs, log_norm = _condition_on(predicted_probs, ll)
        log_normalizer += log_norm
        predicted_probs_next = _predict(filtered_probs, hazard_rates)

        return (log_normalizer, predicted_probs_next), (filtered_probs, predicted_probs)

    initial_dist = jnp.zeros(K + 1)
    initial_dist = initial_dist.at[0].set(1.0)
    carry = (0.0, initial_dist)
    (log_normalizer, _), (filtered_probs, predicted_probs) = scan(_step, carry, pred_log_likes)

    return log_normalizer, filtered_probs, predicted_probs


def cp_backward_filter(
    hazard_rates: Float[Array, "max_duration"],
    pred_log_likes: Float[Array, "num_timesteps max_duration"],
) -> Float[Array, "num_timesteps"]:
    """ """
    K = pred_log_likes.shape[1] - 1

    def _step(carry, ll):
        log_normalizer, backward_pred_probs = carry

        # Condition on emission at time t, being careful not to overflow.
        backward_filt_probs, log_norm = _condition_on(backward_pred_probs, ll)
        # Update the log normalizer.
        log_normalizer += log_norm
        # Predict the next state (going backward in time).
        next_backward_pred_probs = _backward_predict(backward_filt_probs, hazard_rates)
        return (log_normalizer, next_backward_pred_probs), backward_pred_probs

    carry = (0.0, jnp.ones(K + 1))
    _, backward_pred_probs = scan(_step, carry, pred_log_likes, reverse=True)
    return backward_pred_probs


def cp_smoother(
    hazard_rates: Float[Array, "max_duration"],
    pred_log_likes: Float[Array, "num_timesteps max_duration"],
) -> Tuple[Array, Array, Array]:
    """ """
    log_normalizer, filtered_probs, predicted_probs = cp_filter(hazard_rates, pred_log_likes)
    backward_pred_probs = cp_backward_filter(hazard_rates, pred_log_likes)

    # Compute smoothed probabilities
    smoothed_probs = filtered_probs * backward_pred_probs
    norm = smoothed_probs.sum(axis=1, keepdims=True)
    smoothed_probs /= norm

    # Compute transition probabilities up to time T-1
    transition_probs = jnp.einsum(
        "tk, k, t->tk",
        filtered_probs[:-1],
        hazard_rates,
        smoothed_probs[1:, 0] / (predicted_probs[1:, 0] + 1e-15),
    )

    # Compute last transition probs
    transition_probs = jnp.vstack([transition_probs, filtered_probs[-1]])

    return log_normalizer, smoothed_probs, transition_probs


def cp_posterior_sample(
    key: PRNGKey,
    hazard_rates: Float[Array, "max_duration"],
    pred_log_likes: Float[Array, "num_timesteps max_duration"],
) -> Int[Array, "num_timesteps"]:
    r"""Sample a latent sequence from the posterior.

    Args:
        key: random number generator
        hazard_rates: $p(z_{t+1} = 0 \mid z_t)$
        pred_log_likes: $p(x_t \mid x_{1:t-1}, z_t)$

    Returns:
        :sample of the latent states, $z_{1:T}$

    """
    num_timesteps, max_duration = pred_log_likes.shape

    # Run the CP filter
    _, filtered_probs, _ = cp_filter(hazard_rates, pred_log_likes)

    def _step(carry, args):
        run_length = carry
        _key, filtered_probs = args

        # calculate smoothed probabilities and renormalize
        smoothed_probs = filtered_probs * hazard_rates
        smoothed_probs /= smoothed_probs.sum()

        # Sample current run_length
        # note: if z_{t+1} > 0, then z_t must equal (z_{t+1} - 1)
        #       only uncertainty is when z_{t+1} = 0.
        run_length = jnp.where(
            run_length > 0, run_length - 1, jr.choice(_key, a=max_duration, p=smoothed_probs)
        )

        return run_length, run_length

    # Run the HMM smoother
    rngs = jr.split(key, num_timesteps)
    last_length = jr.choice(rngs[-1], a=max_duration, p=filtered_probs[-1])
    args = (rngs[:-1], filtered_probs[:-1])
    _, run_lengths = scan(_step, last_length, args, reverse=True)

    # Stack and return the run lengths
    return jnp.append(run_lengths, last_length)


def cp_posterior_mode(
    hazard_rates: Float[Array, "max_duration"],
    pred_log_likes: Float[Array, "num_timesteps max_duration"],
) -> Int[Array, "num_timesteps"]:
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        hazard_rates: $p(z_{t+1} = j \mid z_t = i, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.

    Returns:
        most likely state sequence
    """
    _, max_duration = pred_log_likes.shape

    # Note: we could allow more general initial distributions for left-censoring
    initial_dist = jnp.zeros(max_duration, dtype=float)
    initial_dist = initial_dist.at[0].set(1.0)

    # Run the backward pass
    def _backward_pass(best_next_score, next_log_likes):
        X = best_next_score + next_log_likes
        scores = jnp.stack(
            [jnp.log(hazard_rates[:-1]) + X[0], jnp.log(1 - hazard_rates[:-1]) + X[1:]]
        )

        best_idx = jnp.argmax(scores, axis=0)
        best_next_state = jnp.where(best_idx == 0, 0, jnp.arange(1, max_duration))
        best_next_score = jnp.max(scores, axis=0)
        return jnp.append(best_next_score, X[0]), jnp.append(best_next_state, 0)

    best_second_score, best_next_states = scan(
        _backward_pass, jnp.zeros(max_duration), pred_log_likes[1:], reverse=True
    )

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(jnp.log(initial_dist) + pred_log_likes[0] + best_second_score)
    _, states = scan(_forward_pass, first_state, best_next_states)

    return jnp.insert(states, 0, first_state)


###
# The functions below are for the special case of a changepoint model with
# scalar Gaussian emissions.
###
def _compute_gaussian_stats(
    emissions: Float[Array, "num_timesteps num_features"], max_duration: Int
) -> Tuple[Array, Array]:
    r"""
    Compute conditional stats of mu_t given runs of varying length ending at time t

    Parameters
    ----------
    emissions: ndarray
        Observed emissions of shape (T x N)
    max_duration: int

    Returns
    -------
    partial_sums: ndarray
        (T, K+1) array x[t-k:t+1].sum() for all t and k
    partial_counts: ndarray
        (T, K+1) array number of data points contributing to the sum

    """

    def _step(carry, x):
        """
        means: (K + 1,) array of means
        """
        sums, counts = carry

        new_sums = jnp.roll(sums, 1)
        new_sums = new_sums.at[0].set(0.0)
        new_sums += x

        new_counts = jnp.roll(counts, 1)
        new_counts = new_counts.at[0].set(0)
        new_counts += 1

        return (new_sums, new_counts), (new_sums, new_counts)

    initial_carry = (jnp.zeros(max_duration), jnp.zeros(max_duration))
    _, (partial_sums, partial_counts) = scan(_step, initial_carry, emissions)
    return partial_sums, partial_counts


# Test that each row = (0, 1, ..., K) except in the top right corner of the matrix


def _compute_gaussian_lls(
    emissions: Float[Array, "num_timesteps num_features"],
    max_duration: Int,
    mu0: Float,
    sigmasq0: Float,
    sigmasq: Float,
) -> Array:
    r"""
    Compute the one-step ahead predictive log likelihoods

        \log p(x[t+1] | x[t-K:t])

    integrating over the latent mean \mu[t] for that run.

    Parameters
    ----------
    emissions: ndarray
        Observed emissions of shape (T x N)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    sigmasq: float or ndarray
        Hierarchical or observation variance

    Returns
    -------
    lls: ndarray
        Computed log-likelihoods of the emissions under the posterior
        predictive distribution
    """

    # Compute the sufficient statistics of the predictive distribution
    def _step(carry, x):
        """
        sums: (K,) array of sums
            sums: (T, K+1) array of sums x_{t-k:t-1} for k=0,...,K
            counts: (T, K+1) number of terms contributing to each sum
        """
        sums, counts = carry

        new_sums = jnp.roll(sums, 1)
        new_sums += x
        new_sums = new_sums.at[0].set(0.0)

        new_counts = jnp.roll(counts, 1)
        new_counts += 1
        new_counts = new_counts.at[0].set(0)
        return (new_sums, new_counts), (sums, counts)

    initial_carry = (
        jnp.zeros(max_duration, dtype=float),
        jnp.zeros(max_duration, dtype=int),
    )
    _, (pred_sums, pred_counts) = scan(_step, initial_carry, emissions)

    # Compute the posterior predictive distribution and return the log prob
    sigmasq_post = 1 / (1 / sigmasq0 + pred_counts / sigmasq)
    mu_post = sigmasq_post * (mu0 / sigmasq0 + pred_sums / sigmasq)
    sigmasq_pred = sigmasq_post + sigmasq
    return tfd.Normal(mu_post, jnp.sqrt(sigmasq_pred)).log_prob(emissions[:, None])


def gaussian_cp_posterior_sample(
    key: PRNGKey,
    emissions: Float[Array, "num_timesteps num_features"],
    hazard_rates: Float[Array, "max_duration"],
    mu0: Float,
    sigmasq0: Float,
    sigmasq: Float,
) -> Tuple[Array, Array]:
    r"""
    Return a sample of the run lengths and the latent means given emissions.

    Parameters
    ----------
    key: jr.PRNGKey
        Random key for sampling
    emissions: ndarray
        Observed emissions of shape (T x N)
    hazard_rates: ndarray
        of shape (K,)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    sigmasq: float or ndarray
        Hierarchical or observation variance

    Returns
    -------
    zs:
    mus:

    #TODO: document this
    """
    max_duration = hazard_rates.shape[0]
    num_timesteps, num_features = emissions.shape

    # First sample the run lengths
    k1, k2 = jr.split(key)
    if num_features > 1:
        lls = vmap(_compute_gaussian_lls, in_axes=(-1, None, 0, 0, 0))(
            emissions, max_duration, mu0, sigmasq0, sigmasq
        )
        lls = lls.sum(axis=0)
    else:
        lls = _compute_gaussian_lls(emissions.squeeze(), max_duration, mu0, sigmasq0, sigmasq)
    zs = cp_posterior_sample(k1, hazard_rates, lls)

    # Then sample the means
    def _backward_sample(mu_next, args):
        key, x_sum, n, z, z_next = args
        sigmasq_post = 1 / (1 / sigmasq0 + n[z] / sigmasq)
        mu_post = sigmasq_post * (mu0 / sigmasq0 + x_sum[z] / sigmasq)

        mu = jnp.where(
            z_next == 0, tfd.Normal(mu_post, jnp.sqrt(sigmasq_post)).sample(seed=key), mu_next
        )
        return mu, mu

    partial_sums, partial_counts = vmap(_compute_gaussian_stats, in_axes=(-1, None), out_axes=-1)(
        emissions, max_duration
    )
    args = (jr.split(k2, num_timesteps), partial_sums, partial_counts, zs, jnp.append(zs[1:], 0))
    _, mus = scan(_backward_sample, jnp.repeat(jnp.nan, num_features), args, reverse=True)
    return zs, mus


def gaussian_cp_posterior_mode(
    emissions: Float[Array, "num_timesteps num_features"],
    hazard_rates: Float[Array, "max_duration"],
    mu0: Float,
    sigmasq0: Float,
    sigmasq: Float,
) -> Tuple[Array, Array]:
    """
    Parameters
    ----------
    emissions: ndarray
        Observed emissions of shape (T x N)
    hazard_rates: ndarray
        of shape (K,)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    sigmasq: float or ndarray
        Hierarchical or observation variance

    Returns
    -------
    zs:
    mus:
    """
    max_duration = hazard_rates.shape[0]
    _, num_features = emissions.shape

    # First compute the most likely run lengths)
    if num_features > 1:
        lls = vmap(_compute_gaussian_lls, in_axes=(-1, None, 0, 0, 0))(
            emissions, max_duration, mu0[:, None], sigmasq0[:, None], sigmasq[:, None]
        )
        lls = lls.sum(axis=0)
    else:
        lls = _compute_gaussian_lls(emissions.squeeze(), max_duration, mu0, sigmasq0, sigmasq)
    zs = cp_posterior_mode(hazard_rates, lls)

    # Now compute the most likely mus
    def _backward_pass(mu_next, args):
        x_sum, n, z, z_next = args
        sigmasq_post = 1 / (1 / sigmasq0 + n[z] / sigmasq)
        mu_post = sigmasq_post * (mu0 / sigmasq0 + x_sum[z] / sigmasq)

        mu = jnp.where(z_next == 0, mu_post, mu_next)
        return mu, mu

    partial_sums, partial_counts = vmap(_compute_gaussian_stats, in_axes=(-1, None), out_axes=-1)(
        emissions, max_duration
    )
    args = (partial_sums, partial_counts, zs, jnp.append(zs[1:], 0))
    _, mus = scan(_backward_pass, jnp.repeat(jnp.nan, num_features), args, reverse=True)
    return zs, mus


def gaussian_cp_smoother(
    emissions: Float[Array, "num_timesteps num_features"],
    hazard_rates: Float[Array, "max_duration"],
    mu0: Float,
    sigmasq0: Float,
    sigmasq: Float,
) -> Tuple[Array, Array, Array, Array]:
    """
    Parameters
    ----------
    emissions: ndarray
        Observed emissions of shape (T x N)
    hazard_rates: ndarray
        of shape (K,)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    sigmasq: float or ndarray
        Hierarchical or observation variance

    Returns
    -------
    log_normalizer:
    smoothed_probs:
    transition_probs:
    smoothed_means:
    """
    max_duration = hazard_rates.shape[0]
    num_states = max_duration - 1
    _, num_features = emissions.shape

    # First compute the most likely run lengths)
    if num_features > 1:
        lls = vmap(_compute_gaussian_lls, in_axes=(-1, None, 0, 0, 0))(
            emissions, max_duration, mu0, sigmasq0, sigmasq
        )
        lls = lls.sum(axis=0)
    else:
        lls = _compute_gaussian_lls(emissions.squeeze(), max_duration, mu0, sigmasq0, sigmasq)
    log_normalizer, smoothed_probs, transition_probs = cp_smoother(hazard_rates, lls)

    # Compute posterior distribution of latent mean for each time and run length
    partial_sums, partial_counts = vmap(_compute_gaussian_stats, in_axes=(-1, None), out_axes=-1)(
        emissions, max_duration
    )
    sigmasq_post = 1 / (1 / sigmasq0 + partial_counts / sigmasq)
    mu_post = sigmasq_post * (mu0 / sigmasq0 + partial_sums / sigmasq)

    def _smooth_single(m):
        # Sum over possible run lengths to compute smoothed means
        inpt = (transition_probs * m).T  # KT
        inpt = inpt[None, None, :, :]  # NCKT
        kernel = jnp.tril(jnp.ones((max_duration, max_duration)))[None, None, :, :]  # OIKK
        smoothed_means = conv(inpt, kernel, (1, 1), [(0, 0), (0, num_states)])[0, 0, 0]  # T
        return smoothed_means

    smoothed_means = vmap(_smooth_single, in_axes=-1, out_axes=-1)(mu_post)  # we want T x N
    return log_normalizer, smoothed_probs, transition_probs, smoothed_means


def sample_gaussian_cp_model(
    key: PRNGKey,
    num_timesteps: Int,
    hazard_rates: Float[Array, "max_duration"],
    mu0: Float,
    sigmasq0: Float,
) -> Tuple[Array, Array]:
    """
    Parameters
    ----------
    key: jr.PRNGKey
        Random key for sampling
    num_timesteps: int
        Number of timesteps, T
    num_features: int
        Number of features, N
    hazard_rates: ndarray
        of shape (K,)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance

    Returns
    -------
    zs:
    mus:
    """

    def _step(carry, key):
        z, mu = carry
        k1, k2 = jr.split(key)

        z_next = jnp.where(jr.bernoulli(k1, hazard_rates[z]), 0, z + 1)
        mu_next = jnp.where(z_next == 0, mu0 + jnp.sqrt(sigmasq0) * jr.normal(k2), mu)

        return (z_next, mu_next), (z, mu)

    # Sample the first timestep
    k1, key = jr.split(key)
    initial_carry = (0, mu0 + jnp.sqrt(sigmasq0) * jr.normal(k1))

    # Run the scan
    _, (zs, mus) = scan(_step, initial_carry, jr.split(key, num_timesteps))

    return zs, mus


def backtracking_lsearch(
    step_size: Float,
    global_means: Float[Array, "num_timesteps num_features"],
    subj_means: Float[Array, "num_subjects num_timesteps num_features"],
    sigmasq_subj: Float,
    hazard_rates: Float[Array, "max_duration"],
    mu0: Float,
    sigmasq0: Float,
    tol: Optional[Float] = 1e-6,
    max_iter: Optional[Int] = 500,
    max_iter_backtracking: Optional[Int] = 1000,
    backtracking_factor: Optional[Float] = 0.6,
) -> Tuple[Array, Array]:
    r"""
    Parameters
    ----------
    global_means: ndarray
        Estimated global means of shape (T x N)
    subj_means: ndarray
        Estaimted subject means of shape (S x T x N)
    sigmasq_subj: float or ndarray
        Hierarchical variance
    hazard_rates: ndarray
        of shape (K,)
    mu0: float or ndarray
        Prior mean
    sigmasq0: float or ndarray
        Prior variance
    tol: float, optoinal
        iteration stops when the gradient mapping is below this tolerance.
    max_iter: int, optional
         Maximum number of iterations
    accelerated: Boolean
        Whether to use the accelerated variant of the algorithim
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

    # Use exponential family magic to compute gradient of the
    # smooth part of the objective (not including the CP prior)
    _, _, _, expected_subj_means = gaussian_cp_smoother(
        global_means, hazard_rates, mu0, sigmasq0, sigmasq_subj
    )
    g = 1 / sigmasq_subj * jnp.sum(subj_means - expected_subj_means, axis=0)  # sum over subjects

    # backtracking line search for effective emissions
    # adapted from openopt/copt/proximal_gradient.py
    n_iterations = 0

    expected_subj_means_ = expected_subj_means
    global_means_ = global_means

    # .. a while loop instead of a for loop ..
    # .. allows for infinite or floating point max_iter ..
    while True:
        # .. compute gradient and step size
        effective_emissions_ = global_means_ - step_size * g
        update_direction = effective_emissions_ - global_means_
        step_size *= 1.1

        # Compute the proximal update by taking a step in the direction of the gradient
        # and using the posterior mode to find the new global states
        global_means_next_ = gaussian_cp_posterior_mode(
            effective_emissions_, hazard_rates, mu0, sigmasq0, step_size
        )[1]
        _, _, _, expected_subj_means_next_ = gaussian_cp_smoother(
            global_means_next_, hazard_rates, mu0, sigmasq0, sigmasq_subj
        )
        g_next_ = (
            1 / sigmasq_subj * jnp.sum(expected_subj_means_ - expected_subj_means_next_, axis=0)
        )  # sum over subjects

        for _ in range(max_iter_backtracking):
            g = g.ravel()
            expected_subj_means_ = expected_subj_means_.ravel()
            update_direction = update_direction.ravel()

            rhs = (
                expected_subj_means_
                + g.dot(update_direction)
                + update_direction.dot(update_direction) / (2.0 * step_size)
            )
            if expected_subj_means_next_ <= rhs:
                # .. step size found ..
                break
            else:
                # .. backtracking, reduce step size ..
                step_size *= backtracking_factor
                effective_emissions_ = global_means_ - step_size * g
                update_direction = global_means_next_ - global_means_
        else:
            warnings.warn("Maxium number of line-search iterations reached")

        certificate = jnp.linalg.norm((global_means_ - global_means_next_) / step_size)
        global_means_ = global_means_next_
        expected_subj_means_ = expected_subj_means_next_
        g = g_next_

        if certificate < tol:
            break

        if n_iterations >= max_iter:
            break
        else:
            n_iterations += 1  # increase iter count and repeat

    if n_iterations >= max_iter:
        warnings.warn(
            "minimize_proximal_gradient did not reach the desired tolerance level",
            RuntimeWarning,
        )

    return global_means_, g
