import jax.numpy as jnp
import jax.random as jr
from jax.lax import conv, scan
from jaxtyping import Array, Float, PyTree
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _normalize(u, axis=0, eps=1e-15):
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


def _condition_on(probs, ll):
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


def _predict(probs, hazard_rates):
    return jnp.concatenate(
        [jnp.array([probs @ hazard_rates]), (1 - hazard_rates[:-1]) * probs[:-1]]
    )


def _backward_predict(probs, hazard_rates):
    p = hazard_rates * probs[0]
    p = p.at[:-1].add((1 - hazard_rates[:-1]) * probs[1:])
    return p


###
# Generic functions for changepoint inference (over run lengths)
###
def cp_filter(hazard_rates, pred_log_likes):
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


def cp_backward_filter(hazard_rates, pred_log_likes):
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
    (log_normalizer, _), backward_pred_probs = scan(_step, carry, pred_log_likes, reverse=True)
    return log_normalizer, backward_pred_probs


def cp_smoother(hazard_rates, pred_log_likes):
    """ """
    log_normalizer, filtered_probs, predicted_probs = cp_filter(hazard_rates, pred_log_likes)
    _, backward_pred_probs = cp_backward_filter(hazard_rates, pred_log_likes)

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


def cp_posterior_sample(key, hazard_rates, pred_log_likes):
    r"""Sample a latent sequence from the posterior.

    Args:
        key: random number generator
        hazard_rates: $p(z_{t+1} = 0 \mid z_t)$
        pred_log_likes: $p(x_t \mid x_{1:t-1}, z_t)$

    Returns:
        :sample of the latent states, $z_{1:T}$

    """
    num_timesteps = pred_log_likes.shape[0]
    num_states = pred_log_likes.shape[1] - 1

    # Run the HMM smoother
    _, filtered_probs = cp_filter(hazard_rates, pred_log_likes)

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
            run_length > 0, run_length - 1, jr.choice(_key, a=num_states + 1, p=smoothed_probs)
        )

        return run_length, run_length

    # Run the HMM smoother
    rngs = jr.split(key, num_timesteps)
    last_length = jr.choice(rngs[-1], a=num_states + 1, p=filtered_probs[-1])
    args = (rngs[:-1], filtered_probs[:-1])
    _, run_lengths = scan(_step, last_length, args, reverse=True)

    # Stack and return the run lengths
    return jnp.append(run_lengths, last_length)
    

def cp_posterior_mode(
    hazard_rates,
    pred_log_likes,
):
    r"""Compute the most likely state sequence. This is called the Viterbi algorithm.

    Args:
        initial_distribution: $p(z_1 \mid u_1, \theta)$
        hazard_rates: $p(z_{t+1} = j \mid z_t = i, \theta)$
        log_likelihoods: $p(y_t \mid z_t, u_t, \theta)$ for $t=1,\ldots, T$.

    Returns:
        most likely state sequence
    """
    num_states = pred_log_likes.shape[1] - 1

    # Note: we could allow more general initial distributions for left-censoring
    initial_dist = jnp.zeros(num_states + 1, dtype=float)
    initial_dist = initial_dist.at[0].set(1.0)

    # Run the backward pass
    def _backward_pass(best_next_score, next_log_likes):
        X = best_next_score + next_log_likes
        scores = jnp.stack(
            [jnp.log(hazard_rates[:-1]) + X[0], jnp.log(1 - hazard_rates[:-1]) + X[1:]]
        )

        best_idx = jnp.argmax(scores, axis=0)
        best_next_state = jnp.where(best_idx == 0, 0, jnp.arange(1, num_states + 1))
        best_next_score = jnp.max(scores, axis=0)
        return jnp.append(best_next_score, X[0]), jnp.append(best_next_state, 0)

    best_second_score, best_next_states = scan(
        _backward_pass, jnp.zeros(num_states + 1), pred_log_likes[1:], reverse=True
    )

    # Run the forward pass
    def _forward_pass(state, best_next_state):
        next_state = best_next_state[state]
        return next_state, next_state

    first_state = jnp.argmax(
        jnp.log(initial_dist) + pred_log_likes[0] + best_second_score
    )
    _, states = scan(_forward_pass, first_state, best_next_states)

    return jnp.insert(first_state, 0, states)


###
# The functions below are for the special case of a changepoint model with
# scalar Gaussian emissions. 
###
def _compute_gaussian_stats(emissions, num_states):
    r"""
    Compute conditional stats of mu_t given runs of varying length ending at time t

    Returns:
        partial_sums: (T, K+1) array x[t-k:t+1].sum() for all t and k
        partial_counts: (T, K+1) array number of data points contributing to the sum
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

    initial_carry = (jnp.zeros(num_states + 1), jnp.zeros(num_states + 1))
    _, (partial_sums, partial_counts) = scan(_step, initial_carry, emissions)
    return partial_sums, partial_counts

# Test that each row = (0, 1, ..., K) except in the top right corner of the matrix


def _compute_gaussian_lls(emissions, partial_sums, partial_counts, mu0, sigmasq0, sigmasq):
    r"""
    Compute the one-step ahead predictive log likelihoods

        \log p(x[t+1] | x[t-K:t])

    integrating over the latent mean \mu[t] for that run.

    Note: partial_sums and parital_counts come from `compute_conditional_stats` above.
    """
    num_states = partial_sums.shape[1] - 1
    
    # Compute sufficient stats of the *predictive* distribution
    pred_sums = jnp.vstack([jnp.zeros(num_states + 1), partial_sums[:-1]])
    pred_counts = jnp.vstack([jnp.zeros(num_states + 1), partial_counts[:-1]])

    # Compute the posterior predictive distribution and return the log prob
    sigmasq_post = 1 / (1 / sigmasq0 + pred_counts / sigmasq)
    mu_post = sigmasq_post * (mu0 / sigmasq0 + pred_sums / sigmasq)
    sigmasq_pred = sigmasq_post + sigmasq
    return tfd.Normal(mu_post, jnp.sqrt(sigmasq_pred)).log_prob(emissions[:, None])


def gaussian_cp_posterior_sample(key, 
                                 emissions : Float[Array, "num_timesteps"], 
                                 hazard_rates  : Float[Array, "max_duration"], 
                                 mu0 : Float, 
                                 sigmasq0 : Float, 
                                 sigmasq : Float):
    r"""
    Return a sample of the run lengths and the latent means given emissions.

    #TODO: document this 
    """
    num_timesteps = emissions.shape[0]
    num_states = hazard_rates.shape[0] - 1

    # First sample the run lengths
    k1, k2 = jr.split(key)
    partial_sums, partial_counts = _compute_gaussian_stats(emissions, num_states)
    lls = _compute_gaussian_lls(emissions, partial_sums, partial_counts, mu0, sigmasq0, sigmasq)
    zs = cp_posterior_sample(k1, hazard_rates, lls)

    # Then sample the means
    def _backward_sample(mu_next, args):
        key, x_sum, n, z, z_next = args
        sigmasq_post = 1 / (1 / sigmasq0 + n[z] / sigmasq)
        mu_post = sigmasq_post * (mu0 / sigmasq0 + x_sum[z] / sigmasq)

        mu = jnp.where(z_next == 0, 
                       tfd.Normal(mu_post, jnp.sqrt(sigmasq_post)).sample(seed=key),
                       mu_next)
        return mu, mu
        
    args = (jr.split(k2, num_timesteps),
            partial_sums, 
            partial_counts,
            zs, 
            jnp.append(zs[1:], 0)) 
    _, mus = scan(_backward_sample, jnp.nan, args, reverse=True)
    return zs, mus


def gaussian_cp_posterior_mode(emissions : Float[Array, "num_timesteps"], 
                               hazard_rates  : Float[Array, "max_duration"], 
                               mu0 : Float, 
                               sigmasq0 : Float, 
                               sigmasq : Float):
    num_states = hazard_rates.shape[0] - 1

    # First compute the most likely run lengths)
    partial_sums, partial_counts = _compute_gaussian_stats(emissions, num_states)
    lls = _compute_gaussian_lls(emissions, partial_sums, partial_counts, mu0, sigmasq0, sigmasq)
    zs = cp_posterior_mode(hazard_rates, lls)

    # Now compute the most likely mus
    def _backward_sample(mu_next, args):
        x_sum, n, z, z_next = args
        sigmasq_post = 1 / (1 / sigmasq0 + n[z] / sigmasq)
        mu_post = sigmasq_post * (mu0 / sigmasq0 + x_sum[z] / sigmasq)

        mu = jnp.where(z_next == 0, 
                       mu_post,
                       mu_next)
        return mu, mu
        
    args = (partial_sums, 
            partial_counts,
            zs, 
            jnp.append(zs[1:], 0)) 
    _, mus = scan(_backward_sample, jnp.nan, args, reverse=True)
    return zs, mus


def gaussian_cp_posterior_mean(emissions : Float[Array, "num_timesteps"], 
                               hazard_rates  : Float[Array, "max_duration"], 
                               mu0 : Float, 
                               sigmasq0 : Float, 
                               sigmasq : Float):
    num_states = hazard_rates.shape[0] - 1

    # First compute the most likely run lengths)
    partial_sums, partial_counts = _compute_gaussian_stats(emissions, num_states)
    lls = _compute_gaussian_lls(emissions, partial_sums, partial_counts, mu0, sigmasq0, sigmasq)
    _, E_zs, transition_probs = cp_smoother(hazard_rates, lls)

    sigmasq_post = 1 / (1 / sigmasq0 + partial_counts / sigmasq)
    mu_post = sigmasq_post * (mu0 / sigmasq0 + partial_sums / sigmasq)

    inpt = (transition_probs * mu_post).T        # KT
    inpt = inpt[None, None, :, :]                # NCKT
    kernel = jnp.tril(jnp.ones((num_states + 1, num_states + 1)))[None, None, :, :]   # OIKK

    E_mus = conv(inpt, kernel, (1, 1), [(0, 0), (0, num_states)])[0, 0, 0] # T
    return E_zs, E_mus


def sample_gaussian_cp_model(key, num_timesteps, hazard_rates, mu0, sigmasq0):
    """ 
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
