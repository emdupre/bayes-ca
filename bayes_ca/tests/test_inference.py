import jax.numpy as jnp
import jax.random as jr
from jax.lax import conv

from bayes_ca.inference import (
    sample_cp_prior,
    compute_pred_log_likes,
    compute_conditional_means,
    hmm_backward_filter,
    hmm_filter,
    hmm_smoother,
)

# Test
key = jr.PRNGKey(0)
num_timesteps = 1000
K = 100  # max run length
lmbda = 0.1  # prob changepoint
mu0 = 0.0  # prior mean
sigmasq0 = 3**2  # prior variance
sigmasq = 0.5**2  # observation variance

# Compute hazard rates under a geometric duration distribution
# Note: we can generalize this for other changepoint distributions
hazard_rates = lmbda * jnp.ones(K + 1)
hazard_rates = hazard_rates.at[-1].set(1.0)

# Sample the prior
this_key, key = jr.split(key)
zs, mus = sample_cp_prior(this_key, num_timesteps, hazard_rates, mu0, sigmasq0)

# Sample noisy observations
this_key, key = jr.split(key)
xs = mus + jnp.sqrt(sigmasq) * jr.normal(this_key, mus.shape)
lls = compute_pred_log_likes(xs, K, mu0, sigmasq0, sigmasq)


def test_log_normalizers(hazard_rates, lls):
    """ """
    forward_normalizer, _, _ = hmm_filter(hazard_rates, lls)
    backward_normalizer, _ = hmm_backward_filter(hazard_rates, lls)
    assert jnp.isclose(forward_normalizer, backward_normalizer, atol=1.0)


def test_kernel_conv(hazard_rates, lls, K):
    """ """
    _, _, transition_probs = hmm_smoother(hazard_rates, lls)
    kernel = jnp.tril(jnp.ones((K + 1, K + 1)))[None, None, :, :]  # OIKK
    assert jnp.allclose(
        conv(
            transition_probs.T[None, None, :, :],  # NCKT
            kernel,  # 11KK
            (1, 1),  # window strides
            [(0, 0), (0, K)],  # padding
        )[
            0, 0, 0
        ],  # T output
        1.0,
    )


def test_posterior_means(xs, K, transition_probs):
    """ """
    conditional_means = compute_conditional_means(xs, K)
    inpt = (transition_probs * conditional_means).T  # KT
    inpt = inpt[None, None, :, :]  # NCKT
    kernel = jnp.tril(jnp.ones((K + 1, K + 1)))[None, None, :, :]  # OIKK
    posterior_means = conv(inpt, kernel, (1, 1), [(0, 0), (0, K)])[0, 0, 0]  # T

    # Test against naive implementation
    t = 60
    foo = 0.0
    for k in range(K + 1):
        for i in range(min(k + 1, num_timesteps - 1)):
            foo += transition_probs[t + i, k] * conditional_means[t + i, k]
    assert jnp.allclose(posterior_means[t], foo)
