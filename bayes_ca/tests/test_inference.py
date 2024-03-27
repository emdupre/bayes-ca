import jax
import jax.numpy as jnp
import jax.random as jr
from jax.lax import conv
import dynamax.hidden_markov_model.inference as hmm

from bayes_ca.inference import (
    _compute_gaussian_stats,
    _compute_gaussian_lls,
    cp_filter,
    cp_backward_filter,
    cp_smoother,
    cp_posterior_mode,
    gaussian_cp_smoother,
    sample_gaussian_cp_model,
)
from bayes_ca._utils import _safe_handling_params

# Test
key = jr.PRNGKey(0)
num_timesteps = 1000
num_features = 3
K = 100  # max run length
lmbda = 0.1  # prob changepoint
mu0 = 0.0  # prior mean
sigmasq0 = 3.0**2  # prior variance
sigmasq = 0.5**2  # observation variance

mu0 = _safe_handling_params(mu0, num_features)
sigmasq0 = _safe_handling_params(sigmasq0, num_features)
sigmasq = _safe_handling_params(sigmasq, num_features)

# Compute hazard rates under a geometric duration distribution
# Note: we can generalize this for other changepoint distributions
hazard_rates = lmbda * jnp.ones(K + 1)
hazard_rates = hazard_rates.at[-1].set(1.0)

# Sample the prior
this_key, key = jr.split(key)
zs, mus = sample_gaussian_cp_model(this_key, num_timesteps, hazard_rates, mu0, sigmasq0)

# Sample noisy observations
this_key, key = jr.split(key)
xs = mus + jnp.sqrt(sigmasq) * jr.normal(this_key, mus.shape)

# partial_sums, partial_counts = _compute_gaussian_stats(xs, K + 1)
partial_sums, partial_counts = jax.vmap(_compute_gaussian_stats, in_axes=(-1, None), out_axes=-1)(
    xs, K + 1
)
# lls = _compute_gaussian_lls(xs, K + 1, mu0, sigmasq0, sigmasq)
lls = jax.vmap(_compute_gaussian_lls, in_axes=(-1, None, 0, 0, 0))(
    xs, K + 1, mu0[:, None], sigmasq0[:, None], sigmasq[:, None]
)
lls = lls.sum(axis=0)
_, _, transition_probs = cp_smoother(hazard_rates, lls)


# def test_log_normalizers():
#     """ """
#     forward_normalizer, _, _ = cp_filter(hazard_rates, lls)
#     backward_normalizer, _ = cp_backward_filter(hazard_rates, lls)
#     assert jnp.isclose(forward_normalizer, backward_normalizer, atol=1.0)

assert hazard_rates[-1] == 1.0
A = jnp.diag(1 - hazard_rates[:-1], k=1)
A = A.at[:, 0].set(hazard_rates)
pi0 = jnp.zeros(K + 1)
pi0 = pi0.at[0].set(1.0)


def test_cp_filter():
    log_normalizer, _, _ = cp_filter(hazard_rates, lls)
    log_normalizer2, _, _ = hmm.hmm_filter(pi0, A, lls)
    assert jnp.isclose(log_normalizer, log_normalizer2, atol=1e-3)


def test_cp_smoother():
    log_normalizer, smoothed_probs, _ = cp_smoother(hazard_rates, lls)
    post = hmm.hmm_smoother(pi0, A, lls)
    assert jnp.allclose(smoothed_probs, post.smoothed_probs, atol=1e-3)


def test_cp_posterior_mode():
    zs = cp_posterior_mode(hazard_rates, lls)
    zs2 = hmm.hmm_posterior_mode(pi0, A, lls)
    assert jnp.allclose(zs, zs2)


def test_kernel_conv():
    """ """
    _, _, transition_probs = cp_smoother(hazard_rates, lls)
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


def test_posterior_means():
    """ """
    sigmasq_post = 1 / (1 / sigmasq0 + partial_counts / sigmasq)
    mu_post = sigmasq_post * (mu0 / sigmasq0 + partial_sums / sigmasq)
    _, _, _, posterior_means = gaussian_cp_smoother(xs, hazard_rates, mu0, sigmasq0, sigmasq)

    # Test against naive implementation
    t = 60
    foo = 0.0
    for k in range(K + 1):
        for i in range(min(k + 1, num_timesteps - 1)):
            foo += transition_probs[t + i, k] * mu_post[t + i, k]
    assert jnp.allclose(posterior_means[t], foo)
