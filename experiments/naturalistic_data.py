from pathlib import Path

import jax.random as jr
import jax.numpy as jnp
from jax import vmap, jit
from fastprogress import progress_bar

import bayes_ca.inference as core
from bayes_ca.data import naturalistic_data


# @jit
def gibbs_sample_subject_means(
    key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
):
    num_subjects = subj_obs.shape[0]
    effective_sigmasq = 1 / (1 / sigmasq_obs + 1 / sigmasq_subj)
    effective_emissions = effective_sigmasq * (
        subj_obs / sigmasq_obs + global_means / sigmasq_subj
    )

    _sample_one = lambda key, y: core.gaussian_cp_posterior_sample(
        key, y, hazard_rates, mu_pri, sigmasq_pri, effective_sigmasq
    )[1]
    return vmap(_sample_one)(jr.split(key, num_subjects), effective_emissions)


# @jit
def prox_update_global_mean(
    stepsize, global_means, subj_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
):
    # Use exponential family magic to compute gradient of the
    # smooth part of the objective (not including the CP prior)
    _, _, _, expected_subj_means = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu_pri, sigmasq_pri, sigmasq_subj
    )
    g = 1 / sigmasq_subj * jnp.sum(subj_means - expected_subj_means, axis=0)  # sum over subjects

    # Compute the proximal update by taking a step in the direction of the gradient
    # and using the posterior mode to find the new global states
    effective_emissions = global_means + stepsize * g
    return core.gaussian_cp_posterior_mode(
        effective_emissions, hazard_rates, mu_pri, sigmasq_pri, stepsize
    )[1]


@jit
def step(
    key,
    stepsize,
    subj_obs,
    sigmasq_obs,
    global_means,
    sigmasq_subj,
    mu_pri,
    sigmasq_pri,
    hazard_rates,
):
    # Sample new subject means
    subj_means = gibbs_sample_subject_means(
        key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
    )

    # Update the global mean
    global_means = prox_update_global_mean(
        stepsize, global_means, subj_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
    )

    return global_means, subj_means


# load experimental data
data_dir = Path("/", "Users", "emdupre", "Desktop", "brainiak_data")
bold = naturalistic_data(data_dir)
train = bold[..., :8].T
test = bold[..., 8:].T
num_subjects, num_timesteps, num_features = train.shape

# initialize random key for sampling
key = jr.PRNGKey(0)
this_key, key = jr.split(key)

# set relevant hyperparams
num_states = 100
max_duration = num_states + 1
hazard_prob = 0.01
hazard_rates = hazard_prob * jnp.ones(max_duration)
hazard_rates = hazard_rates.at[-1].set(1.0)

mu_pri = jnp.repeat(0.0, num_features)
sigmasq_pri = 3.0**2
sigmasq_subj = 0.5 * sigmasq_pri  # Note: variance of jump size is 2 * sigmasq_pri
sigmasq_obs = jnp.var(train)  # flattened variance of training sample


stepsize = 0.001
global_means = jnp.zeros((num_timesteps, num_features))

for itr in progress_bar(range(10000)):
    this_key, key = jr.split(key)
    global_means, subj_means = step(
        this_key,
        stepsize,
        train,
        sigmasq_obs,
        global_means,
        sigmasq_subj,
        mu_pri,
        sigmasq_pri,
        hazard_rates,
    )