import click
from scipy import stats
import jax.numpy as jnp
import jax.random as jr
from jax import vmap, jit
import matplotlib.pyplot as plt
from fastprogress import progress_bar
from sklearn.decomposition import PCA, FactorAnalysis
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions

import bayes_ca.inference as core
from bayes_ca.prox_grad import pgd_jaxopt
from bayes_ca.data import naturalistic_data
from bayes_ca._utils import _safe_handling_params


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


@jit
def step(
    key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
):

    # Sample new subject means
    subj_means = gibbs_sample_subject_means(
        key, subj_obs, sigmasq_obs, global_means, sigmasq_subj, mu_pri, sigmasq_pri, hazard_rates
    )

    # Update the global mean
    # result = pgd(global_means, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates)  # 48m
    # global_means = result.x

    result = pgd_jaxopt(  # 55m
        global_means, subj_means, mu_pri, sigmasq_pri, sigmasq_subj, hazard_rates
    )
    global_means = result.params

    joint_lp = core.joint_lp(
        global_means,
        subj_means,
        subj_obs,
        mu_pri,
        sigmasq_pri,
        sigmasq_subj,
        sigmasq_obs,
        hazard_rates,
    )

    # prints will execute and output on the first call/run/compilation, but are then jitted away
    # and so will be silenced in a subsequent function call that is not being re-jitted - Andy
    # print("I am being jitted !")

    return global_means, subj_means, joint_lp


@click.command()
@click.option("--seed", default=0, help="Random seed.")
@click.option("--mu_pri", default=0.0, help="")
@click.option("--sigmasq_pri", default=1.0**2, help="")
@click.option("--sigmasq_subj", default=1.0**2, help="")
@click.option("--hazard_prob", default=0.01, help="")
@click.option("--max_duration", default=100, help="")
@click.option(
    "--data_dir", default="/Users/emdupre/Desktop/brainiak_data", help="Data directory."
)
@click.option("--verbose", default=False, help="")
def main(seed, data_dir, mu_pri, sigmasq_pri, sigmasq_subj, hazard_prob, max_duration, verbose):
    """

    Parameters
    ----------

    Returns
    -------
    """
    key = jr.PRNGKey(seed)

    bold = naturalistic_data(data_dir=data_dir)
    train = bold[..., :8].T
    # z-scoring in time, following Baldassano model
    train = jnp.asarray([stats.zscore(t, axis=1, ddof=1) for t in train])

    if verbose:
        # look at variance explained in one train subject
        pca = PCA(n_components=0.90).fit(train[0])
        PC_values = jnp.arange(pca.n_components_) + 1
        plt.plot(PC_values, pca.explained_variance_ratio_, "o-")
        plt.title("Scree Plot")
        plt.xlabel("Principal Component")
        plt.ylabel("Variance Explained")
        plt.show()

        # compare Factor Analysis and PCA dim reductions
        fa = FactorAnalysis(n_components=40).fit(train[0])
        pca = PCA(n_components=40).fit(train[0])
        plt.plot(fa.components_[0], label="FactorAnalysis")
        plt.plot(pca.components_[0], label="PCA")
        plt.legend()
        plt.show()

    pca_train = jnp.asarray([FactorAnalysis(n_components=1).fit_transform(t) for t in train])
    num_subjects, _, num_features = pca_train.shape

    # model settings
    mu_pri = _safe_handling_params(mu_pri, num_features)
    sigmasq_pri = _safe_handling_params(sigmasq_pri, num_features)
    sigmasq_subj = _safe_handling_params(sigmasq_subj, num_features)
    sigmasq_obs = jnp.sqrt(jnp.var(pca_train, axis=(0, 1)))

    max_duration = max_duration
    hazard_rates = hazard_prob * jnp.ones(max_duration)
    hazard_rates = hazard_rates.at[-1].set(1.0)

    # create train split and initialize globals
    global_means = jnp.mean(pca_train, axis=0)

    _, _, _, smooth_means = core.gaussian_cp_smoother(
        global_means, hazard_rates, mu_pri, sigmasq_pri, sigmasq_obs
    )
    plt.plot(global_means, label=f"1D input data after FactorAnalysis")
    plt.plot(smooth_means, label="smoothed means")
    plt.legend()
    plt.show()

    lps = []
    for _ in progress_bar(range(2)):
        this_key, key = jr.split(key)
        global_means, subj_means, train_lp = step(
            this_key,
            pca_train,
            sigmasq_obs,
            global_means,
            sigmasq_subj,
            mu_pri,
            sigmasq_pri,
            hazard_rates,
        )
        lps.append(train_lp)

    plt.plot(lps)
    plt.show()

    # l = plt.plot(subj_means[0][:, 0], label=f"sampled $\mu^n$ for subj. 0")[0]
    # plt.plot(pca_train[0][:, 0], "o", color=l.get_color(), alpha=0.1, lw=3)
    plt.plot(global_means, label="sampled $\mu^0$")
    plt.title(f"FA : 1D sampled means, $\sigma^2_{{subj}}$ : {jnp.sqrt(sigmasq_subj[0]):.2f}")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
