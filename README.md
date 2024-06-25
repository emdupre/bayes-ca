# Bayes-CA

Code for Bayesian Changepoint Analysis, intended for applications to fMRI data.
Note that the current code requires low-dimensional data ; for real data applications, we assume that this decomposition has been handled separately.

## Recommended environment

We recommend creating a separate conda environment for running these experiments.
To do so, you can run the following commands from this directory:

```
conda create -n bayes-ca python=3.10
conda activate bayes-ca
pip install -e .
```