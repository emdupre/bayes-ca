from .inference import (
    _safe_handling_params,
    _compute_gaussian_lls,
    _compute_gaussian_stats,
    cp_filter,
    cp_backward_filter,
    cp_smoother,
    cp_posterior_mode,
    cp_posterior_sample,
    gaussian_cp_smoother,
    sample_gaussian_cp_model,
    gaussian_cp_posterior_mode,
    gaussian_cp_posterior_sample,
)

from .prox_grad import line_search, nesterov_acceleration
