"""
Bayesian linear regression on the California Housing data.

This package just groups the functions that were originally in one notebook.
The statistical code (model, MCMC, diagnostics) is identical to the original
project; it is only split into modules.
"""

from .data import load_and_prepare_data, fit_ols_baseline
from .model import log_inv_gamma_pdf, log_posterior
from .mcmc import run_mh_chain
from .diagnostics import (
    autocorr,
    ess_from_acf,
    split_rhat,
    geweke_z,
    build_diagnostics_table,
)
from .plots import (
    plot_traces_for_all_params,
    plot_running_means_for_all_params,
    plot_acf_for_params,
)
from .predictive import posterior_predictive_samples
