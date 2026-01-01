import numpy as np
from scipy.special import gammaln


def log_inv_gamma_pdf(sigma2, a0, b0):
    """
    Log-density of Inv-Gamma(a0, b0):

        p(s^2) ∝ b0^a0 / Gamma(a0) * s^(-a0-1) * exp(-b0 / s^2)

    log p(s^2) = a0 log b0 - log Gamma(a0) - (a0+1) log s^2 - b0/s^2
    """
    if sigma2 <= 0:
        return -np.inf
    return (
        a0 * np.log(b0)
        - gammaln(a0)
        - (a0 + 1.0) * np.log(sigma2)
        - (b0 / sigma2)
    )


def log_posterior(beta, log_sigma2, X, y, tau2=10.0, a0=2.0, b0=1.0):
    """
    Log posterior of (beta, log_sigma2) up to an additive constant.

    - beta: parameter vector (length p_local)
    - log_sigma2: scalar (log of noise variance)
    - X: design matrix (n x p_local)
    - y: response vector (length n)

    Same implementation as in the original project.
    """
    beta = np.asarray(beta)
    X = np.asarray(X)
    y = np.asarray(y)

    n, p_local = X.shape
    sigma2 = np.exp(log_sigma2)

    # Guard against extreme values of sigma^2 (same spirit as original)
    if sigma2 <= 1e-12 or sigma2 > 1e6:
        return -np.inf

    # Likelihood: y | X, beta, sigma2 ~ N(X beta, sigma2 I)
    resid = y - X @ beta
    ll = (
        -0.5 * n * np.log(2.0 * np.pi * sigma2)
        - 0.5 * np.sum(resid ** 2) / sigma2
    )

    # Prior on beta: N(0, tau2 I)
    lp_beta = (
        -0.5 * p_local * np.log(2.0 * np.pi * tau2)
        - 0.5 * np.sum(beta ** 2) / tau2
    )

    # Prior on sigma2: Inv-Gamma(a0, b0)
    lp_sigma2 = log_inv_gamma_pdf(sigma2, a0, b0)

    # Jacobian for transformation sigma2 = exp(log_sigma2)
    jac = log_sigma2  # log |d sigma2 / d log_sigma2| = log_sigma2

    return ll + lp_beta + lp_sigma2 + jac
