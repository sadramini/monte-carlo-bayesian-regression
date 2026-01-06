import numpy as np
from .model import log_posterior


def run_mh_chain(
    X,
    y,
    n_iter=12000,
    burn=4000,
    beta_step=0.02,
    log_sigma2_step=0.05,
    tau2=10.0,
    a0=2.0,
    b0=1.0,
    init_beta=None,
    init_log_sigma2=None,
    seed=0,
):
    """
    Component-wise random-walk Metropolis-Hastings, exactly as in the project:

    - Updates beta_0,...,beta_p one at a time
    - Then updates log(sigma^2)
    - Returns all draws plus post-burn draws + acceptance rates.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    n, p_local = X.shape

    rng = np.random.default_rng(seed)

    # Initialize
    if init_beta is None:
        beta = np.zeros(p_local)
    else:
        beta = np.asarray(init_beta).copy()

    if init_log_sigma2 is None:
        log_sigma2 = 0.0  # sigma^2 = 1
    else:
        log_sigma2 = float(init_log_sigma2)

    betas = np.zeros((n_iter, p_local))
    log_sigma2s = np.zeros(n_iter)

    # Initial log-posterior and safety check
    current_lp = log_posterior(beta, log_sigma2, X, y, tau2=tau2, a0=a0, b0=b0)
    if not np.isfinite(current_lp):
        raise RuntimeError("Initial log-posterior is not finite.")

    accepts_beta = 0
    accepts_sig = 0

    for t in range(n_iter):
        # --- update beta_0,...,beta_{p-1} (component-wise) ---
        for j in range(p_local):
            beta_prop = beta.copy()
            beta_prop[j] = beta_prop[j] + rng.normal(0.0, beta_step)
            prop_lp = log_posterior(
                beta_prop, log_sigma2, X, y, tau2=tau2, a0=a0, b0=b0
            )
            log_alpha = prop_lp - current_lp

            if np.log(rng.uniform()) < log_alpha:
                beta = beta_prop
                current_lp = prop_lp
                accepts_beta += 1

        # --- update log(sigma^2) ---
        log_sigma2_prop = log_sigma2 + rng.normal(0.0, log_sigma2_step)
        prop_lp = log_posterior(
            beta, log_sigma2_prop, X, y, tau2=tau2, a0=a0, b0=b0
        )
        log_alpha = prop_lp - current_lp

        if np.log(rng.uniform()) < log_alpha:
            log_sigma2 = log_sigma2_prop
            current_lp = prop_lp
            accepts_sig += 1

        betas[t] = beta
        log_sigma2s[t] = log_sigma2

    # Acceptance rates (same as in PDF)
    acc_beta_rate = accepts_beta / (n_iter * p_local)  # per beta update
    acc_sig_rate = accepts_sig / n_iter

    # Discard burn-in
    betas_post = betas[burn:]
    log_sigma2_post = log_sigma2s[burn:]

    return {
        "betas": betas,
        "log_sigma2": log_sigma2s,
        "betas_post": betas_post,
        "log_sigma2_post": log_sigma2_post,
        "accept_rate_beta": acc_beta_rate,
        "accept_rate_sigma2": acc_sig_rate,
        "burn": burn,
        "n_iter": n_iter,
    }
