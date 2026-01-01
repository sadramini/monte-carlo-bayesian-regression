import numpy as np


def posterior_predictive_samples(beta_samples, log_sigma2_samples, X_new, n_draws, seed=0):
    """
    Draw Y_new from the posterior predictive distribution:

        Y_new = X_new @ beta + eps, eps ~ N(0, sigma^2)

    This follows the structure of the function in the original project.
    """
    rng = np.random.default_rng(seed)

    beta_samples = np.asarray(beta_samples)
    log_sigma2_samples = np.asarray(log_sigma2_samples)
    X_new = np.asarray(X_new)

    n_total = beta_samples.shape[0]
    idx = rng.choice(n_total, size=n_draws, replace=False)

    betas_draw = beta_samples[idx]
    sigma2_draw = np.exp(log_sigma2_samples[idx])

    # y_pred_draws: (n_draws, n_new)
    mean_part = betas_draw @ X_new.T
    noise = rng.normal(0.0, np.sqrt(sigma2_draw)[:, None], size=mean_part.shape)
    y_pred_draws = mean_part + noise
    return y_pred_draws
