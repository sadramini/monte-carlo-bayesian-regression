import numpy as np
import pandas as pd


def autocorr(x, max_lag=100):
    """
    Autocorrelation function up to max_lag, same as in the project.
    """
    x = np.asarray(x)
    x = x - x.mean()
    denom = np.sum(x ** 2)
    if denom < 1e-12:
        return np.full(max_lag + 1, np.nan)
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            num = denom
        else:
            num = np.sum(x[: len(x) - lag] * x[lag:])
        acf.append(num / denom)
    return np.array(acf)


def ess_from_acf(x, max_lag=200):
    """
    Effective sample size using the same method as in the PDF:

    - compute autocorr(x)
    - sum positive lags until the first negative / NaN
    - ESS = len(x) / (1 + 2 * sum_rho)
    """
    x = np.asarray(x)
    if np.var(x) < 1e-12:
        return np.nan
    acf = autocorr(x, max_lag=max_lag)
    if np.all(np.isnan(acf)):
        return np.nan
    s = 0.0
    for k in range(1, len(acf)):
        if np.isnan(acf[k]) or acf[k] < 0:
            break
        s += acf[k]
    tau_int = 1 + 2 * s
    return len(x) / tau_int


def split_rhat(chains_1d):
    """
    Split-Rhat exactly as in the project:

    - split each chain into two halves
    - compute W and B on the split chains
    """
    split = []
    for x in chains_1d:
        x = np.asarray(x)
        half = len(x) // 2
        split.append(x[:half])
        split.append(x[half:2 * half])

    m = len(split)
    n = min(len(s) for s in split)
    split = [s[:n] for s in split]

    chain_means = np.array([s.mean() for s in split])
    chain_vars = np.array([s.var(ddof=1) for s in split])

    W = chain_vars.mean()
    B = n * chain_means.var(ddof=1)
    var_hat = (n - 1) / n * W + (1.0 / n) * B
    return np.sqrt(var_hat / W)


def geweke_z(x, first=0.1, last=0.5):
    """
    Simple Geweke diagnostic, same as in the project.
    """
    x = np.asarray(x)
    n = len(x)
    n_first = int(first * n)
    n_last = int(last * n)
    a = x[:n_first]
    b = x[-n_last:]

    va = a.var(ddof=1)
    vb = b.var(ddof=1)
    if va < 1e-12 or vb < 1e-12:
        return np.nan
    z = (a.mean() - b.mean()) / np.sqrt(va / len(a) + vb / len(b))
    return z


def build_diagnostics_table(chains, param_names):
    """
    This creates the diagnostics DataFrame exactly as in the notebook:

    - reports for Intercept + first 3 features (indices 0,1,2,3)
    - and for log(sigma^2)
    - columns:
        param, Rhat, ESS (min over chains), ESS (mean over chains),
        Geweke z (mean), Geweke z (max abs)
    """
    report_idx = [0, 1, 2, 3]  # intercept + first 3 features
    rows = []

    # beta's diagnostics
    for idx in report_idx:
        post_samples = [ch["betas_post"][:, idx] for ch in chains]
        rhat = split_rhat(post_samples)
        ess_list = [ess_from_acf(s, max_lag=300) for s in post_samples]
        geweke_list = [geweke_z(s) for s in post_samples]
        rows.append(
            {
                "param": param_names[idx],
                "Rhat": rhat,
                "ESS (min over chains)": np.nanmin(ess_list),
                "ESS (mean over chains)": np.nanmean(ess_list),
                "Geweke z (mean)": np.nanmean(geweke_list),
                "Geweke z (max abs)": np.nanmax(np.abs(geweke_list)),
            }
        )

    # log(sigma^2) diagnostics
    post_sig = [ch["log_sigma2_post"] for ch in chains]
    ess_sig = [ess_from_acf(s, max_lag=300) for s in post_sig]
    gew_sig = [geweke_z(s) for s in post_sig]
    rows.append(
        {
            "param": "log(sigma^2)",
            "Rhat": split_rhat(post_sig),
            "ESS (min over chains)": np.nanmin(ess_sig),
            "ESS (mean over chains)": np.nanmean(ess_sig),
            "Geweke z (mean)": np.nanmean(gew_sig),
            "Geweke z (max abs)": np.nanmax(np.abs(gew_sig)),
        }
    )

    diag_df = pd.DataFrame(rows)
    return diag_df
