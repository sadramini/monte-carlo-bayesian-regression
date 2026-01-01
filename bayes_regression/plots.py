from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from .diagnostics import autocorr


def plot_traces_for_all_params(chains, param_names, out_dir: Path):
    """
    Trace plots for all beta's + log(sigma^2), exactly like the notebook,
    but saved as PNGs into out_dir instead of shown on screen.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chains = len(chains)
    n_iter, p = chains[0]["betas"].shape

    # beta traces
    for j in range(p):
        plt.figure(figsize=(8, 4))
        for c in range(n_chains):
            plt.plot(chains[c]["betas"][:, j], alpha=0.7, label=f"chain {c+1}")
        plt.xlabel("Iteration")
        plt.ylabel(param_names[j])
        plt.title(f"Trace: {param_names[j]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"trace_{param_names[j]}.png")
        plt.close()

    # log(sigma^2)
    plt.figure(figsize=(8, 4))
    for c in range(n_chains):
        plt.plot(chains[c]["log_sigma2"], alpha=0.7, label=f"chain {c+1}")
    plt.xlabel("Iteration")
    plt.ylabel("log(sigma^2)")
    plt.title("Trace: log(sigma^2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "trace_log_sigma2.png")
    plt.close()


def plot_running_means_for_all_params(chains, param_names, out_dir: Path):
    """
    Running means per chain for each parameter (same logic as notebook),
    saved as PNGs.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    n_chains = len(chains)
    n_iter, p = chains[0]["betas"].shape

    # beta running means
    for j in range(p):
        plt.figure(figsize=(8, 4))
        for c in range(n_chains):
            x = chains[c]["betas"][:, j]
            running_mean = np.cumsum(x) / np.arange(1, n_iter + 1)
            plt.plot(running_mean, alpha=0.7, label=f"chain {c+1}")
        plt.xlabel("Iteration")
        plt.ylabel(f"Running mean of {param_names[j]}")
        plt.title(f"Running mean: {param_names[j]}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"running_mean_{param_names[j]}.png")
        plt.close()

    # log(sigma^2)
    plt.figure(figsize=(8, 4))
    for c in range(n_chains):
        x = chains[c]["log_sigma2"]
        running_mean = np.cumsum(x) / np.arange(1, n_iter + 1)
        plt.plot(running_mean, alpha=0.7, label=f"chain {c+1}")
    plt.xlabel("Iteration")
    plt.ylabel("Running mean of log(sigma^2)")
    plt.title("Running mean: log(sigma^2)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "running_mean_log_sigma2.png")
    plt.close()


def plot_acf_for_params(chain, param_names, idx_list=None, max_lag=80, out_path: Path = None):
    """
    ACF plots for one chain and a few parameters.

    This is the same as the 'plot_acf_for_params' function in the project.
    If out_path is provided, save; otherwise just show (for interactive use).
    """
    if idx_list is None:
        idx_list = [0, 1, 2]  # Intercept, MedInc, HouseAge

    rows = len(idx_list) + 1
    plt.figure(figsize=(10, 8))

    for r, idx in enumerate(idx_list):
        plt.subplot(rows, 1, r + 1)
        acf_vals = autocorr(chain["betas_post"][:, idx], max_lag=max_lag)
        plt.plot(acf_vals, marker="o", markersize=3)
        plt.title(f"ACF: {param_names[idx]}")
        plt.ylim(-0.2, 1.05)

    plt.subplot(rows, 1, rows)
    acf_logsig = autocorr(chain["log_sigma2_post"], max_lag=max_lag)
    plt.plot(acf_logsig, marker="o", markersize=3)
    plt.title("ACF: log(sigma^2)")
    plt.ylim(-0.2, 1.05)

    plt.tight_layout()

    if out_path is None:
        plt.show()
    else:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path)
        plt.close()
