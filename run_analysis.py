from pathlib import Path
import numpy as np
import pandas as pd
import time

from bayes_regression import (
    load_and_prepare_data,
    fit_ols_baseline,
    run_mh_chain,
    build_diagnostics_table,
    autocorr,
    plot_traces_for_all_params,
    plot_running_means_for_all_params,
    plot_acf_for_params,
    posterior_predictive_samples,
)


def main():
    # ------------------------------------------------------------------
    # 1–3. Data loading, splitting, standardizing (exact same steps)
    # ------------------------------------------------------------------
    X_train, X_test, y_train, y_test, param_names = load_and_prepare_data()

    print(f"Train shape: {X_train.shape}")
    print(f"Test shape:  {X_test.shape}")

    # ------------------------------------------------------------------
    # 4. Prior hyperparameters (same as original)
    # ------------------------------------------------------------------
    tau2 = 10.0
    a0, b0 = 2.0, 1.0

    # Chain settings
    n_iter = 12000
    burn = 4000

    # Proposal settings
    beta_step = 0.02
    log_sigma2_step = 0.05

    # ------------------------------------------------------------------
    # 5. OLS baseline on standardized predictors
    # ------------------------------------------------------------------
    ols_model = fit_ols_baseline(X_train, y_train)

    # ------------------------------------------------------------------
    # 6–8. Run multiple chains (main experiment)
    # ------------------------------------------------------------------
    n_chains = 4
    chains = []
    base_seed = 123

    t0_global = time.perf_counter()
    for m in range(n_chains):
        sd = base_seed + m
        print(f"Running chain {m+1}/{n_chains} with seed {sd}...")
        out = run_mh_chain(
            X_train,
            y_train,
            n_iter=n_iter,
            burn=burn,
            beta_step=beta_step,
            log_sigma2_step=log_sigma2_step,
            tau2=tau2,
            a0=a0,
            b0=b0,
            init_beta=None,
            init_log_sigma2=None,
            seed=sd,
        )
        chains.append(out)
    t1_global = time.perf_counter()
    print(f"Total sampling time: {t1_global - t0_global:.2f} seconds")

    # Print acceptance rates (same style as PDF)
    for i, ch in enumerate(chains):
        print(
            f"Chain {i+1} accept_rate_beta (per-dim): {ch['accept_rate_beta']:.3f} | "
            f"accept_rate_sigma2: {ch['accept_rate_sigma2']:.3f}"
        )

    # ------------------------------------------------------------------
    # Folder setup for results
    # ------------------------------------------------------------------
    results_dir = Path("results")
    plots_dir = results_dir / "plots"
    tables_dir = results_dir / "tables"
    plots_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 9–12. Diagnostics: traces, running means, ESS, R-hat, Geweke, ACF
    # ------------------------------------------------------------------
    plot_traces_for_all_params(chains, param_names, plots_dir)
    plot_running_means_for_all_params(chains, param_names, plots_dir)

    diag_df = build_diagnostics_table(chains, param_names)
    diag_df.to_csv(tables_dir / "diagnostics.csv", index=False)
    print(diag_df)

    # MCSE (using ESS (mean over chains), like in the notebook)
    mcse_rows = []
    beta_all = np.vstack([ch["betas_post"] for ch in chains])
    log_sigma2_all = np.concatenate([ch["log_sigma2_post"] for ch in chains])

    for j, name in enumerate(param_names[:4]):  # same subset as diagnostics
        samples = beta_all[:, j]
        row = diag_df.loc[diag_df["param"] == name].iloc[0]
        ess_mean = row["ESS (mean over chains)"]
        if np.isnan(ess_mean) or ess_mean <= 0:
            mcse = np.nan
        else:
            mcse = np.sqrt(np.var(samples, ddof=1) / ess_mean)
        mcse_rows.append({"param": name, "MCSE": mcse})

    mcse_df = pd.DataFrame(mcse_rows)
    mcse_df.to_csv(tables_dir / "mcse.csv", index=False)

    # ACF plots (one chain, a few parameters)
    plot_acf_for_params(
        chains[0],
        param_names,
        idx_list=[0, 1, 2],
        max_lag=60,
        out_path=plots_dir / "acf_selected_params.png",
    )

    # ------------------------------------------------------------------
    # 13–14. Combine chains + posterior summaries
    # ------------------------------------------------------------------
    sigma2_all = np.exp(log_sigma2_all)

    summ_rows = []
    p = beta_all.shape[1]
    for j in range(p):
        s = beta_all[:, j]
        summ_rows.append(
            {
                "param": param_names[j],
                "post_mean": np.mean(s),
                "post_median": np.median(s),
                "CI_2.5%": np.quantile(s, 0.025),
                "CI_97.5%": np.quantile(s, 0.975),
                "P(beta>0)": np.mean(s > 0),
            }
        )
    summ_df = pd.DataFrame(summ_rows)
    summ_df.to_csv(tables_dir / "posterior_summary.csv", index=False)
    print(summ_df)

    # ------------------------------------------------------------------
    # 15. Posterior predictive on test data (same logic as notebook)
    # ------------------------------------------------------------------
    y_pred_draws = posterior_predictive_samples(
        beta_all, log_sigma2_all, X_test, n_draws=2000, seed=base_seed + 999
    )

    pred_mean = y_pred_draws.mean(axis=0)
    pred_lower = np.quantile(y_pred_draws, 0.025, axis=0)
    pred_upper = np.quantile(y_pred_draws, 0.975, axis=0)

    pred_df = pd.DataFrame(
        {
            "y_true": y_test,
            "pred_mean": pred_mean,
            "pred_2.5": pred_lower,
            "pred_97.5": pred_upper,
        }
    )
    pred_df.to_csv(tables_dir / "posterior_predictive_test.csv", index=False)

    # ------------------------------------------------------------------
    # 16. OLS vs Bayesian (you can add more comparison if you like)
    # ------------------------------------------------------------------
    ols_df = pd.DataFrame(
        {"param": param_names, "OLS_estimate": ols_model.params}
    )
    ols_df.to_csv(tables_dir / "ols_estimates.csv", index=False)

    print("Analysis complete. Results saved in 'results/plots' and 'results/tables'.")


if __name__ == "__main__":
    main()
