from pathlib import Path
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt


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
    import time

    n_chains = 4
    chains = []

    # Randomize MCMC seeds per run (while keeping data fixed)
    base_seed = int(time.time())
    print(f"Using base random seed: {base_seed}")

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
    # Folder setup for results (ANCHOR TO SCRIPT DIRECTORY)
    # ------------------------------------------------------------------
    base_dir = Path(__file__).resolve().parent
    results_dir = base_dir / "results"
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

    # --- NEW: ESS per second for main experiment ----------------------
    total_time = t1_global - t0_global

    ess_eff_rows = []
    for _, row in diag_df.iterrows():
        ess_eff_rows.append(
            {
                "param": row["param"],
                "ESS_per_second": row["ESS (mean over chains)"] / total_time,
            }
        )

    ess_eff_df = pd.DataFrame(ess_eff_rows)
    ess_eff_df.to_csv(tables_dir / "ess_per_second.csv", index=False)
    # ------------------------------------------------------------------

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
    # NEW: Proposal variance sensitivity experiment (beta_step grid)
    # ------------------------------------------------------------------
    beta_steps = [0.005, 0.01, 0.02, 0.05]
    sensitivity_rows = []

    for beta_step_sens in beta_steps:
        print(f"\nRunning sensitivity experiment: beta_step = {beta_step_sens}")
        sens_chains = []

        t0 = time.perf_counter()
        for m in range(n_chains):
            sd = base_seed + m
            out = run_mh_chain(
                X_train,
                y_train,
                n_iter=n_iter,
                burn=burn,
                beta_step=beta_step_sens,
                log_sigma2_step=log_sigma2_step,
                tau2=tau2,
                a0=a0,
                b0=b0,
                seed=sd,
            )
            sens_chains.append(out)

        t1 = time.perf_counter()
        elapsed = t1 - t0

        diag_df_sens = build_diagnostics_table(sens_chains, param_names)

        for _, row in diag_df_sens.iterrows():
            sensitivity_rows.append(
                {
                    "beta_step": beta_step_sens,
                    "param": row["param"],
                    "Rhat": row["Rhat"],
                    "ESS_mean": row["ESS (mean over chains)"],
                    "ESS_per_second": row["ESS (mean over chains)"] / elapsed,
                }
            )


    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(tables_dir / "sensitivity.csv", index=False)
    # ------------------------------------------------------------------
    # ---------------------------------------------------------
    # Plot: ESS/sec vs beta_step (tuning efficiency curve)
    # ---------------------------------------------------------
    # We group by beta_step and parameter so each parameter
    # gets its own efficiency curve across proposal values.

    fig, ax = plt.subplots()

    for param, g in sens_df.groupby("param"):
        g_sorted = g.sort_values("beta_step")
        ax.plot(
            g_sorted["beta_step"],
            g_sorted["ESS_per_second"],
            marker="o",
            label=param,
        )

    ax.set_xlabel("beta_step")
    ax.set_ylabel("ESS per second")
    ax.set_title("Sampler efficiency across proposal scales\n(ESS/sec vs beta_step)")
    ax.legend()
    fig.tight_layout()

    fig.savefig(plots_dir / "ess_per_second_vs_beta_step.png", dpi=200)
    plt.close(fig)
    # ---------------------------------------------------------

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
