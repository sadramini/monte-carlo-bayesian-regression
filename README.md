# Monte Carlo Bayesian Regression

This project implements a **Bayesian Linear Regression model** estimated
using a **Random-Walk Metropolis--Hastings MCMC sampler**. The workflow
is fully reproducible and organized as a modular research-style
pipeline, including posterior inference, convergence diagnostics, Monte
Carlo error assessment, and posterior predictive evaluation.

The primary application is the **California Housing dataset**, and the
project also includes comparison with a classical OLS baseline.

------------------------------------------------------------------------

## 🎯 Project Goals

-   Implement Bayesian linear regression from first principles
-   Use a hand‑written MCMC sampler (Random‑Walk Metropolis--Hastings)
-   Run multiple independent Markov chains
-   Evaluate convergence and sampler efficiency
-   Quantify Monte Carlo Standard Error (MCSE)
-   Perform posterior predictive checks
-   Compare against OLS regression as a frequentist benchmark
-   Produce clean, reproducible research outputs (tables + plots)

This repository is structured as a transparent and extensible **learning
and research project**, not just a script that prints results.

------------------------------------------------------------------------

## 🧠 Model Overview

The model assumes the standard Gaussian linear regression form

\[ y = X`\beta `{=tex}+ `\varepsilon `{=tex},
`\quad `{=tex}`\varepsilon `{=tex}`\sim `{=tex}`\mathcal `{=tex}N(0,
`\sigma`{=tex}\^2 I). \]

Priors:

-   (`\beta `{=tex}`\sim `{=tex}`\mathcal `{=tex}N(0,`\tau`{=tex}\^2 I))
-   (`\sigma`{=tex}\^2 `\sim `{=tex}`\text{Inverse‑Gamma}`{=tex}(a,b))

The posterior is explored using a **Random‑Walk MH sampler** with:

-   multiple chains
-   chain‑level adaptation settings
-   diagnostics stored to disk

------------------------------------------------------------------------

## 🧩 Project Structure

    .
    ├── data.py              # Load & preprocess dataset
    ├── model.py             # Likelihood + prior + posterior log‑density
    ├── mcmc.py              # RW‑Metropolis sampler + chain manager
    ├── diagnostics.py       # R-hat, ESS, MCSE, trace utilities
    ├── plots.py             # Trace, density, pair & PPC plots
    ├── predictive.py        # Posterior predictive simulation
    ├── run_analysis.py      # Main end‑to‑end pipeline
    └── results/
        ├── plots/           # Saved figures
        └── tables/          # CSV summaries and diagnostics

Each module is intentionally independent and reusable. The pipeline
script (`run_analysis.py`) orchestrates the full experiment.

------------------------------------------------------------------------

## 🚀 Workflow

Run the main analysis pipeline:

``` bash
python run_analysis.py
```

This performs:

1.  Load & preprocess data
2.  Initialize chains
3.  Run MCMC sampling
4.  Produce posterior summary tables
5.  Compute convergence diagnostics
6.  Evaluate MCSE
7.  Generate posterior predictive samples
8.  Compare results with OLS regression
9.  Save figures & tables to `results/`

All outputs are written to disk for full reproducibility.

------------------------------------------------------------------------

## 📊 Outputs

The pipeline generates:

### Tables (`results/tables/`)

-   posterior_summary.csv
-   diagnostics_rhat_ess_mcse.csv
-   chain_acceptance_rates.csv
-   ols_comparison.csv

### Plots (`results/plots/`)

-   trace plots
-   marginal posterior densities
-   chain comparison plots
-   posterior predictive checks
-   OLS vs posterior distributions

These are suitable for reports, coursework, or research documentation.

------------------------------------------------------------------------

## 🧪 Reproducibility Philosophy

The `main` branch contains only:

-   validated sampler settings
-   stable diagnostic workflows
-   interpretable experiment outputs

Experimental tuning work is intentionally kept out of `main` to preserve
a clean reference implementation.

------------------------------------------------------------------------

## 🧭 Possible Extensions

Ideas for future experimentation:

-   Alternative priors (Laplace, Horseshoe, ridge‑type shrinkage)
-   Hierarchical regression
-   Comparison with NUTS (Stan / PyMC) as a gold‑standard reference
-   More advanced MH tuning strategies
-   Automatic step‑size adaptation

------------------------------------------------------------------------

## 🙌 Author

This project is developed by **Sadra** as a hands‑on exploration of
Bayesian inference, MCMC sampling, and reproducible research pipelines.
