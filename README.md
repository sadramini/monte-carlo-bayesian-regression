# Bayesian Linear Regression — California Housing (MCMC via Random‑Walk Metropolis)

This repository implements a full **Bayesian linear regression** analysis of the California Housing dataset using a
**component‑wise Random‑Walk Metropolis–Hastings sampler**. The project reproduces the exact modeling and computational
approach of the original research notebook, but organizes the code into a clean, reproducible Python project structure.

All estimates, diagnostics, and posterior predictive quantities are generated directly from the MCMC chains and saved
into structured `results/plots/` and `results/tables/` folders for transparency and reproducibility.

---

## ⭐ Project Goals

- Implement Bayesian linear regression with:
  - standardized predictors and intercept
  - Normal prior on regression coefficients
  - Inverse‑Gamma prior on noise variance
- Sample from the posterior via **Random‑Walk Metropolis–Hastings**
- Diagnose convergence using:
  - Traceplots
  - Running means
  - Autocorrelation functions (ACF)
  - Effective Sample Size (ESS)
  - Split‑\(\hat R\)
  - Geweke z‑scores
- Compare posterior summaries with OLS estimates
- Produce posterior predictive intervals on test data
- Save all outputs in a structured results directory

The statistical model, priors, sampler design, and diagnostics are **identical to the original project** — only the code
organization has changed.

---

## 🧠 Statistical Model

Let

\(
y \mid X, \beta, \sigma^2 \sim \mathcal N(X\beta,\ \sigma^2 I)
\)

**Priors**

\(
\beta \sim \mathcal N(0,\ \tau^2 I)
\)

\(
\sigma^2 \sim \text{Inv‑Gamma}(a_0,\ b_0)
\)

Sampling is performed in \(\beta\) and \(\log\sigma^2\) using:

- component‑wise random‑walk MH for coefficients
- scalar random‑walk MH for \(\log\sigma^2\)
- Jacobian correction for the log‑variance transformation

Default hyperparameters (from the original project):

| Parameter | Value |
|----------|------:|
| \(\tau^2\) | 10 |
| \(a_0\) | 2 |
| \(b_0\) | 1 |

---

## 🧩 Project Structure

```
bayes-california-housing/
├── bayes_regression/
│   ├── data.py              # dataset loading + preprocessing
│   ├── model.py             # priors + log posterior
│   ├── mcmc.py              # Random-Walk MH sampler
│   ├── diagnostics.py       # ESS, R-hat, Geweke, ACF
│   ├── plots.py             # trace, running mean, acf plots
│   ├── predictive.py        # posterior predictive samples
│   └── __init__.py
├── run_analysis.py          # main entrypoint to reproduce results
├── results/
│   ├── plots/               # generated figures
│   └── tables/              # csv output tables
└── README.md
```

This layout makes the project:

- reproducible
- modular
- extensible
- research‑friendly

while preserving the exact original computation pipeline.

---

## ▶️ Running the Analysis

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full pipeline:

```bash
python run_analysis.py
```

This will:

1) load & standardize data  
2) fit OLS baseline  
3) run 4 independent MCMC chains  
4) compute convergence diagnostics  
5) generate posterior summaries  
6) compute predictive intervals  
7) save all outputs to `results/`

---

## 📁 Output Artifacts

### 📊 Tables — saved in `results/tables/`

| File | Description |
|------|-------------|
| `posterior_summary.csv` | posterior means, SDs, credible intervals, \(P(\beta>0)\) |
| `diagnostics.csv` | ESS, split‑\(\hat R\), Geweke statistics |
| `mcse.csv` | Monte‑Carlo Standard Errors |
| `posterior_predictive_test.csv` | predictive means & 95% intervals |
| `ols_estimates.csv` | OLS reference coefficients |

---

### 📉 Plots — saved in `results/plots/`

- Traceplots for each parameter
- Running‑mean plots
- ACF plots for selected parameters
- \(\log\sigma^2\) convergence plots

These correspond exactly to the figures in the original project but are exported programmatically.

---

## 🧪 Convergence Assessment

Convergence is evaluated using:

- Split‑\(\hat R \approx 1.00\) across chains
- ESS (min & mean over chains)
- Geweke z‑scores
- visual trace stability
- ACF decay behavior

Where appropriate, longer burn‑in or additional iterations may be used to
increase ESS while leaving the posterior unchanged.

---

## 🎯 Interpretation Summary

Across chains, the posterior exhibits the same qualitative structure as the original work:

- Income (`MedInc`) — strong positive effect
- Rooms (`AveRooms`) — negative effect
- Bedrooms (`AveBedrms`) — positive effect
- Population — near zero effect
- Intercept — stable across runs
- Residual variance — consistent across chains

Posterior predictive intervals cover most test points appropriately,
indicating good calibration.

---

## 🔁 Reproducibility Philosophy

This project is designed to:

- preserve the mathematical formulation of the original notebook
- maintain the *exact* sampler and diagnostics
- improve transparency and structure
- separate computation from experiment outputs

Nothing in the statistical workflow has been altered — only organized.

---

## 🤝 Acknowledgements

Dataset: **California Housing — Scikit‑Learn**  
Methods: Bayesian Linear Regression + Random‑Walk Metropolis  
Diagnostics: Gelman–Rubin, ESS, Geweke

---

## 📌 Future Extensions (optional)

Potential follow‑up work:

- adaptive proposal scaling
- block‑update proposals for correlated predictors
- marginal likelihood estimation
- posterior predictive scoring metrics
- comparison with HMC / NUTS

These would extend capability without altering the current results pipeline.

---

## 📝 Citation (if used academically)

If you use or adapt this project, please cite:

> Bayesian Linear Regression with Random‑Walk Metropolis Sampling —
> California Housing Posterior Analysis (2026).

---

## 👍 Author Notes

This repository was structured to preserve the **original code, modeling decisions,
and inferential results**, while making the project cleaner and easier to share,
reproduce, and review.

