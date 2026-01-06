
# Monte‑Carlo Bayesian Regression — California Housing

This project implements a full **Bayesian linear regression** analysis of the California Housing dataset using a
**Random‑Walk Metropolis–Hastings (MCMC) sampler**. The goal of the project is to provide a transparent, educational,
and reproducible implementation of Bayesian inference where the sampling process, diagnostics, and results can be fully
inspected and replicated.

The project is based on the original research analysis and has been reorganized into a clean Python package structure,
while preserving the **same model, priors, sampler logic, and interpretation workflow**.

All analysis outputs are generated programmatically and written to:

```text
results/plots/
results/tables/
```

This ensures that anyone running the project can regenerate the exact same artifacts, inspect diagnostics, and evaluate
posterior results in a structured and reproducible way.

---

## 🎯 Project Objectives

This project aims to:

- implement Bayesian linear regression from first principles
- sample from the posterior using Random‑Walk Metropolis
- study convergence behavior across multiple chains
- evaluate sampler efficiency using diagnostics (ESS, ESS/sec, R-hat, Geweke, ACF)
- tune the proposal variance via a sensitivity experiment
- compute posterior summaries and predictive distributions
- make the full workflow reproducible and reviewable

The project is intended primarily as a **methodological and learning‑focused implementation**, rather than an automated
black‑box Bayesian modeling tool.

---

## 🧠 Statistical Model

Likelihood

\(
y \mid X, \beta, \sigma^2 \sim \mathcal N(X\beta,\ \sigma^2 I)
\)

Priors

\(
\beta \sim \mathcal N(0,\ \tau^2 I)
\)

\(
\sigma^2 \sim \text{Inv‑Gamma}(a_0,\ b_0)
\)

Sampling is performed in:

- \( \beta \) (regression coefficients)
- \( \log\sigma^2 \) (log‑variance parameter)

using a **Random‑Walk Metropolis–Hastings algorithm**.

The transformation to \(\log\sigma^2\) includes the proper Jacobian adjustment, consistent with the original formulation.

Default hyperparameters:

| Parameter | Value |
|----------|------:|
| \( \tau^2 \) | 10 |
| \( a_0 \) | 2 |
| \( b_0 \) | 1 |

These values match the original project analysis.

---

## 🧩 Repository Structure

```text
bayes_regression/
│   data.py              # dataset loading & preprocessing
│   model.py             # priors + log posterior
│   mcmc.py              # Random‑Walk Metropolis sampler
│   diagnostics.py       # ESS, split‑Rhat, Geweke, ACF
│   plots.py             # trace, running mean, ACF, tuning plots
│   predictive.py        # posterior predictive sampling
│   __init__.py
run_analysis.py          # main analysis pipeline
results/
│   plots/               # generated figures
│   tables/              # generated CSV outputs
data/                    # optional workspace for user datasets
README.md
```

The codebase is organized to separate:

- modeling and inference logic
- diagnostics and visualization
- execution pipeline
- generated outputs

while still reproducing the original computational workflow.

---

## ▶️ Running the Analysis

Clone the repository

```bash
git clone https://github.com/sadramini/monte-carlo-bayesian-regression.git
cd monte-carlo-bayesian-regression
```

(Optional) create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# or
venv\Scripts\Activate         # Windows
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the full pipeline

```bash
python run_analysis.py
```

The script will automatically:

1. load and standardize the California Housing dataset  
2. fit an OLS baseline model  
3. run multiple independent MCMC chains  
4. compute convergence diagnostics (ESS, R-hat, Geweke, ACF)  
5. compute Monte Carlo standard errors (MCSE)  
6. run a proposal variance sensitivity experiment over multiple `beta_step` values  
7. generate a tuning plot of ESS/sec vs `beta_step`  
8. generate posterior summaries  
9. evaluate posterior predictive performance  
10. export plots and tables to `results/`

No manual steps are required beyond running the script.

---

## 📊 Output Artifacts

### Tables — saved in `results/tables/`

- `posterior_summary.csv`
- `diagnostics.csv`
- `mcse.csv`
- `ess_per_second.csv`              — ESS per second for the main experiment
- `sensitivity.csv`                 — proposal variance sensitivity (beta_step grid)
- `posterior_predictive_test.csv`
- `ols_estimates.csv`

These files include:

- posterior means and credible intervals  
- effective sample size & R‑hat values  
- Geweke Z‑statistics  
- Monte‑Carlo standard errors  
- ESS per second (efficiency) for each parameter  
- efficiency comparison across different proposal scales  
- predictive interval results on the test set  

---

### Plots — saved in `results/plots/`

- traceplots for each parameter and chain
- running mean stabilization plots
- ACF plots for selected parameters
- log‑variance trace diagnostics
- `ess_per_second_vs_beta_step.png` — tuning curve: ESS/sec vs `beta_step` for each parameter

These visual diagnostics allow evaluation of:

- stationarity
- chain agreement
- autocorrelation structure
- burn‑in sufficiency
- how sampler efficiency changes with the proposal variance

---

## 🔧 MCMC Tuning & Efficiency

The sampler uses a **component-wise Random‑Walk Metropolis** scheme that updates:

- all regression coefficients \(\beta_j\) one at a time
- then the log‑variance parameter \(\log\sigma^2\)

For the main experiment, we:

- run 4 independent chains
- compute ESS and R-hat per parameter
- compute **ESS per second**, using total wall‑clock time, to measure efficiency

### ESS per second

The file `ess_per_second.csv` reports, for each parameter:

- `ESS_per_second = ESS_mean / total_sampling_time`

This is a direct measure of how many *effectively independent* draws per second the sampler produces, and makes it easy to compare efficiency across parameters.

### Proposal variance sensitivity experiment

To avoid ad‑hoc tuning, the script performs a dedicated sensitivity study over a grid of proposal scales:

```python
beta_steps = [0.005, 0.01, 0.02, 0.05]
```

For each value in this grid, the pipeline:

1. reruns the MCMC chains with that `beta_step`
2. recomputes ESS, R-hat and ESS/sec
3. stores the results in `sensitivity.csv`

This allows us to empirically study the trade‑off between:

- very small steps (slow random walk, high autocorrelation)
- very large steps (high rejection rate)
- an intermediate “sweet spot” where ESS/sec is maximized

### Tuning plot

The figure `ess_per_second_vs_beta_step.png` visualizes the sensitivity results:

- x‑axis: `beta_step`
- y‑axis: `ESS_per_second`
- one line per parameter

The resulting curves exhibit the expected **U‑shape** for Random‑Walk Metropolis:

- small `beta_step` → inefficient exploration (low ESS/sec)  
- moderate `beta_step` (≈ 0.01–0.02) → highest efficiency  
- large `beta_step` (0.05) → efficiency drops again due to more rejections  

Based on this analysis, the main experiment uses a proposal scale in the high‑efficiency region (`beta_step = 0.02`), balancing good mixing with robust convergence diagnostics.

---

## 🧪 Convergence & Reproducibility Notes

The current repository reflects the **validated version of the sampler and workflow** used in the original analysis.

Experimental modifications were explored during development, including:

- alternative burn‑in schedules  
- different proposal scales (`beta_step`)  
- additional iterations  
- block‑update proposals for \(\beta\)

After evaluating diagnostics and posterior stability, the project was intentionally restored to the configuration that:

- produced consistent and interpretable inference  
- aligned with the original methodology  
- maintained reproducibility  

The goal of the repository is to provide a **clear, faithful, and transparent implementation**, rather than maximize sampling aggressiveness or efficiency at the expense of methodological clarity.

---

## 🧭 Interpretation Summary

Posterior estimates exhibit expected behavior:

- income (`MedInc`) — strong positive association  
- rooms (`AveRooms`) — negative association  
- bedrooms (`AveBedrms`) — positive effect  
- population — weak effect  
- intercept and variance — stable across chains  

Moderate autocorrelation in some coefficients is typical for RW‑MH in correlated predictor spaces, but convergence metrics (ESS, R‑hat, Geweke) and ESS/sec indicate that inference remains valid and reasonably efficient.

---

## 📌 Reproducibility Philosophy

This project follows a research‑oriented structure:

- code is version‑controlled  
- results are generated programmatically  
- data handling is explicit and isolated  
- diagnostics are exported for inspection  

This allows:

- independent verification  
- rerunning under new conditions  
- extending analysis cleanly  

without modifying core model code.

---

## 📄 License (MIT License)

This project is distributed under the MIT License:

```text
MIT License

Copyright (c) 2026

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙌 Closing Notes

This repository was structured to make the analysis:

- clear  
- reproducible  
- interpretable  
- academically presentable  

while preserving the **original modeling decisions and inference process**.

Feedback, extensions, or replication attempts are welcome.
