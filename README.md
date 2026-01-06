
# Monte‑Carlo Bayesian Regression — California Housing

This project implements a complete **Bayesian linear regression workflow** for the California Housing dataset, estimated using a **Random‑Walk Metropolis–Hastings (MCMC) sampler**. The goal is not only to fit a model, but to make the full inference process **transparent, reproducible, and methodologically rigorous**, with diagnostics, efficiency analysis, and posterior evaluation performed programmatically.

All results produced by the pipeline are saved to:

```
results/plots/
results/tables/
```

so the full analysis can be replicated and inspected.

---

## 🎯 Project Goals

This project demonstrates an end‑to‑end Bayesian workflow:

- define the likelihood and priors explicitly  
- implement a component‑wise Random‑Walk Metropolis sampler  
- run multiple independent chains  
- assess convergence and stability using standard diagnostics  
- quantify sampler efficiency using ESS and ESS/sec  
- study how proposal variance affects performance  
- compute posterior summaries and predictive uncertainty

The design emphasizes **clarity, reproducibility, and interpretability** over black‑box automation.

---

## 🧠 Statistical Model

Likelihood

\[
y \mid X,\beta,\sigma^2 \sim \mathcal N(X\beta,\sigma^2 I)
\]

Priors

\[
\beta \sim \mathcal N(0,\tau^2 I),
\quad
\sigma^2 \sim \text{Inv‑Gamma}(a_0,b_0)
\]

Inference is performed in:

- regression coefficients \(\beta\)
- log‑variance parameter \(\log\sigma^2\)

using a **Random‑Walk Metropolis–Hastings sampler** with Jacobian adjustment for the log‑variance transformation.

Default hyperparameters:

| Parameter | Value |
|--------:|-----:|
| \(\tau^2\) | 10 |
| \(a_0\) | 2 |
| \(b_0\) | 1 |

---

## 📁 Repository Structure

```
bayes_regression/
│ data.py            # dataset loading & preprocessing
│ model.py           # priors + log posterior
│ mcmc.py            # Metropolis–Hastings sampler
│ diagnostics.py     # ESS, R-hat, Geweke, ACF
│ plots.py           # plotting utilities
│ predictive.py      # posterior predictive sampling
│ __init__.py
run_analysis.py      # analysis pipeline entry point
results/
  plots/
  tables/
```

---

## ▶️ Running the Analysis

```
python run_analysis.py
```

The script automatically:

1) loads and standardizes the dataset  
2) fits an OLS baseline for comparison  
3) runs **four independent MCMC chains**  
4) computes diagnostics (ESS, split‑Rhat, Geweke, ACF)  
5) computes MCSE from ESS  
6) measures total sampling time  
7) runs a **proposal‑variance sensitivity study** across several `beta_step` values  
8) summarizes posterior parameters  
9) evaluates posterior predictive performance on a held‑out test set  
10) exports all figures and tables

---

## 📊 Outputs

### Tables (saved in `results/tables/`)

- `posterior_summary.csv` — posterior means, medians, credible intervals, sign probabilities  
- `diagnostics.csv` — ESS (per parameter, across chains), R‑hat, Geweke statistics  
- `mcse.csv` — Monte‑Carlo standard error using mean‑ESS  
- `ess_per_second.csv` — efficiency of the **main experiment** (ESS / wall‑clock‑time)  
- `sensitivity.csv` — ESS, R‑hat, and ESS/sec across a grid of proposal scales  
- `posterior_predictive_test.csv` — predictive means & intervals on test data  
- `ols_estimates.csv` — OLS coefficients on standardized predictors

Together these files give a reproducible, quantitative view of posterior accuracy, chain behavior, and sampler efficiency.

---

### Plots (saved in `results/plots/`)

- parameter traceplots (per chain)
- running‑mean stabilization plots
- ACF plots for selected parameters
- tuning curve: **ESS/sec vs `beta_step`**

The tuning curve visualizes how proposal variance affects performance:

- very small steps → slow random‑walk exploration  
- very large steps → high rejection rate  
- intermediate region → maximum ESS/sec  

This confirms that the chosen proposal scale lies in the **high‑efficiency regime**.

---

## 🔧 MCMC Implementation & Diagnostics

The sampler performs:

- sequential updates of each \(\beta_j\)
- a separate update of \(\log\sigma^2\)

For each experiment, we evaluate:

- split‑Rhat (chain agreement)
- ESS (total and per‑parameter)
- Geweke early–late z‑scores
- ACF structure
- MCSE
- ESS/sec (efficiency per second)

Efficiency is computed as:

\[
\text{ESS/sec} =
\frac{\text{ESS (mean over chains)}}{\text{total sampling time}}
\]

This allows comparing tuning choices on a **computationally‑meaningful scale**.

To avoid ad‑hoc tuning, the pipeline runs a **proposal‑variance sensitivity experiment** over a grid of `beta_step` values and reports efficiency for each.

---

## 🧭 Interpretation Summary

Across chains we observe:

- R‑hat ≈ 1 → good chain agreement  
- strong ESS for most parameters  
- slower but valid mixing for correlated predictors  
- stable posterior means and intervals  
- predictive intervals consistent with model uncertainty  

The tuning experiment exhibits the expected RW‑MH pattern:

- inefficient at very small step sizes  
- optimal efficiency near the chosen proposal scale  
- efficiency drop at overly‑large steps  

which supports the final sampler configuration.

---

## 🧑‍🔬 Reproducibility Philosophy

The project is structured so that:

- all results are generated automatically
- no manual tuning steps are hidden
- diagnostics are exportable and reviewable
- experiments can be rerun under new settings

The focus is on **methodological transparency** rather than automation.

---

## 📄 License

MIT License — see repository file for details.

---

If you extend or adapt this workflow for new datasets or models, contributions and replication results are welcome.
