# Monte Carlo Bayesian Regression — California Housing

This project implements Bayesian linear regression for the California Housing dataset using a
**Random‑Walk Metropolis–Hastings (MCMC) sampler**, with full diagnostics and reproducible outputs.

The current version of the repository reflects the **original, validated version of the project code**, after reverting
experimental tuning changes that did not improve convergence or ESS performance. The project now prioritizes:

- correctness and reproducibility
- faithfulness to the original implementation
- transparency of diagnostics and results

All results are generated programmatically and written to:

```
results/plots/
results/tables/
```

so users can review, reproduce, and interpret the full posterior analysis.

---

## 🧠 Project Summary

This project performs:

- Bayesian linear regression with standardized predictors
- Gaussian likelihood
- Normal prior on coefficients
- Inverse‑Gamma prior on variance
- MCMC sampling using component‑wise Random‑Walk Metropolis
- Four independent chains
- Full posterior diagnostics
- Posterior predictive evaluation

The intent of the project is **educational and methodological** — to explicitly implement and study MCMC sampling behavior,
rather than rely on black‑box Bayesian software.

---

## 📦 Repository Structure

```
bayes_regression/
│   data.py              # dataset loading & preprocessing
│   model.py             # priors + log posterior
│   mcmc.py              # Random‑Walk Metropolis sampler
│   diagnostics.py       # ESS, R‑hat, Geweke, ACF
│   plots.py             # trace, running mean, ACF plots
│   predictive.py        # posterior predictive sampling
│   __init__.py
run_analysis.py          # main pipeline script
results/
│   plots/               # generated figures
│   tables/              # generated CSV outputs
data/                    # optional user dataset storage
README.md
```

The statistical implementation matches the original research notebook, but organized into a modular Python project.

---

## ▶️ How to Run the Project

Clone the repository:

```bash
git clone https://github.com/sadramini/monte-carlo-bayesian-regression.git
cd monte-carlo-bayesian-regression
```

(Optional) create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate       # macOS / Linux
# or
venv\Scripts\Activate          # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis:

```bash
python run_analysis.py
```

This will automatically:

1. load & standardize the dataset  
2. run multiple MCMC chains  
3. compute posterior diagnostics  
4. export trace and ACF plots  
5. compute posterior predictive distributions  
6. save outputs to `results/`

---

## 📊 Generated Outputs

### Results Tables (`results/tables/`)

- `posterior_summary.csv`
- `diagnostics.csv`
- `mcse.csv`
- `posterior_predictive_test.csv`
- `ols_estimates.csv`

### Plots (`results/plots/`)

- traceplots for each parameter
- running means per chain
- ACF curves
- log‑variance diagnostics

These correspond to the analysis workflow in the original project.

---

## 🔎 About Refinements & Revisions

During development, several sampler refinements were tested:

- increased iterations
- alternative burn‑in schedules
- adjusted proposal scales
- block‑update proposals for β

After evaluation, these modifications **did not improve convergence quality or ESS performance**
for this specific model and dataset.

To preserve:

- interpretability
- correctness
- reproducibility
- comparability with the original results

the project has been restored to the **validated, original sampler configuration**.

The repository now reflects:

> the final and recommended version of the analysis

Further tuning experiments may be explored in the future, but are intentionally excluded from the main branch
to keep the primary pipeline stable and consistent.

---

## 🧪 Convergence Diagnostics

The project reports:

- split‑R̂
- ESS (min & mean across chains)
- Geweke z‑scores
- autocorrelation decay
- trace and running mean stability

These diagnostics confirmed that:

- chains reach stationarity
- parameters are well‑identified
- posterior estimates are stable
- predictive behavior is reasonable

Some parameters mix more slowly (expected with RW‑MH), but performance remains acceptable for inference.

---

## 📌 Reproducibility Philosophy

This project is structured to:

- separate **code**, **results**, and **data**
- expose every computational step
- encourage transparent MCMC analysis
- make replication straightforward

Nothing in the statistical model has been altered — only reorganized for clarity and maintainability.

---

## 📄 License (MIT License)

This project is released under the MIT License:

```
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

If you use this work in academic or research contexts, citation is appreciated.

---

## 🤝 Acknowledgements

Dataset: California Housing — Scikit‑Learn  
Methods: Bayesian Linear Regression + MCMC  
Diagnostics: R̂, ESS, Geweke, ACF

---

## 🙌 Author Notes

This repository was organized to preserve the **original analysis and results**
while making the project cleaner, easier to reproduce, and suitable for sharing publicly.

Suggestions, critique, and extension ideas are always welcome.
