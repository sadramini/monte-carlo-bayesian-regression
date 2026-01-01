# Data Folder — Project Notes

This folder is reserved for **local or external data files** used by the
Bayesian California Housing regression project.

The original project uses the **California Housing dataset from scikit‑learn**, 
which is downloaded programmatically and therefore is **not stored in this
repository**. The `data/` folder is provided so users may optionally place
alternative datasets here for experimentation or replication studies.

---

## 📂 Purpose of This Folder

The `data/` directory is intended for:

- locally stored datasets
- alternative model input files
- experimental or extended analysis data
- derived / preprocessed datasets (optional)

By default, the project does **not commit data files** to GitHub to avoid:

- large repository size
- licensing restrictions
- privacy concerns
- accidental sharing of sensitive files

If you place any data here, it will remain local to your machine unless
manually committed.

---

## 📌 Default Dataset (Used in This Project)

The main project uses the:

> **California Housing Dataset — Scikit‑Learn**

It is automatically loaded via:

- `sklearn.datasets.fetch_california_housing()`

This ensures:

- reproducibility
- accessibility without downloads
- no storage of binary data files

No files are required inside this folder for the main analysis to run.

---

## 🧱 Expected File Conventions (If User Adds Data)

If you add your own dataset, recommended conventions are:

- CSV or Parquet format
- clearly documented column names
- include a brief description in this folder

Example:

```
data/
├── README.md
├── custom_dataset.csv
└── metadata_notes.txt
```

---

## ⚠️ Data Privacy & Ethics Notice

Do **not** place sensitive or identifying data into this repository, including:

- personal or medical records
- proprietary datasets
- confidential organizational data

If such data is required for research, keep it local and excluded from version control.

---

## 📝 Reproducibility Notes

This project follows good research‑computing practice:

- code is version‑controlled
- results are stored under `results/`
- data is treated as an external input

This separation ensures that:

- computations remain transparent
- datasets can be changed without modifying code
- analyses are repeatable with different inputs

---

## 🤝 Contributing or Extending Data Use

If you plan to:

- add a new dataset
- rerun the Bayesian model on different features
- evaluate a new domain

please document:

1) dataset source  
2) preprocessing steps  
3) license or usage restrictions  

so results remain interpretable and reproducible.

---

## 👍 Summary

The `data/` folder is an **optional local workspace** for datasets.
The default pipeline works without any files here, but the folder exists to
support future extensions, experiments, or custom input data.

