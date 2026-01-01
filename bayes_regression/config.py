from dataclasses import dataclass
from pathlib import Path


@dataclass
class Config:
    # MCMC settings
    n_chains: int = 4
    n_iter: int = 12_000
    burn_in: int = 4_000

    # Proposal scales
    beta_step: float = 0.02
    log_sigma_step: float = 0.1

    # Priors
    tau2: float = 10.0
    a0: float = 2.0
    b0: float = 1.0

    # Output folders
    results_dir: Path = Path("results")
    plots_dir: Path = Path("results") / "plots"
    tables_dir: Path = Path("results") / "tables"

    # Random seed
    seed: int = 123
