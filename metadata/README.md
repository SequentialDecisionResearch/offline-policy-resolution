# Reproducibility Metadata

The empirical pipeline records metadata needed to audit the computational experiment.

During a formal run, the program creates files including:

- `frozen_protocol.json` — frozen scientific configuration used for the experiment
- `protocol_lock.json` — protocol and source-code lock information
- `environment.json` — Python, platform, package versions, protocol hash, and script hash
- `download_*.json` — market-data download provenance and hashes
- `data_quality_*.json` — asset-level data validation information
- `logger_sanity_by_seed.csv` — behavior-policy logging and coverage diagnostics

The formal protocol uses the same features, actions, logger family, target-policy construction, OPE estimators, evidence windows, scoring design, and metrics across QQQ, SPY, and IWM.

The reconstructed full-action one-step benchmark is reserved for scoring and is not used to tune evidence windows, policy parameters, nuisance models, or asset-specific settings.

Raw market data are not redistributed in this repository.
