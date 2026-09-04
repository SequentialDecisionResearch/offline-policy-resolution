# Offline Policy Resolution

Reproducible research on off-policy evaluation, historical evidence, and policy resolution under uncertainty.

This repository contains the replication materials associated with the research paper:

**Historical Evidence for Off-Policy Evaluation of ETF Allocation Policies:  
When Better Value Estimates Do Not Resolve Policy Choice**

The main question is not only whether an off-policy estimator can estimate policy value accurately, but whether the available historical evidence is strong enough to resolve an actual candidate-versus-baseline policy decision.

The experiments use a common contextual-bandit protocol across QQQ, SPY, and IWM and study:

- off-policy evaluation accuracy;
- historical evidence-window length;
- target-policy action coverage and effective sample size;
- candidate-versus-baseline policy resolution;
- bootstrap uncertainty;
- transaction-cost and reward-specification robustness.

## Repository structure

- `src/` — empirical pipeline and validation code
- `data/` — data-source and reconstruction information
- `results/` — selected machine-readable results used in the paper
- `figures/` — final publication figures
- `paper/` — manuscript and public article versions
- `requirements.txt` — Python dependencies

## Data

Raw market data are not redistributed in this repository.

The empirical pipeline downloads the required historical data programmatically and records data provenance, sample information, software versions, and hashes where appropriate.

## Reproducibility

The main experiment uses a frozen common protocol across all assets. Counterfactual full-action outcomes are reserved for scoring and are not used for tuning the target policies, evidence windows, or OPE procedures.

Detailed execution instructions will be provided together with the public replication code.

## License

Code in this repository is released under the MIT License.

Third-party data remain subject to the terms and licenses of their original providers.
