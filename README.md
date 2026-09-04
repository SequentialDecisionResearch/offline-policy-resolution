# Offline Policy Resolution

Reproducible research on off-policy evaluation, policy resolution, and decision-making under uncertainty.

This repository contains the replication materials associated with the research paper:

**Historical Evidence for Off-Policy Evaluation of ETF Allocation Policies:  
When Better Value Estimates Do Not Resolve Policy Choice**

## Research question

The central question is decision-based:

> When does logged historical evidence provide enough information to justify choosing a candidate policy over a baseline policy?

Off-policy evaluation (OPE) is often judged by value-estimation accuracy, action overlap, effective sample size, or other estimator diagnostics. These quantities are important, but they do not necessarily answer the deployment question.

Two policy values may be estimated reasonably well while the difference between them remains too uncertain to determine which policy should be preferred.

The main distinction studied in this project is therefore:

**value estimation is not the same as policy resolution.**

The empirical benchmark uses a common one-step contextual-bandit protocol across QQQ, SPY, and IWM. The financial setting is used as a truth-auditable test bed rather than as a proposal for a new trading strategy.

## Main empirical questions

The experiments study:

- off-policy value-estimation accuracy;
- historical evidence-window length;
- target-policy action coverage and effective sample size;
- candidate-versus-baseline policy resolution;
- bootstrap uncertainty and evidence-based abstention;
- robustness to transaction costs, reward specification, and policy contrast.

Four standard OPE estimators are included:

- Direct Method (DM);
- Inverse Propensity Scoring (IPS);
- Self-Normalized IPS (SNIPS);
- Doubly Robust estimation (DR).

The primary scientific emphasis is not on introducing a new estimator. Instead, the experiments ask whether apparently improved OPE diagnostics actually provide enough evidence to resolve a deployment decision.

## Empirical design

The study uses a common frozen protocol for:

- QQQ;
- SPY;
- IWM.

At each decision point, the action space is:

- reduce exposure;
- keep exposure unchanged;
- increase exposure.

The same feature construction, action space, behavior-logging framework, target-policy construction, OPE procedures, evidence windows, forward-scoring design, and evaluation metrics are used across all three ETFs.

A stochastic behavior policy generates partial-feedback logs. The OPE procedures receive only the reward associated with the logged action.

Under the one-step price-taking benchmark, the rewards corresponding to all available actions can be reconstructed after the next market return is observed. These reconstructed rewards are withheld from the learner and used only for evaluation.

This creates a truth-auditable one-step benchmark in which OPE operates under partial feedback while the experimenter can later evaluate estimation error and candidate-versus-baseline policy resolution.

The benchmark is deliberately one-step. It does not reconstruct the multi-period trajectory that would have resulted from a different sequence of actions.

## Historical evidence

The main historical-evidence experiment compares four pre-specified OPE windows:

- all available pre-anchor history;
- last 252 observations;
- last 126 observations;
- last 60 observations.

These windows are fixed in advance rather than selected using the withheld counterfactual benchmark.

The purpose is to examine the trade-off between temporal recency and statistical evidence.

More recent data may be more locally relevant, but shorter windows also contain less information. The experiment therefore treats historical-memory length as an empirical question rather than assuming that either more recent or more extensive history must always be preferable.

## Action coverage and effective sample size

A separate experiment studies whether improving target-policy action coverage improves OPE quality.

Recent observations are added under two mechanisms:

1. recent observations generated under the original behavior policy;
2. target-aware re-logging on the same stored contexts.

The comparison is designed to separate temporal recency from action support.

Effective sample size (ESS) is used as an overlap diagnostic, but ESS is not interpreted as evidence that the historical observations are temporally representative or that the policy decision has been resolved.

## Policy resolution

The primary deployment comparison is between a candidate policy and a baseline policy.

Let

\[
\Delta V
=
V(\pi_{\mathrm{candidate}})
-
V(\pi_{\mathrm{baseline}}).
\]

The central issue is not merely whether each policy value can be estimated accurately, but whether the evidence is sufficiently precise to determine the sign of the policy-value difference.

The experiments therefore compare the estimated candidate-baseline gap with its uncertainty using paired bootstrap procedures.

Failure to resolve the policy difference is treated as an informative outcome rather than forcing a ranking from point estimates.

In this framework, abstention can be the appropriate decision when the historical evidence is insufficient to justify switching policies.

## Main empirical finding

Across the three ETF benchmarks, standard OPE diagnostics can improve without making the candidate-versus-baseline decision resolvable.

In particular, the experiments show that:

- longer historical evidence can reduce value-estimation error;
- improved target-policy action coverage can increase effective sample size;
- these improvements do not necessarily produce a comparably large reduction in uncertainty around the candidate-baseline policy difference.

The broader lesson is therefore:

> Better value estimates do not automatically imply a better-supported policy decision.

This distinction is relevant beyond the financial benchmark to contextual bandits, reinforcement learning, recommendation systems, experimentation platforms, resource-allocation systems, and other logged decision environments.

## Repository structure

```text
offline-policy-resolution/
├── CITATION.cff
├── LICENSE
├── README.md
├── requirements.txt
│
├── src/
│   ├── offline_policy_resolution_pipeline.py
│   └── validate_offline_policy_resolution_outputs.py
│
├── data/
│   └── README.md
│
├── results/
│   ├── T01_etf_sample.csv
│   ├── T02_baseline_ope.csv
│   ├── T03_memory_comparison.csv
│   ├── T04_recency_vs_coverage.csv
│   ├── T04b_recency_coverage_full_curve.csv
│   ├── T05_policy_resolution.csv
│   ├── T06_cross_etf_helpers.csv
│   ├── T07_transaction_cost_robustness.csv
│   ├── T08_risk_reward_robustness.csv
│   └── T09_alternative_policy_contrast.csv
│
├── figures/
│   ├── F1_baseline_ope_mae_by_etf.png
│   ├── F2_memory_mae_by_etf.png
│   ├── F3_recency_coverage_IWM.png
│   ├── F3_recency_coverage_QQQ.png
│   ├── F3_recency_coverage_SPY.png
│   ├── F4_policy_gap_vs_uncertainty.png
│   └── F5_transaction_cost_robustness.png
│
├── metadata/
│   └── README.md
│
└── paper/
    ├── README.md
    ├── Historical_Evidence_for_Off_Policy_Evaluation_of_ETF_Allocation_Policies.pdf
    └── When_Better_Offline_Evaluation_Still_Cannot_Choose_the_Better_Policy.pdf

Code

The main empirical program is:

src/offline_policy_resolution_pipeline.py

It implements the common cross-ETF experimental protocol, including:

market-data acquisition and validation;
feature construction;
behavior-log simulation;
target-policy construction;
DM, IPS, SNIPS, and DR evaluation;
historical-memory experiments;
recency and action-coverage experiments;
candidate-baseline policy-resolution analysis;
bootstrap uncertainty estimation;
transaction-cost robustness;
risk-sensitive reward robustness;
alternative policy-contrast robustness;
article tables and publication figures.

The output-validation program is:

src/validate_offline_policy_resolution_outputs.py

It checks the required article tables and detailed outputs, verifies the expected ETF coverage, checks the frozen-protocol conditions, and audits key structural properties of the formal experiment.

Requirements

The replication code requires Python 3.10 or later.

Main Python dependencies are listed in requirements.txt:

NumPy;
pandas;
SciPy;
scikit-learn;
Matplotlib;
yfinance.
Data

Raw market data are not redistributed in this repository.

The empirical pipeline downloads the required historical market data programmatically.

The program records information including:

data source;
requested sample range;
realized sample range;
number of observations;
software version;
data hashes;
data-quality checks.

Raw data remain subject to the terms and conditions of their original providers.

The data/ directory contains additional information about the data-reconstruction process.

Reproducibility and protocol control

The main experiment uses a frozen common protocol across all assets.

The reconstructed full-action one-step benchmark is reserved for scoring and is not used to tune:

historical evidence windows;
target-policy parameters;
evidence budgets;
ESS thresholds;
nuisance models;
ETF-specific settings.

The pipeline records reproducibility metadata including:

frozen scientific protocol;
protocol hash;
source-code hash;
Python version;
package versions;
data-download provenance;
data-quality information;
logger sanity diagnostics.

Additional information is provided in metadata/README.md.

The public replication code and selected machine-readable article outputs are included in this repository.

Selected results

The results/ directory contains the compact machine-readable tables used to summarize the main article findings.

These include:

ETF sample and data audit;
baseline OPE estimator comparison;
historical-memory comparison;
recency versus action-coverage analysis;
full recency/coverage curves;
candidate-baseline policy-resolution results;
cross-ETF summary statistics;
transaction-cost robustness;
risk-sensitive reward robustness;
alternative policy-contrast robustness.

Large intermediate caches and raw downloaded market-data files are intentionally excluded from the repository.

Figures

The figures/ directory contains the final publication-level figures generated from the formal experiment, including:

baseline OPE estimation error;
historical evidence-window comparison;
recency and action-coverage results;
policy gap versus uncertainty;
transaction-cost robustness.
Papers and public articles

The paper/ directory contains public versions associated with this research project.

Research paper

Historical Evidence for Off-Policy Evaluation of ETF Allocation Policies:
When Better Value Estimates Do Not Resolve Policy Choice

Public-facing article

When Better Offline Evaluation Still Cannot Choose the Better Policy

The public-facing article emphasizes the broader reinforcement-learning and AI-deployment interpretation of the empirical results.

Interpretation and limitations

The experiment is a controlled empirical demonstration rather than a universal impossibility theorem.

Several limitations are important.

The full-action benchmark is one-step and conditional on the logged pre-action position. It does not reconstruct the multi-period trajectory that would have followed from an alternative sequence of actions.

The behavior logs are simulated rather than collected from a production decision system.

The target policies are data-adaptive.

QQQ, SPY, and IWM are historical market paths rather than independent draws from a general population of environments.

The finding that all available history performs best in this benchmark should not be interpreted as a general recommendation to always use all historical data. Other environments, horizons, or structural changes could favor shorter evidence windows.

Likewise, the bootstrap-based one-sided decision rule is used as an empirical decision diagnostic. It should not be interpreted as a finite-sample certified deployment-safety guarantee.

The contribution of the benchmark is therefore methodological and empirical:

historical OPE should be evaluated not only by whether policy values can be estimated, but also by whether the available evidence is strong enough to resolve the policy decision that actually matters.

Citation

If you use the code, results, or experimental design in this repository, please cite the associated research paper.

GitHub citation metadata are provided in CITATION.cff.

Permanent arXiv, SSRN, or DOI identifiers can be added once they become available.

License

Code in this repository is released under the MIT License.

Third-party market data and other external data sources remain subject to the terms, licenses, and access conditions of their original providers.
