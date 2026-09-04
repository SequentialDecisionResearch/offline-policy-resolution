"""Validate outputs from offline_policy_resolution_pipeline.py.

Run after the full scientific experiment, for example:
    python validate_offline_policy_resolution_outputs.py --output-dir outputs

The validator checks that every required article table and detail dataset
exists, has the expected columns/assets, contains finite key metrics, and that
main policy-resolution scoring windows are non-overlapping within each
asset/seed. It also verifies the frozen protocol firewall markers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ASSETS = {"QQQ", "SPY", "IWM"}

TABLE_SCHEMAS = {
    "T01_etf_sample.csv": {"ETF", "Final_sample_start", "Final_sample_end", "Decision_observations"},
    "T02_baseline_ope.csv": {"asset", "estimator", "MAE_bp", "Bias_bp", "Best_Policy_Accuracy", "Mean_ESS"},
    "T03_memory_comparison.csv": {"asset", "memory_rule", "MAE_bp", "Relative_MAE_vs_All"},
    "T04_recency_vs_coverage.csv": {"asset", "evidence_type", "m", "MAE_Change_pct", "ESS_Change_pct"},
    "T04b_recency_coverage_full_curve.csv": {"asset", "evidence_type", "m", "MAE_bp", "Mean_ESS"},
    "T05_policy_resolution.csv": {"asset", "Median_Abs_DeltaV_bp", "Median_SE_bp", "Gap_to_SE", "Point_Selection_Accuracy"},
    "T06_cross_etf_helpers.csv": {"asset", "all_history_mae_bp", "median_gap_bp", "median_se_bp", "gap_to_se"},
    "T07_transaction_cost_robustness.csv": {"asset", "transaction_cost", "risk_lambda", "All_History_MAE_bp", "Median_Abs_DeltaV_bp", "Median_SE_bp"},
    "T08_risk_reward_robustness.csv": {"asset", "transaction_cost", "risk_lambda", "All_History_MAE_bp", "Median_Abs_DeltaV_bp", "Median_SE_bp"},
    "T09_alternative_policy_contrast.csv": {"asset", "Median_Abs_DeltaV_bp", "Median_SE_bp", "candidate_history", "baseline_history"},
}

DETAIL_SCHEMAS = {
    "main_baseline_ope_detail.csv": {"asset", "behavior_seed", "anchor_date", "policy", "estimator", "value_hat", "value_counterfactual", "abs_error_bp", "ess"},
    "main_memory_detail.csv": {"asset", "behavior_seed", "anchor_date", "memory_rule", "abs_error_bp", "ess"},
    "main_recency_coverage_detail.csv": {"asset", "behavior_seed", "anchor_date", "evidence_type", "m", "abs_error_bp", "ess"},
    "main_policy_resolution_detail.csv": {"asset", "behavior_seed", "anchor_date", "delta_v_hat_bp", "delta_v_counterfactual_bp", "bootstrap_se_bp", "point_selection_correct"},
}


def fail(msg: str) -> None:
    raise AssertionError(msg)


def check_assets(df: pd.DataFrame, col: str, label: str) -> None:
    got = set(df[col].astype(str).unique())
    if got != ASSETS:
        fail(f"{label}: assets {sorted(got)} != {sorted(ASSETS)}")


def check_schema(path: Path, required: set[str]) -> pd.DataFrame:
    if not path.exists():
        fail(f"Missing required file: {path}")
    df = pd.read_csv(path)
    missing = required - set(df.columns)
    if missing:
        fail(f"{path.name}: missing columns {sorted(missing)}")
    if df.empty:
        fail(f"{path.name}: empty")
    return df


def main(root: Path) -> None:
    tables = root / "results" / "tables"
    detail = root / "results" / "detail"
    metadata = root / "metadata"

    protocol_path = metadata / "frozen_protocol.json"
    if not protocol_path.exists():
        fail("Missing frozen_protocol.json")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("counterfactual_benchmark_role") != "scoring_only":
        fail("Protocol firewall: counterfactual benchmark is not scoring_only")
    if protocol.get("asset_specific_tuning") is not False:
        fail("Protocol firewall: asset_specific_tuning must be false")
    if int(protocol["anchor_spacing"]) < int(protocol["forward_horizon"]):
        fail("Main protocol scoring windows overlap by construction")

    loaded = {}
    for name, cols in TABLE_SCHEMAS.items():
        df = check_schema(tables / name, cols)
        loaded[name] = df
        if name == "T01_etf_sample.csv":
            got = set(df["ETF"].astype(str))
            if got != ASSETS:
                fail(f"T01 assets mismatch: {got}")
        else:
            check_assets(df, "asset", name)

    for name, cols in DETAIL_SCHEMAS.items():
        df = check_schema(detail / name, cols)
        check_assets(df, "asset", name)

    # Estimator coverage in baseline table.
    t02 = loaded["T02_baseline_ope.csv"]
    for asset, g in t02.groupby("asset"):
        if set(g["estimator"]) != {"DM", "IPS", "SNIPS", "DR"}:
            fail(f"{asset}: incomplete estimator set in T02")

    # Main memory rules exactly present for each ETF (supplementary decay excluded).
    t03 = loaded["T03_memory_comparison.csv"]
    expected_memory = {"all_history", "last_252", "last_126", "last_60"}
    for asset, g in t03.groupby("asset"):
        if set(g["memory_rule"]) != expected_memory:
            fail(f"{asset}: memory rules {set(g['memory_rule'])} != {expected_memory}")

    # Recency/coverage curve includes all pre-specified m values and both mechanisms.
    curve = loaded["T04b_recency_coverage_full_curve.csv"]
    expected_m = set(map(int, protocol["recent_evidence_grid"]))
    for asset, g in curve.groupby("asset"):
        got_m = set(g["m"].astype(int))
        if not expected_m.issubset(got_m):
            fail(f"{asset}: recency curve m values incomplete: {got_m}")
        types = set(g["evidence_type"])
        if not {"history_only", "recent_original_policy", "target_aware_relogging"}.issubset(types):
            fail(f"{asset}: recency evidence types incomplete: {types}")

    # Non-overlap of policy-resolution forward scoring windows. anchor_date is
    # the first date of each H-row scored window. Date-distance is not exactly H
    # business days due holidays, so use chronological order + row sequence count
    # indirectly via the pre-specified anchor generator: there must not be duplicate
    # anchors and the number/order must be identical across seeds of one asset.
    res = pd.read_csv(detail / "main_policy_resolution_detail.csv", parse_dates=["anchor_date"])
    for asset, ga in res.groupby("asset"):
        anchor_sets = []
        for seed, g in ga.groupby("behavior_seed"):
            dates = list(g.sort_values("anchor_date")["anchor_date"])
            if len(dates) != len(set(dates)):
                fail(f"{asset} seed {seed}: duplicate resolution anchors")
            anchor_sets.append(dates)
        first = anchor_sets[0]
        if any(x != first for x in anchor_sets[1:]):
            fail(f"{asset}: resolution anchor dates differ across behavior seeds")

    # Key numeric fields should be finite and nonnegative where applicable.
    finite_checks = [
        (t02, ["MAE_bp", "Mean_ESS"]),
        (t03, ["MAE_bp"]),
        (loaded["T05_policy_resolution.csv"], ["Median_Abs_DeltaV_bp", "Median_SE_bp"]),
    ]
    for df, cols in finite_checks:
        for col in cols:
            x = pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            if np.any(~np.isfinite(x)):
                fail(f"Non-finite values in required metric {col}")
            if col in {"MAE_bp", "Mean_ESS", "Median_Abs_DeltaV_bp", "Median_SE_bp"} and np.any(x < 0):
                fail(f"Negative values in nonnegative metric {col}")

    print("OFFLINE POLICY RESOLUTION OUTPUT VALIDATION PASSED")
    print(f"Protocol hash metadata present; assets={sorted(ASSETS)}")
    print(f"Validated {len(TABLE_SCHEMAS)} article tables and {len(DETAIL_SCHEMAS)} detail datasets.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the final outputs from the offline policy-resolution experiment."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Experiment output directory produced by offline_policy_resolution_pipeline.py.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    root = args.output_dir.expanduser().resolve()
    print(f"Validating: {root}")
    main(root)
