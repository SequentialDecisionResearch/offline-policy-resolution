"""
Offline Policy Resolution + Off-Policy Evaluation empirical pipeline
===================================================================

Target paper
------------
    Historical Evidence for Off-Policy Evaluation of ETF Allocation Policies:
    When Better Value Estimates Do Not Resolve Policy Choice

This program implements the pre-specified cross-ETF protocol for QQQ, SPY,
and IWM used in the accompanying research paper.

MAIN SCIENTIFIC RULES
---------------------
1. QQQ, SPY, and IWM use the SAME features, actions, logger family, target-policy
   construction, OPE estimators, memory rules, anchor design, and metrics.
2. The reconstructed one-step counterfactual benchmark is SCORING ONLY. It is
   never used to select memory windows, temperatures, evidence budgets, ESS
   thresholds, nuisance models, or ETF-specific settings.
3. Main forward scoring windows are non-overlapping (H=80, spacing=80).
4. DR/DM nuisance predictions are time-ordered and out-of-sample: each prediction
   block is produced by a reward model trained strictly before that block.
5. The primary article outcomes are evaluation quality and policy-comparison
   resolution, NOT cumulative trading profit.

WHAT THE FULL RUN PRODUCES
--------------------------
Main article tables:
  T01 ETF sample/data audit
  T02 baseline OPE accuracy across ETFs (DM/IPS/SNIPS/DR)
  T03 historical-memory comparison (all / 252 / 126 / 60)
  T04 recency versus target-policy action coverage
  T05 candidate-baseline policy gap versus uncertainty
  T06 cross-ETF numerical summary helpers
  T07 transaction-cost robustness
  T08 risk-sensitive reward robustness
  T09 alternative target-policy contrast robustness

Machine-readable detail files are also saved for every experiment, plus logger
sanity checks, protocol/environment metadata, hashes, and publication figures.
Optional supplementary functions can generate an exploration/evaluability sweep
and a synthetic LCB stress test, but they are disabled by default because they
are outside the core paper.

DEPENDENCIES
------------
Python 3.10+
    numpy, pandas, scipy, scikit-learn, matplotlib, yfinance

Example
-------
    python offline_policy_resolution_pipeline.py --output-dir ./outputs --mode full

For a timing-only test (scientific outputs must NOT be interpreted):
    python offline_policy_resolution_pipeline.py --output-dir ./outputs --mode pilot

The pipeline is resumable: expensive behavior logs and rolling nuisance
predictions are cached by asset, seed, and reward specification.
"""





import argparse
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


# =============================================================================
# 1. Frozen journal protocol
# =============================================================================

ACTIONS = np.array([-1, 0, 1], dtype=int)
ACTION_TO_INDEX = {-1: 0, 0: 1, 1: 2}
INDEX_TO_ACTION = {0: -1, 1: 0, 2: 1}

MARKET_FEATURES = ["ret_1", "ret_5", "ret_20", "vol_20", "ma_spread", "drawdown"]
MODEL_FEATURES = [
    "intercept",
    "ret_1",
    "ret_5",
    "ret_20",
    "vol_20",
    "ma_spread",
    "drawdown",
    "position_before",
]
QHAT_COLUMNS = ["qhat_m1", "qhat_0", "qhat_p1"]
CF_COLUMNS = ["reward_cf_m1", "reward_cf_0", "reward_cf_p1"]


@dataclass(frozen=True)
class Protocol:
    # Assets/data. A 2000 start keeps the design comparable while respecting
    # IWM's shorter history; actual final start is recorded asset by asset.
    assets: Tuple[str, ...] = ("QQQ", "SPY", "IWM")
    start_date: str = "2000-01-01"
    end_date: Optional[str] = None
    require_adjusted_close: bool = True

    # Position/reward.
    initial_position: float = 0.50
    position_step: float = 0.10
    min_position: float = 0.0
    max_position: float = 1.0
    baseline_cost: float = 0.0005
    cost_grid: Tuple[float, ...] = (0.0, 0.0005, 0.0010)

    # One pre-specified risk-sensitive alternative.  With daily vol around 1%,
    # lambda=.5 creates about 0.5bp/day penalty at full exposure, comparable in
    # scale to the 0.5bp turnover charge for a 10-point exposure change.
    risk_lambda: float = 0.50

    # Behavior policy: linear Thompson sampling.
    prior_precision: float = 1.0
    assumed_reward_noise_std: float = 0.01
    ts_mc_draws: int = 400
    logging_probability_floor: float = 0.03

    # Target policy learner.
    ridge_alpha: float = 2.0
    baseline_policy_history: int = 756
    candidate_policy_history: int = 126
    alternative_candidate_history: int = 252
    base_temperature: float = 1.50
    candidate_temperature: float = 1.50
    q_scale_floor: float = 0.00025
    target_policy_probability_floor: float = 0.02
    conservative_mix_weight: float = 0.20

    # Main evidence-memory rules. No oracle-selected best window.
    memory_windows: Tuple[int, ...] = (252, 126, 60)
    include_fixed_decay_supplement: bool = True
    fixed_decay_half_life: int = 126

    # Walk-forward design.
    min_history: int = 1260
    forward_horizon: int = 80
    anchor_spacing: int = 80
    static_eval_window: int = 80

    # Time-ordered nuisance estimation used by DM/DR.
    nuisance_warmup: int = 252
    nuisance_prediction_block: int = 63

    # Recency-vs-coverage experiment. This is pre-specified; no benchmark-based
    # choice of m. The common table can summarize m=80 while the full curve is
    # always saved.
    recent_evidence_grid: Tuple[int, ...] = (0, 20, 40, 80)
    target_aware_mix: float = 0.20

    # Inference.
    bootstrap_resamples: int = 500
    bootstrap_block_length: int = 20
    confidence_level: float = 0.95

    # Repeated logging. 20 is the minimum planned journal run in the placeholder.
    behavior_seed_count: int = 20
    base_seed: int = 20260826

    # Optional supplements, disabled by default in full main run.
    run_exploration_supplement: bool = False
    exploration_epsilons: Tuple[float, ...] = (0.01, 0.03, 0.05, 0.10)
    exploration_seed_count: int = 10
    run_synthetic_lcb_supplement: bool = False
    synthetic_lcb_repeats: int = 500

    # Reporting/caching.
    save_figures: bool = True
    show_figures: bool = False
    force_download: bool = False
    force_recompute_logs: bool = False
    force_recompute_qhat: bool = False

    # Explicit protocol firewall markers.
    counterfactual_benchmark_role: str = "scoring_only"
    asset_specific_tuning: bool = False
    # Target policies are fitted from the same pre-anchor historical log that can
    # also contribute OPE evidence. This is therefore a data-adaptive-policy design.
    target_policy_design: str = "data_adaptive_pre_anchor_history"


# =============================================================================
# 2. Utilities, paths, metadata, hashes
# =============================================================================


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_package_version(package: str) -> str:
    try:
        return importlib_metadata.version(package)
    except importlib_metadata.PackageNotFoundError:
        return "not-installed"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_dict(cfg: Protocol) -> dict:
    d = asdict(cfg)
    # Run-control flags are not scientific parameters; retain them in metadata
    # but exclude from the scientific protocol hash.
    return d


def scientific_protocol_dict(cfg: Protocol) -> dict:
    d = asdict(cfg).copy()
    for k in [
        "save_figures",
        "show_figures",
        "force_download",
        "force_recompute_logs",
        "force_recompute_qhat",
        "run_exploration_supplement",
        "run_synthetic_lcb_supplement",
    ]:
        d.pop(k, None)
    return d


def protocol_hash(cfg: Protocol) -> str:
    payload = json.dumps(scientific_protocol_dict(cfg), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def stable_asset_salt(asset: str) -> int:
    return int(hashlib.sha256(asset.encode("utf-8")).hexdigest()[:8], 16)


def deterministic_rng(base_seed: int, asset: str, seed_index: int, salt: int = 0) -> np.random.Generator:
    seed = (
        int(base_seed)
        + 1000003 * int(seed_index)
        + stable_asset_salt(asset)
        + 7919 * int(salt)
    ) % (2**32 - 1)
    return np.random.default_rng(seed)


def action_index(action: int) -> int:
    return ACTION_TO_INDEX[int(action)]


def clip_position(x: float, cfg: Protocol) -> float:
    return float(np.clip(x, cfg.min_position, cfg.max_position))


def stable_softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-12)
    z = np.asarray(scores, dtype=float) / temp
    z -= np.max(z, axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=-1, keepdims=True)


def apply_probability_floor(probs: np.ndarray, epsilon: float) -> np.ndarray:
    p = np.asarray(probs, dtype=float)
    k = p.shape[-1]
    if not (0.0 <= epsilon < 1.0 / k):
        raise ValueError(f"epsilon must be in [0,{1.0/k})")
    out = (1.0 - k * epsilon) * p + epsilon
    return out / np.sum(out, axis=-1, keepdims=True)


def model_matrix(df: pd.DataFrame) -> np.ndarray:
    return df[MODEL_FEATURES].to_numpy(dtype=float)


def observed_action_indices(df: pd.DataFrame) -> np.ndarray:
    return np.array([action_index(a) for a in df["action"].to_numpy(dtype=int)], dtype=int)


def ensure_dirs(root: Path) -> Dict[str, Path]:
    paths = {
        "root": root,
        "raw": root / "data" / "raw",
        "processed": root / "data" / "processed",
        "metadata": root / "metadata",
        "cache_logs": root / "cache" / "behavior_logs",
        "cache_qhat": root / "cache" / "qhat",
        "detail": root / "results" / "detail",
        "tables": root / "results" / "tables",
        "figures": root / "results" / "figures",
        "supplement": root / "results" / "supplement",
        "logs": root / "results" / "run_logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def save_environment_metadata(cfg: Protocol, paths: Mapping[str, Path], script_path: Path) -> None:
    """Save the current software/environment snapshot.

    This function intentionally does NOT overwrite the frozen scientific protocol.
    Formal protocol locking is handled separately by ``enforce_protocol_lock``.
    """
    meta = {
        "created_utc": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": safe_package_version("numpy"),
        "pandas": safe_package_version("pandas"),
        "scipy": safe_package_version("scipy"),
        "scikit_learn": safe_package_version("scikit-learn"),
        "matplotlib": safe_package_version("matplotlib"),
        "yfinance": safe_package_version("yfinance"),
        "protocol_hash": protocol_hash(cfg),
        "script_sha256": sha256_file(script_path) if script_path.exists() else None,
    }
    (paths["metadata"] / "environment.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _scientific_hash_from_saved_protocol(saved: Mapping[str, object]) -> str:
    d = dict(saved)
    for k in [
        "save_figures",
        "show_figures",
        "force_download",
        "force_recompute_logs",
        "force_recompute_qhat",
        "run_exploration_supplement",
        "run_synthetic_lcb_supplement",
    ]:
        d.pop(k, None)
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def enforce_protocol_lock(
    cfg: Protocol,
    paths: Mapping[str, Path],
    script_path: Path,
    mode: str,
) -> None:
    """Create/check a write-once formal-run lock.

    Download and timing-only pilot runs do not create a lock.  ``main``, ``full``
    and ``robustness`` create it once and thereafter refuse scientific-parameter
    or source-code changes in the same output directory.
    """
    if mode not in {"main", "full", "robustness"}:
        return

    frozen_path = paths["metadata"] / "frozen_protocol.json"
    lock_path = paths["metadata"] / "protocol_lock.json"
    current_hash = protocol_hash(cfg)
    current_script_sha = sha256_file(script_path) if script_path.exists() else None

    if frozen_path.exists():
        saved = json.loads(frozen_path.read_text(encoding="utf-8"))
        saved_hash = _scientific_hash_from_saved_protocol(saved)
        if saved_hash != current_hash:
            raise RuntimeError(
                "Frozen protocol mismatch in this output directory. "
                f"saved={saved_hash}, current={current_hash}. "
                "Do not mix formal results from different scientific protocols; "
                "use a new output directory."
            )
    else:
        tmp = frozen_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(protocol_dict(cfg), indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        tmp.replace(frozen_path)

    if lock_path.exists():
        lock = json.loads(lock_path.read_text(encoding="utf-8"))
        if str(lock.get("protocol_hash")) != current_hash:
            raise RuntimeError(
                "protocol_lock.json does not match the current scientific protocol. "
                "Use the original code/protocol or a new output directory."
            )
        locked_script = lock.get("script_sha256")
        if locked_script and current_script_sha and str(locked_script) != current_script_sha:
            raise RuntimeError(
                "The pipeline source file changed after the formal protocol was locked. "
                "For a paper run, restore the locked source or start a new output directory."
            )
    else:
        lock = {
            "created_utc": utc_now(),
            "protocol_hash": current_hash,
            "script_sha256": current_script_sha,
            "mode_at_lock": mode,
            "target_policy_design": cfg.target_policy_design,
        }
        tmp = lock_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(lock, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(lock_path)


def validate_protocol(cfg: Protocol) -> None:
    if cfg.counterfactual_benchmark_role != "scoring_only":
        raise ValueError("Counterfactual benchmark must be scoring_only.")
    if cfg.asset_specific_tuning:
        raise ValueError("ETF-specific tuning is forbidden in the journal protocol.")
    if cfg.target_policy_design != "data_adaptive_pre_anchor_history":
        raise ValueError("Target-policy design must remain the pre-specified data-adaptive design.")
    if not math.isclose(cfg.base_temperature, cfg.candidate_temperature, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(
            "Final journal protocol requires the same softmax temperature for "
            "baseline and candidate target policies."
        )
    if cfg.anchor_spacing < cfg.forward_horizon:
        raise ValueError("Main scoring windows must be non-overlapping.")
    if len(set(cfg.assets)) != len(cfg.assets):
        raise ValueError("Duplicate ETF symbols in protocol.")
    if max(cfg.recent_evidence_grid) >= cfg.min_history:
        raise ValueError("Recent evidence pool is unreasonably large relative to min_history.")
    if cfg.nuisance_warmup < 100:
        raise ValueError("Nuisance warmup is too small for this design.")
    if cfg.bootstrap_block_length > cfg.forward_horizon:
        # Bootstrap is on historical OPE evidence, so this is not mathematically
        # forbidden, but a month-like block is the intended pre-specified scale.
        raise ValueError("Bootstrap block length should not exceed the forward horizon.")


# =============================================================================
# 3. Data acquisition and common ETF feature pipeline
# =============================================================================


def import_yfinance():
    try:
        import yfinance as yf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "yfinance is required. Install it in the SAME Python environment with: pip install yfinance"
        ) from exc
    return yf


def flatten_yfinance_columns(raw: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if not isinstance(raw.columns, pd.MultiIndex):
        return raw
    # yfinance versions differ in MultiIndex order. Prefer selecting ticker level.
    for level in range(raw.columns.nlevels):
        vals = raw.columns.get_level_values(level).astype(str)
        if symbol in set(vals):
            try:
                return raw.xs(symbol, axis=1, level=level)
            except Exception:
                pass
    out = raw.copy()
    out.columns = [str(c[0]) for c in raw.columns]
    return out


def select_adjusted_close(raw: pd.DataFrame, require_adjusted: bool = True) -> Tuple[pd.Series, str]:
    for col in ["Adj Close", "adj_close", "adjusted_close", "Adjusted Close"]:
        if col in raw.columns:
            s = pd.to_numeric(raw[col], errors="coerce")
            s.name = "price"
            return s, col
    if not require_adjusted:
        for col in ["Close", "close"]:
            if col in raw.columns:
                s = pd.to_numeric(raw[col], errors="coerce")
                s.name = "price"
                return s, col
    raise KeyError(
        "Adjusted close not found. The program refuses to silently change the reward definition. "
        f"Columns={list(raw.columns)}"
    )


def download_asset_yfinance(asset: str, cfg: Protocol, paths: Mapping[str, Path]) -> pd.DataFrame:
    raw_path = paths["raw"] / f"{asset.lower()}_yfinance.csv"
    meta_path = paths["metadata"] / f"download_{asset.lower()}.json"

    if raw_path.exists() and not cfg.force_download:
        raw = pd.read_csv(raw_path, index_col=0, parse_dates=True)
        raw.index = pd.to_datetime(raw.index).tz_localize(None)
        raw.index.name = "date"
        return raw.sort_index()

    yf = import_yfinance()
    kwargs = dict(
        tickers=asset,
        start=cfg.start_date,
        end=cfg.end_date,
        auto_adjust=False,
        actions=True,
        progress=False,
        threads=False,
        timeout=30,
    )
    try:
        raw = yf.download(**kwargs, multi_level_index=False)
    except TypeError:
        raw = yf.download(**kwargs)
    if raw is None or raw.empty:
        raise RuntimeError(f"Empty download for {asset}. Check internet/yfinance.")

    raw = flatten_yfinance_columns(raw, asset).sort_index()
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw.index.name = "date"
    raw.to_csv(raw_path)

    meta = {
        "download_utc": utc_now(),
        "asset": asset,
        "source": "Yahoo Finance via yfinance",
        "start_requested": cfg.start_date,
        "end_requested": cfg.end_date,
        "rows": int(len(raw)),
        "columns": [str(c) for c in raw.columns],
        "sha256": sha256_file(raw_path),
        "yfinance_version": safe_package_version("yfinance"),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return raw


def validate_raw_market_data(asset: str, raw: pd.DataFrame, cfg: Protocol) -> Dict[str, object]:
    if raw.empty:
        raise ValueError(f"{asset}: raw data empty")
    idx = pd.to_datetime(raw.index)
    if idx.has_duplicates:
        raise ValueError(f"{asset}: duplicate dates")
    if not idx.is_monotonic_increasing:
        raise ValueError(f"{asset}: dates not ascending")
    price, col = select_adjusted_close(raw, cfg.require_adjusted_close)
    arr = price.to_numpy(dtype=float)
    finite = np.isfinite(arr)
    if finite.sum() < 1000:
        raise ValueError(f"{asset}: too few usable adjusted-close rows")
    if np.any(arr[finite] <= 0):
        raise ValueError(f"{asset}: non-positive adjusted close")
    return {
        "asset": asset,
        "rows_raw": int(len(raw)),
        "raw_start": str(idx.min().date()),
        "raw_end": str(idx.max().date()),
        "price_column": col,
    }


def build_market_features(asset: str, raw: pd.DataFrame, cfg: Protocol, paths: Mapping[str, Path]) -> pd.DataFrame:
    quality = validate_raw_market_data(asset, raw, cfg)
    price, price_col = select_adjusted_close(raw, cfg.require_adjusted_close)
    price = price.dropna().sort_index()

    df = pd.DataFrame(index=price.index)
    df["price"] = price
    daily_ret = price.pct_change()
    df["ret_1"] = daily_ret
    df["ret_5"] = price.pct_change(5)
    df["ret_20"] = price.pct_change(20)
    df["vol_20"] = daily_ret.rolling(20).std(ddof=1)
    ma5 = price.rolling(5).mean()
    ma20 = price.rolling(20).mean()
    df["ma_spread"] = (ma5 - ma20) / ma20
    df["drawdown"] = price / price.cummax() - 1.0
    # Never include next_return in MODEL_FEATURES.
    df["next_return"] = price.shift(-1) / price - 1.0
    df = df.replace([np.inf, -np.inf], np.nan).dropna().copy()

    required = cfg.min_history + cfg.forward_horizon + max(cfg.recent_evidence_grid)
    if len(df) <= required:
        raise ValueError(f"{asset}: only {len(df)} processed rows; need > {required}")

    out = paths["processed"] / f"{asset.lower()}_features.csv"
    df.to_csv(out)
    quality.update(
        {
            "price_column": price_col,
            "rows_processed": int(len(df)),
            "processed_start": str(df.index.min().date()),
            "processed_end": str(df.index.max().date()),
            "processed_sha256": sha256_file(out),
            "model_features": MODEL_FEATURES,
            "next_return_in_model_features": False,
        }
    )
    (paths["metadata"] / f"data_quality_{asset.lower()}.json").write_text(
        json.dumps(quality, indent=2), encoding="utf-8"
    )
    return df


def load_or_build_all_assets(cfg: Protocol, paths: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    out: Dict[str, pd.DataFrame] = {}
    for asset in cfg.assets:
        raw = download_asset_yfinance(asset, cfg, paths)
        out[asset] = build_market_features(asset, raw, cfg, paths)
    return out


# =============================================================================
# 4. Behavior logger and physical counterfactual firewall
# =============================================================================


class OnlineBayesianLinearModel:
    """Gaussian Bayesian linear regression with rank-one inverse updates."""

    def __init__(self, dimension: int, prior_precision: float, noise_std: float) -> None:
        self.dimension = int(dimension)
        self.noise_var = max(float(noise_std) ** 2, 1e-12)
        self.cov = np.eye(self.dimension, dtype=float) / float(prior_precision)
        self.b = np.zeros(self.dimension, dtype=float)

    def mean(self) -> np.ndarray:
        return self.cov @ self.b

    def sample_theta(self, rng: np.random.Generator, n: int) -> np.ndarray:
        cov = (self.cov + self.cov.T) / 2.0 + 1e-12 * np.eye(self.dimension)
        return rng.multivariate_normal(self.mean(), cov, size=n, check_valid="ignore")

    def update(self, x: np.ndarray, reward: float) -> None:
        x = np.asarray(x, dtype=float)
        # Precision update A <- A + x x' / sigma^2 implemented on A^{-1}.
        u = x / math.sqrt(self.noise_var)
        cu = self.cov @ u
        denom = 1.0 + float(u @ cu)
        self.cov = self.cov - np.outer(cu, cu) / max(denom, 1e-12)
        self.cov = (self.cov + self.cov.T) / 2.0
        self.b += x * float(reward) / self.noise_var


class LinearThompsonLogger:
    def __init__(self, cfg: Protocol) -> None:
        self.cfg = cfg
        self.models = {
            int(a): OnlineBayesianLinearModel(
                len(MODEL_FEATURES), cfg.prior_precision, cfg.assumed_reward_noise_std
            )
            for a in ACTIONS
        }

    def probabilities(
        self,
        x: np.ndarray,
        rng: np.random.Generator,
        epsilon: Optional[float] = None,
        mc_draws: Optional[int] = None,
    ) -> np.ndarray:
        draws = int(mc_draws or self.cfg.ts_mc_draws)
        eps = self.cfg.logging_probability_floor if epsilon is None else float(epsilon)
        scores = np.empty((draws, len(ACTIONS)), dtype=float)
        for j, a in enumerate(ACTIONS):
            theta = self.models[int(a)].sample_theta(rng, draws)
            scores[:, j] = theta @ x
        winners = np.argmax(scores, axis=1)
        counts = np.bincount(winners, minlength=len(ACTIONS)).astype(float)
        raw = counts / counts.sum()
        return apply_probability_floor(raw, eps)

    def sample_action(
        self, x: np.ndarray, rng: np.random.Generator, epsilon: Optional[float] = None
    ) -> Tuple[int, np.ndarray]:
        p = self.probabilities(x, rng, epsilon=epsilon)
        idx = int(rng.choice(len(ACTIONS), p=p))
        return int(ACTIONS[idx]), p

    def update(self, x: np.ndarray, action: int, reward: float) -> None:
        self.models[int(action)].update(x, reward)


def context_vector(row: pd.Series, position_before: float) -> np.ndarray:
    return np.array(
        [
            1.0,
            float(row["ret_1"]),
            float(row["ret_5"]),
            float(row["ret_20"]),
            float(row["vol_20"]),
            float(row["ma_spread"]),
            float(row["drawdown"]),
            float(position_before),
        ],
        dtype=float,
    )


def one_step_reward_vector(
    market_row: pd.Series,
    position_before: float,
    cfg: Protocol,
    transaction_cost: float,
    risk_lambda: float,
) -> Tuple[np.ndarray, np.ndarray]:
    rewards: List[float] = []
    positions: List[float] = []
    for action in ACTIONS:
        pa = clip_position(position_before + cfg.position_step * int(action), cfg)
        turnover = abs(pa - position_before)
        reward = (
            pa * float(market_row["next_return"])
            - float(transaction_cost) * turnover
            - float(risk_lambda) * (pa**2) * (float(market_row["vol_20"]) ** 2)
        )
        rewards.append(float(reward))
        positions.append(pa)
    return np.asarray(rewards, dtype=float), np.asarray(positions, dtype=float)


def reward_tag(transaction_cost: float, risk_lambda: float) -> str:
    return f"c{transaction_cost:.6f}_lam{risk_lambda:.6f}".replace(".", "p").replace("-", "m")


def behavior_cache_paths(
    paths: Mapping[str, Path], asset: str, seed_index: int, transaction_cost: float, risk_lambda: float
) -> Tuple[Path, Path]:
    tag = reward_tag(transaction_cost, risk_lambda)
    base = f"{asset}_{tag}_seed{seed_index:03d}"
    return paths["cache_logs"] / f"{base}_log.csv", paths["cache_logs"] / f"{base}_score.csv"


def simulate_behavior_log(
    asset: str,
    market: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    paths: Mapping[str, Path],
    transaction_cost: Optional[float] = None,
    risk_lambda: float = 0.0,
    epsilon: Optional[float] = None,
    cache: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate partial-feedback log + separate scoring benchmark.

    The returned ``log`` contains NO unchosen rewards.  The returned ``score``
    contains the full one-step counterfactual vector and is never passed to
    target-policy fitting or OPE estimation.
    """
    c = cfg.baseline_cost if transaction_cost is None else float(transaction_cost)
    log_path, score_path = behavior_cache_paths(paths, asset, seed_index, c, risk_lambda)
    if cache and log_path.exists() and score_path.exists() and not cfg.force_recompute_logs:
        log = pd.read_csv(log_path, index_col=0, parse_dates=True).sort_index()
        score = pd.read_csv(score_path, index_col=0, parse_dates=True).sort_index()
        return log, score

    rng = deterministic_rng(cfg.base_seed, asset, seed_index, salt=int(round(c * 1e6)) + int(round(risk_lambda * 1000)))
    logger = LinearThompsonLogger(cfg)
    position = cfg.initial_position
    log_rows: List[Dict[str, object]] = []
    score_rows: List[Dict[str, object]] = []

    for date, row in market.iterrows():
        x = context_vector(row, position)
        rewards, positions = one_step_reward_vector(row, position, cfg, c, risk_lambda)
        action, probs = logger.sample_action(x, rng, epsilon=epsilon)
        idx = action_index(action)
        observed = float(rewards[idx])
        position_after = float(positions[idx])

        log_rows.append(
            {
                "date": pd.Timestamp(date),
                "intercept": 1.0,
                "ret_1": float(row["ret_1"]),
                "ret_5": float(row["ret_5"]),
                "ret_20": float(row["ret_20"]),
                "vol_20": float(row["vol_20"]),
                "ma_spread": float(row["ma_spread"]),
                "drawdown": float(row["drawdown"]),
                "position_before": float(position),
                "action": int(action),
                "behavior_prob": float(probs[idx]),
                "behavior_prob_m1": float(probs[0]),
                "behavior_prob_0": float(probs[1]),
                "behavior_prob_p1": float(probs[2]),
                "reward_observed": observed,
                "position_after": position_after,
            }
        )
        score_rows.append(
            {
                "date": pd.Timestamp(date),
                "reward_cf_m1": float(rewards[0]),
                "reward_cf_0": float(rewards[1]),
                "reward_cf_p1": float(rewards[2]),
                "next_return": float(row["next_return"]),
            }
        )
        logger.update(x, action, observed)
        position = position_after

    log = pd.DataFrame(log_rows).set_index("date").sort_index()
    score = pd.DataFrame(score_rows).set_index("date").sort_index()
    run_log_sanity_checks(asset, log, cfg)
    if cache:
        log.to_csv(log_path)
        score.to_csv(score_path)
    return log, score


def run_log_sanity_checks(asset: str, log: pd.DataFrame, cfg: Protocol) -> Dict[str, object]:
    probs = log[["behavior_prob_m1", "behavior_prob_0", "behavior_prob_p1"]].to_numpy(dtype=float)
    sums = probs.sum(axis=1)
    if not np.allclose(sums, 1.0, atol=1e-10):
        raise AssertionError(f"{asset}: behavior probability rows do not sum to 1")
    if np.any(probs < cfg.logging_probability_floor - 1e-12):
        raise AssertionError(f"{asset}: probability floor violated")
    if np.any((log["position_before"] < -1e-12) | (log["position_before"] > 1 + 1e-12)):
        raise AssertionError(f"{asset}: position_before out of range")
    freq = log["action"].value_counts(normalize=True).reindex(ACTIONS, fill_value=0.0)
    return {
        "asset": asset,
        "rows": int(len(log)),
        "min_realized_propensity": float(log["behavior_prob"].min()),
        "max_probability_sum_error": float(np.max(np.abs(sums - 1.0))),
        "freq_m1": float(freq.loc[-1]),
        "freq_0": float(freq.loc[0]),
        "freq_p1": float(freq.loc[1]),
    }


# =============================================================================
# 5. Reward model and target policies
# =============================================================================


class RidgeRewardModel:
    def __init__(self, alpha: float) -> None:
        self.alpha = float(alpha)
        self.scaler = StandardScaler()
        self.models: Dict[int, Ridge] = {}
        self.fallback: Dict[int, float] = {}
        self.fitted = False

    def fit(self, df: pd.DataFrame) -> "RidgeRewardModel":
        if len(df) < 10:
            raise ValueError("Too few rows to fit reward model")
        X = model_matrix(df)
        self.scaler.fit(X)
        Xs = self.scaler.transform(X)
        y = df["reward_observed"].to_numpy(dtype=float)
        aobs = df["action"].to_numpy(dtype=int)
        global_mean = float(np.mean(y))
        for a in ACTIONS:
            mask = aobs == int(a)
            if mask.sum() >= 3:
                m = Ridge(alpha=self.alpha, fit_intercept=True)
                m.fit(Xs[mask], y[mask])
                self.models[int(a)] = m
                self.fallback[int(a)] = float(np.mean(y[mask]))
            else:
                self.fallback[int(a)] = global_mean
        self.fitted = True
        return self

    def predict_matrix(self, X: np.ndarray) -> np.ndarray:
        if not self.fitted:
            raise RuntimeError("Reward model not fitted")
        Xs = self.scaler.transform(np.asarray(X, dtype=float))
        q = np.empty((len(Xs), len(ACTIONS)), dtype=float)
        for j, a in enumerate(ACTIONS):
            if int(a) in self.models:
                q[:, j] = self.models[int(a)].predict(Xs)
            else:
                q[:, j] = self.fallback[int(a)]
        return q

    def predict_dataframe(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_matrix(model_matrix(df))


class Policy:
    name = "policy"

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ScaledSoftmaxQPolicy(Policy):
    def __init__(
        self,
        q_model: RidgeRewardModel,
        temperature: float,
        q_scale: float,
        min_prob: float,
        name: str,
    ) -> None:
        self.q_model = q_model
        self.temperature = float(temperature)
        self.q_scale = max(float(q_scale), 1e-12)
        self.min_prob = float(min_prob)
        self.name = name

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        q = self.q_model.predict_matrix(X)
        centered = q - np.mean(q, axis=1, keepdims=True)
        p = stable_softmax(centered / self.q_scale, self.temperature)
        return apply_probability_floor(p, self.min_prob)


class UniformPolicy(Policy):
    def __init__(self, name: str = "uniform") -> None:
        self.name = name

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X), len(ACTIONS)), 1.0 / len(ACTIONS), dtype=float)


class MixturePolicy(Policy):
    def __init__(self, first: Policy, second: Policy, second_weight: float, name: str) -> None:
        self.first = first
        self.second = second
        self.second_weight = float(second_weight)
        self.name = name

    def probabilities(self, X: np.ndarray) -> np.ndarray:
        p1 = self.first.probabilities(X)
        p2 = self.second.probabilities(X)
        return (1.0 - self.second_weight) * p1 + self.second_weight * p2


def robust_q_scale(q: np.ndarray, floor: float) -> float:
    sd = np.std(np.asarray(q, dtype=float), axis=1, ddof=0)
    sd = sd[np.isfinite(sd)]
    return max(float(np.median(sd)) if len(sd) else 0.0, float(floor))


def fit_policy_pair(
    history: pd.DataFrame,
    cfg: Protocol,
    candidate_history: Optional[int] = None,
) -> Tuple[Policy, Policy, Policy]:
    recent_n = int(candidate_history or cfg.candidate_policy_history)
    if len(history) < max(cfg.baseline_policy_history, recent_n):
        raise ValueError("Insufficient history for target policy pair")
    long_df = history.iloc[-cfg.baseline_policy_history :]
    recent_df = history.iloc[-recent_n:]
    long_model = RidgeRewardModel(cfg.ridge_alpha).fit(long_df)
    recent_model = RidgeRewardModel(cfg.ridge_alpha).fit(recent_df)
    long_scale = robust_q_scale(long_model.predict_dataframe(long_df), cfg.q_scale_floor)
    recent_scale = robust_q_scale(recent_model.predict_dataframe(recent_df), cfg.q_scale_floor)
    base = ScaledSoftmaxQPolicy(
        long_model, cfg.base_temperature, long_scale, cfg.target_policy_probability_floor, "base_756"
    )
    new = ScaledSoftmaxQPolicy(
        recent_model,
        cfg.candidate_temperature,
        recent_scale,
        cfg.target_policy_probability_floor,
        f"candidate_{recent_n}",
    )
    mix = MixturePolicy(base, new, cfg.conservative_mix_weight, "conservative_mix")
    return base, new, mix


# =============================================================================
# 6. Rolling-origin nuisance predictions (DM/DR cross-fitting)
# =============================================================================


def qhat_cache_path(
    paths: Mapping[str, Path], asset: str, seed_index: int, transaction_cost: float, risk_lambda: float
) -> Path:
    tag = reward_tag(transaction_cost, risk_lambda)
    return paths["cache_qhat"] / f"{asset}_{tag}_seed{seed_index:03d}_qhat.csv"


def rolling_origin_qhat(
    asset: str,
    log: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    paths: Mapping[str, Path],
    transaction_cost: float,
    risk_lambda: float,
    cache: bool = True,
) -> pd.DataFrame:
    path = qhat_cache_path(paths, asset, seed_index, transaction_cost, risk_lambda)
    if cache and path.exists() and not cfg.force_recompute_qhat:
        return pd.read_csv(path, index_col=0, parse_dates=True).sort_index()

    n = len(log)
    q = np.full((n, 3), np.nan, dtype=float)
    train_end_dates: List[Optional[pd.Timestamp]] = [None] * n
    block_ids = np.full(n, -1, dtype=int)
    block = int(cfg.nuisance_prediction_block)
    warm = int(cfg.nuisance_warmup)

    block_id = 0
    for start in range(warm, n, block):
        end = min(n, start + block)
        train = log.iloc[:start]
        pred = log.iloc[start:end]
        model = RidgeRewardModel(cfg.ridge_alpha).fit(train)
        q[start:end, :] = model.predict_dataframe(pred)
        train_end = pd.Timestamp(train.index[-1])
        for k in range(start, end):
            train_end_dates[k] = train_end
            block_ids[k] = block_id
            if not (train_end < pd.Timestamp(log.index[k])):
                raise AssertionError("Nuisance leakage: training_end >= prediction_date")
        block_id += 1

    out = pd.DataFrame(q, index=log.index, columns=QHAT_COLUMNS)
    out["training_end"] = train_end_dates
    out["prediction_block"] = block_ids
    if cache:
        out.to_csv(path)
    return out


def valid_qhat_mask(qhat: pd.DataFrame) -> np.ndarray:
    return np.all(np.isfinite(qhat[QHAT_COLUMNS].to_numpy(dtype=float)), axis=1)


# =============================================================================
# 7. OPE estimators from precomputed out-of-time q-hat
# =============================================================================


@dataclass
class OPEResult:
    estimate: float
    ess: float
    ess_fraction: float
    max_importance_weight: float
    p99_importance_weight: float
    max_combined_weight_fraction: float


def policy_probs(policy: Policy, df: pd.DataFrame) -> np.ndarray:
    return policy.probabilities(model_matrix(df))


def normalize_sample_weights(weights: Optional[np.ndarray], n: int) -> np.ndarray:
    if weights is None:
        return np.ones(n, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(w) != n:
        raise ValueError("sample weight length mismatch")
    if np.any(~np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("invalid sample weights")
    return w


def effective_sample_size(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    denom = float(np.sum(w**2))
    if denom <= 0:
        return 0.0
    return float((w.sum() ** 2) / denom)


def ope_components_from_qhat(df: pd.DataFrame, qhat: pd.DataFrame, policy: Policy) -> Dict[str, np.ndarray]:
    if not df.index.equals(qhat.index):
        qhat = qhat.reindex(df.index)
    q = qhat[QHAT_COLUMNS].to_numpy(dtype=float)
    if np.any(~np.isfinite(q)):
        raise ValueError("OPE called with missing q-hat values")
    probs = policy_probs(policy, df)
    idx = observed_action_indices(df)
    behavior = df["behavior_prob"].to_numpy(dtype=float)
    reward = df["reward_observed"].to_numpy(dtype=float)
    if np.any(behavior <= 0):
        raise ValueError("Non-positive behavior propensity")
    target_taken = probs[np.arange(len(df)), idx]
    imp = target_taken / behavior
    model_value = np.sum(probs * q, axis=1)
    q_taken = q[np.arange(len(df)), idx]
    dr = model_value + imp * (reward - q_taken)
    return {"importance": imp, "reward": reward, "model_value": model_value, "dr": dr}


def evaluate_ope_from_qhat(
    df: pd.DataFrame,
    qhat: pd.DataFrame,
    policy: Policy,
    estimator: str,
    sample_weights: Optional[np.ndarray] = None,
) -> OPEResult:
    sw = normalize_sample_weights(sample_weights, len(df))
    comp = ope_components_from_qhat(df, qhat, policy)
    imp = comp["importance"]
    est = estimator.lower()
    if est == "dm":
        v = float(np.sum(sw * comp["model_value"]) / np.sum(sw))
    elif est == "ips":
        v = float(np.sum(sw * imp * comp["reward"]) / np.sum(sw))
    elif est == "snips":
        denom = float(np.sum(sw * imp))
        v = float(np.sum(sw * imp * comp["reward"]) / denom) if denom > 1e-12 else float("nan")
    elif est == "dr":
        v = float(np.sum(sw * comp["dr"]) / np.sum(sw))
    else:
        raise ValueError(estimator)
    combined = sw * imp
    total = float(np.sum(combined))
    return OPEResult(
        estimate=v,
        ess=effective_sample_size(combined),
        ess_fraction=float(effective_sample_size(combined) / len(df)),
        max_importance_weight=float(np.max(imp)),
        p99_importance_weight=float(np.quantile(imp, 0.99)),
        max_combined_weight_fraction=float(np.max(combined) / total) if total > 0 else float("nan"),
    )


def counterfactual_value(policy: Policy, context_df: pd.DataFrame, score_df: pd.DataFrame) -> float:
    score = score_df.reindex(context_df.index)
    if score[CF_COLUMNS].isna().any().any():
        raise ValueError("Counterfactual scoring rows missing")
    p = policy_probs(policy, context_df)
    r = score[CF_COLUMNS].to_numpy(dtype=float)
    return float(np.mean(np.sum(p * r, axis=1)))


def counterfactual_daily_value(policy: Policy, context_df: pd.DataFrame, score_df: pd.DataFrame) -> np.ndarray:
    score = score_df.reindex(context_df.index)
    p = policy_probs(policy, context_df)
    r = score[CF_COLUMNS].to_numpy(dtype=float)
    return np.sum(p * r, axis=1)


# =============================================================================
# 8. Anchors and evidence rules
# =============================================================================


def forward_anchors(n: int, cfg: Protocol, reserve_before_future: int = 0, spacing: Optional[int] = None) -> List[int]:
    step = int(spacing or cfg.anchor_spacing)
    first = cfg.min_history
    last = n - reserve_before_future - cfg.forward_horizon
    if last < first:
        return []
    return list(range(first, last + 1, step))


def static_anchors(n: int, cfg: Protocol) -> List[int]:
    # i is the END of an 80-row held-out OPE/scoring block. Both policy fitting
    # and q-hat generation for each row are time-safe.
    first = cfg.min_history + cfg.static_eval_window
    last = n
    return list(range(first, last + 1, cfg.anchor_spacing))


def evidence_slice(history: pd.DataFrame, qhat: pd.DataFrame, rule: str, cfg: Protocol) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if rule == "all_history":
        df = history.copy()
    elif rule.startswith("last_"):
        n = int(rule.split("_")[1])
        df = history.iloc[-min(n, len(history)) :].copy()
    elif rule == f"decay_{cfg.fixed_decay_half_life}":
        df = history.copy()
    else:
        raise ValueError(f"Unknown memory rule {rule}")
    q = qhat.reindex(df.index)
    mask = valid_qhat_mask(q)
    df = df.iloc[np.flatnonzero(mask)].copy()
    q = q.iloc[np.flatnonzero(mask)].copy()
    if rule.startswith("decay_"):
        n = len(df)
        ages = np.arange(n, 0, -1, dtype=float)
        kappa = math.log(2.0) / float(cfg.fixed_decay_half_life)
        w = np.exp(-kappa * ages)
    else:
        w = np.ones(len(df), dtype=float)
    if len(df) == 0:
        raise ValueError("No valid evidence after q-hat warmup")
    return df, q, w


# =============================================================================
# 9. Experiment 1: baseline OPE benchmark across ETFs
# =============================================================================


def experiment_baseline_ope(
    asset: str,
    log: pd.DataFrame,
    score: pd.DataFrame,
    qhat: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for i in static_anchors(len(log), cfg):
        eval_start = i - cfg.static_eval_window
        train = log.iloc[:eval_start]
        eval_df = log.iloc[eval_start:i]
        eval_q = qhat.reindex(eval_df.index)
        mask = valid_qhat_mask(eval_q)
        eval_df = eval_df.iloc[np.flatnonzero(mask)]
        eval_q = eval_q.iloc[np.flatnonzero(mask)]
        if len(train) < cfg.min_history or len(eval_df) < cfg.static_eval_window // 2:
            continue
        base, new, mix = fit_policy_pair(train, cfg)
        policies: List[Policy] = [base, new, mix, UniformPolicy()]
        for policy in policies:
            truth = counterfactual_value(policy, eval_df, score)
            for estimator in ["dm", "ips", "snips", "dr"]:
                res = evaluate_ope_from_qhat(eval_df, eval_q, policy, estimator)
                err = res.estimate - truth
                rows.append(
                    {
                        "asset": asset,
                        "behavior_seed": seed_index,
                        "anchor_date": pd.Timestamp(eval_df.index[-1]),
                        "eval_start": pd.Timestamp(eval_df.index[0]),
                        "eval_end": pd.Timestamp(eval_df.index[-1]),
                        "policy": policy.name,
                        "estimator": estimator.upper(),
                        "value_hat": res.estimate,
                        "value_counterfactual": truth,
                        "signed_error": err,
                        "abs_error": abs(err),
                        "signed_error_bp": err * 1e4,
                        "abs_error_bp": abs(err) * 1e4,
                        "ess": res.ess,
                        "ess_fraction": res.ess_fraction,
                        "max_importance_weight": res.max_importance_weight,
                        "p99_importance_weight": res.p99_importance_weight,
                        "protocol_hash": protocol_hash(cfg),
                    }
                )
    return pd.DataFrame(rows)


def summarize_baseline_ope(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    base = (
        detail.groupby(["asset", "estimator"], as_index=False)
        .agg(
            MAE_bp=("abs_error_bp", "mean"),
            RMSE_bp=("signed_error_bp", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            Bias_bp=("signed_error_bp", "mean"),
            Mean_ESS=("ess", "mean"),
            Mean_ESS_Fraction=("ess_fraction", "mean"),
            N_Comparisons=("abs_error_bp", "size"),
        )
    )
    rank_rows: List[Dict[str, object]] = []
    for (asset, estimator), g0 in detail.groupby(["asset", "estimator"]):
        top: List[float] = []
        corrs: List[float] = []
        for (_, seed), gseed in g0.groupby(["anchor_date", "behavior_seed"]):
            if gseed["policy"].nunique() < 2:
                continue
            top.append(float(gseed.loc[gseed["value_hat"].idxmax(), "policy"] == gseed.loc[gseed["value_counterfactual"].idxmax(), "policy"]))
            rho = spearmanr(gseed["value_counterfactual"], gseed["value_hat"]).statistic
            if np.isfinite(rho):
                corrs.append(float(rho))
        rank_rows.append(
            {
                "asset": asset,
                "estimator": estimator,
                "Best_Policy_Accuracy": float(np.mean(top)) if top else np.nan,
                "Mean_Spearman": float(np.mean(corrs)) if corrs else np.nan,
            }
        )
    return base.merge(pd.DataFrame(rank_rows), on=["asset", "estimator"], how="left").sort_values(["asset", "MAE_bp"])


# =============================================================================
# 10. Experiment 2: fixed historical-memory rules
# =============================================================================


def experiment_memory(
    asset: str,
    log: pd.DataFrame,
    score: pd.DataFrame,
    qhat: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    candidate_history: Optional[int] = None,
) -> pd.DataFrame:
    rules = ["all_history"] + [f"last_{n}" for n in cfg.memory_windows]
    if cfg.include_fixed_decay_supplement:
        rules.append(f"decay_{cfg.fixed_decay_half_life}")
    rows: List[Dict[str, object]] = []
    for i in forward_anchors(len(log), cfg):
        history = log.iloc[:i]
        future = log.iloc[i : i + cfg.forward_horizon]
        if len(future) < cfg.forward_horizon:
            continue
        _, new, _ = fit_policy_pair(history, cfg, candidate_history=candidate_history)
        truth = counterfactual_value(new, future, score)
        for rule in rules:
            ev, eq, w = evidence_slice(history, qhat, rule, cfg)
            res = evaluate_ope_from_qhat(ev, eq, new, "dr", sample_weights=w)
            err = res.estimate - truth
            rows.append(
                {
                    "asset": asset,
                    "behavior_seed": seed_index,
                    "anchor_date": pd.Timestamp(future.index[0]),
                    "memory_rule": rule,
                    "candidate_history": int(candidate_history or cfg.candidate_policy_history),
                    "value_hat": res.estimate,
                    "value_counterfactual": truth,
                    "signed_error": err,
                    "abs_error": abs(err),
                    "signed_error_bp": err * 1e4,
                    "abs_error_bp": abs(err) * 1e4,
                    "ess": res.ess,
                    "ess_fraction": res.ess_fraction,
                    "n_logged": len(ev),
                    "protocol_hash": protocol_hash(cfg),
                }
            )
    return pd.DataFrame(rows)


def summarize_memory(detail: pd.DataFrame, main_only: bool = True) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    d = detail.copy()
    if main_only:
        d = d[~d["memory_rule"].str.startswith("decay_")]
    summary = (
        d.groupby(["asset", "memory_rule"], as_index=False)
        .agg(
            MAE_bp=("abs_error_bp", "mean"),
            RMSE_bp=("signed_error_bp", lambda x: float(np.sqrt(np.mean(np.asarray(x) ** 2)))),
            Bias_bp=("signed_error_bp", "mean"),
            Mean_ESS=("ess", "mean"),
            N_Windows=("anchor_date", "nunique"),
            N_Seeds=("behavior_seed", "nunique"),
        )
    )
    all_mae = summary[summary["memory_rule"] == "all_history"][["asset", "MAE_bp"]].rename(columns={"MAE_bp": "All_History_MAE_bp"})
    summary = summary.merge(all_mae, on="asset", how="left")
    summary["Relative_MAE_vs_All"] = summary["MAE_bp"] / summary["All_History_MAE_bp"] - 1.0
    return summary.sort_values(["asset", "MAE_bp"])


# =============================================================================
# 11. Experiment 3: recent evidence vs target-policy coverage
# =============================================================================


def take_most_recent(df: pd.DataFrame, m: int) -> pd.DataFrame:
    if m <= 0:
        return df.iloc[:0].copy()
    return df.iloc[-min(int(m), len(df)) :].copy()


def fixed_qhat_for_pool(pre_pool_history: pd.DataFrame, pool: pd.DataFrame, cfg: Protocol) -> pd.DataFrame:
    model = RidgeRewardModel(cfg.ridge_alpha).fit(pre_pool_history)
    q = model.predict_dataframe(pool)
    return pd.DataFrame(q, index=pool.index, columns=QHAT_COLUMNS)


def target_aware_relog_pool(
    pool_context: pd.DataFrame,
    pool_score: pd.DataFrame,
    base: Policy,
    new: Policy,
    cfg: Protocol,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Re-log fixed stored contexts; no alternative multi-step position path."""
    X = model_matrix(pool_context)
    pb = base.probabilities(X)
    pn = new.probabilities(X)
    beh = (1.0 - cfg.target_aware_mix) * pb + cfg.target_aware_mix * pn
    score = pool_score.reindex(pool_context.index)[CF_COLUMNS].to_numpy(dtype=float)
    rows: List[Dict[str, object]] = []
    for k, (date, row) in enumerate(pool_context.iterrows()):
        idx = int(rng.choice(len(ACTIONS), p=beh[k]))
        rec = row.to_dict()
        rec.update(
            {
                "date": pd.Timestamp(date),
                "action": int(ACTIONS[idx]),
                "behavior_prob": float(beh[k, idx]),
                "behavior_prob_m1": float(beh[k, 0]),
                "behavior_prob_0": float(beh[k, 1]),
                "behavior_prob_p1": float(beh[k, 2]),
                "reward_observed": float(score[k, idx]),
            }
        )
        rows.append(rec)
    return pd.DataFrame(rows).set_index("date").sort_index()


def experiment_recency_coverage(
    asset: str,
    log: pd.DataFrame,
    score: pd.DataFrame,
    qhat: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
) -> pd.DataFrame:
    max_m = max(cfg.recent_evidence_grid)
    # 160 spacing prevents overlap of the full pool+future 160-row blocks.
    anchors = forward_anchors(len(log), cfg, reserve_before_future=max_m, spacing=2 * cfg.anchor_spacing)
    rows: List[Dict[str, object]] = []
    for i in anchors:
        history = log.iloc[:i]
        pool = log.iloc[i : i + max_m]
        pool_score = score.iloc[i : i + max_m]
        future = log.iloc[i + max_m : i + max_m + cfg.forward_horizon]
        if len(pool) < max_m or len(future) < cfg.forward_horizon:
            continue
        base, new, _ = fit_policy_pair(history, cfg)
        truth = counterfactual_value(new, future, score)
        hist_ev, hist_q, hist_w = evidence_slice(history, qhat, "all_history", cfg)

        pool_q = fixed_qhat_for_pool(history, pool, cfg)
        relog = target_aware_relog_pool(
            pool,
            pool_score,
            base,
            new,
            cfg,
            deterministic_rng(cfg.base_seed, asset, seed_index, salt=4000 + i),
        )

        for m in cfg.recent_evidence_grid:
            m = int(m)
            if m == 0:
                res = evaluate_ope_from_qhat(hist_ev, hist_q, new, "dr", hist_w)
                rows.append(
                    {
                        "asset": asset,
                        "behavior_seed": seed_index,
                        "anchor_date": pd.Timestamp(future.index[0]),
                        "evidence_type": "history_only",
                        "m": 0,
                        "value_hat": res.estimate,
                        "value_counterfactual": truth,
                        "abs_error_bp": abs(res.estimate - truth) * 1e4,
                        "signed_error_bp": (res.estimate - truth) * 1e4,
                        "ess": res.ess,
                        "protocol_hash": protocol_hash(cfg),
                    }
                )
                continue

            recent_original = take_most_recent(pool, m)
            recent_relog = take_most_recent(relog, m)
            recent_q = pool_q.reindex(recent_original.index)
            for etype, recent_df in [
                ("recent_original_policy", recent_original),
                ("target_aware_relogging", recent_relog),
            ]:
                combined = pd.concat([hist_ev, recent_df], axis=0)
                combined_q = pd.concat([hist_q[QHAT_COLUMNS], recent_q[QHAT_COLUMNS]], axis=0)
                weights = np.ones(len(combined), dtype=float)
                res = evaluate_ope_from_qhat(combined, combined_q, new, "dr", weights)
                rows.append(
                    {
                        "asset": asset,
                        "behavior_seed": seed_index,
                        "anchor_date": pd.Timestamp(future.index[0]),
                        "evidence_type": etype,
                        "m": m,
                        "value_hat": res.estimate,
                        "value_counterfactual": truth,
                        "abs_error_bp": abs(res.estimate - truth) * 1e4,
                        "signed_error_bp": (res.estimate - truth) * 1e4,
                        "ess": res.ess,
                        "protocol_hash": protocol_hash(cfg),
                    }
                )
    return pd.DataFrame(rows)


def summarize_recency_coverage(detail: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if detail.empty:
        return pd.DataFrame(), pd.DataFrame()
    curve = (
        detail.groupby(["asset", "evidence_type", "m"], as_index=False)
        .agg(
            MAE_bp=("abs_error_bp", "mean"),
            Bias_bp=("signed_error_bp", "mean"),
            Mean_ESS=("ess", "mean"),
            N_Windows=("anchor_date", "nunique"),
            N_Seeds=("behavior_seed", "nunique"),
        )
        .sort_values(["asset", "evidence_type", "m"])
    )
    base = curve[(curve["evidence_type"] == "history_only") & (curve["m"] == 0)][["asset", "MAE_bp", "Mean_ESS"]].rename(
        columns={"MAE_bp": "Baseline_MAE_bp", "Mean_ESS": "Baseline_ESS"}
    )
    max_m = int(detail["m"].max())
    # history_only only exists at m=0; summarize the largest pre-specified recent-evidence budget.
    main = curve[curve["m"].eq(max_m) & curve["evidence_type"].ne("history_only")].merge(base, on="asset", how="left")
    main["MAE_Change_pct"] = 100.0 * (main["MAE_bp"] / main["Baseline_MAE_bp"] - 1.0)
    main["ESS_Change_pct"] = 100.0 * (main["Mean_ESS"] / main["Baseline_ESS"] - 1.0)
    return curve, main


# =============================================================================
# 12. Experiment 4: candidate-baseline statistical resolution
# =============================================================================


def moving_block_indices(n: int, block_length: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    L = max(1, min(int(block_length), n))
    out: List[int] = []
    while len(out) < n:
        s = int(rng.integers(0, n - L + 1))
        out.extend(range(s, s + L))
    return np.asarray(out[:n], dtype=int)


def bootstrap_dr_delta(
    evidence: pd.DataFrame,
    qhat: pd.DataFrame,
    new: Policy,
    base: Policy,
    cfg: Protocol,
    rng: np.random.Generator,
) -> Dict[str, float]:
    cn = ope_components_from_qhat(evidence, qhat, new)
    cb = ope_components_from_qhat(evidence, qhat, base)
    diff = cn["dr"] - cb["dr"]
    point = float(np.mean(diff))
    boot = np.empty(cfg.bootstrap_resamples, dtype=float)
    for b in range(cfg.bootstrap_resamples):
        idx = moving_block_indices(len(diff), cfg.bootstrap_block_length, rng)
        boot[b] = float(np.mean(diff[idx]))
    se = float(np.std(boot, ddof=1))
    z = float(norm.ppf(cfg.confidence_level))
    return {
        "delta_hat": point,
        "bootstrap_se": se,
        "lcb_normal": point - z * se,
        "bootstrap_q05": float(np.quantile(boot, 0.05)),
        "bootstrap_q50": float(np.quantile(boot, 0.50)),
        "bootstrap_q95": float(np.quantile(boot, 0.95)),
    }


def experiment_policy_resolution(
    asset: str,
    log: pd.DataFrame,
    score: pd.DataFrame,
    qhat: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    candidate_history: Optional[int] = None,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for i in forward_anchors(len(log), cfg):
        history = log.iloc[:i]
        future = log.iloc[i : i + cfg.forward_horizon]
        if len(future) < cfg.forward_horizon:
            continue
        base, new, _ = fit_policy_pair(history, cfg, candidate_history=candidate_history)
        evidence, eq, _ = evidence_slice(history, qhat, "all_history", cfg)
        stats = bootstrap_dr_delta(
            evidence,
            eq,
            new,
            base,
            cfg,
            deterministic_rng(cfg.base_seed, asset, seed_index, salt=7000 + i + int(candidate_history or cfg.candidate_policy_history)),
        )
        truth_new = counterfactual_value(new, future, score)
        truth_base = counterfactual_value(base, future, score)
        delta_cf = truth_new - truth_base
        selected_candidate = stats["delta_hat"] > 0
        lcb_candidate = stats["lcb_normal"] > 0
        rows.append(
            {
                "asset": asset,
                "behavior_seed": seed_index,
                "anchor_date": pd.Timestamp(future.index[0]),
                "candidate_history": int(candidate_history or cfg.candidate_policy_history),
                "delta_v_hat": stats["delta_hat"],
                "delta_v_counterfactual": delta_cf,
                "delta_v_hat_bp": stats["delta_hat"] * 1e4,
                "delta_v_counterfactual_bp": delta_cf * 1e4,
                "bootstrap_se": stats["bootstrap_se"],
                "bootstrap_se_bp": stats["bootstrap_se"] * 1e4,
                "lcb_normal": stats["lcb_normal"],
                "lcb_normal_bp": stats["lcb_normal"] * 1e4,
                "point_select_candidate": int(selected_candidate),
                "lcb_select_candidate": int(lcb_candidate),
                "counterfactual_candidate_better": int(delta_cf > 0),
                "point_selection_correct": int(selected_candidate == (delta_cf > 0)),
                "lcb_selection_correct": int(lcb_candidate == (delta_cf > 0)),
                "point_worse_policy_adopted": int(selected_candidate and delta_cf < 0),
                "lcb_worse_policy_adopted": int(lcb_candidate and delta_cf < 0),
                "n_evidence": len(evidence),
                "protocol_hash": protocol_hash(cfg),
            }
        )
    return pd.DataFrame(rows)


def conditional_unsafe_rate(df: pd.DataFrame, adopt_col: str, unsafe_col: str) -> float:
    adopted = df[adopt_col].astype(bool)
    if adopted.sum() == 0:
        return float("nan")
    return float(df.loc[adopted, unsafe_col].mean())


def summarize_policy_resolution(detail: pd.DataFrame) -> pd.DataFrame:
    if detail.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for asset, g in detail.groupby("asset"):
        med_gap = float(np.median(np.abs(g["delta_v_counterfactual_bp"])))
        med_se = float(np.median(g["bootstrap_se_bp"]))
        rows.append(
            {
                "asset": asset,
                "Median_Abs_DeltaV_bp": med_gap,
                "Median_SE_bp": med_se,
                "Gap_to_SE": med_gap / med_se if med_se > 0 else np.nan,
                "Point_Selection_Accuracy": float(g["point_selection_correct"].mean()),
                "Point_Adoption_Rate": float(g["point_select_candidate"].mean()),
                "Point_Worse_Adoption_Rate": conditional_unsafe_rate(g, "point_select_candidate", "point_worse_policy_adopted"),
                "LCB_Selection_Accuracy": float(g["lcb_selection_correct"].mean()),
                "LCB_Adoption_Rate": float(g["lcb_select_candidate"].mean()),
                "LCB_Worse_Adoption_Rate": conditional_unsafe_rate(g, "lcb_select_candidate", "lcb_worse_policy_adopted"),
                "N_Windows": int(g["anchor_date"].nunique()),
                "N_Seeds": int(g["behavior_seed"].nunique()),
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 13. Seed-level main run, resumable output
# =============================================================================


def seed_result_dir(paths: Mapping[str, Path], asset: str, seed_index: int, tag: str = "main") -> Path:
    p = paths["detail"] / "per_seed" / tag / asset / f"seed_{seed_index:03d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def assert_cached_protocol(df: pd.DataFrame, cfg: Protocol, label: str) -> None:
    """Refuse to reuse per-seed scientific results from another protocol."""
    if "protocol_hash" not in df.columns:
        raise RuntimeError(f"{label}: cached result has no protocol_hash; recompute in a clean output directory.")
    got = set(df["protocol_hash"].dropna().astype(str).unique())
    expected = {protocol_hash(cfg)}
    if got != expected:
        raise RuntimeError(
            f"{label}: cached protocol hash {sorted(got)} != current {sorted(expected)}. "
            "Do not mix cached results across scientific protocols."
        )


def run_main_seed(
    asset: str,
    market: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    paths: Mapping[str, Path],
) -> Dict[str, pd.DataFrame]:
    outdir = seed_result_dir(paths, asset, seed_index, "main")
    expected = [outdir / "baseline.csv", outdir / "memory.csv", outdir / "recency_coverage.csv", outdir / "resolution.csv"]
    if all(p.exists() for p in expected):
        cached = {
            "baseline": pd.read_csv(expected[0], parse_dates=["anchor_date", "eval_start", "eval_end"]),
            "memory": pd.read_csv(expected[1], parse_dates=["anchor_date"]),
            "recency": pd.read_csv(expected[2], parse_dates=["anchor_date"]),
            "resolution": pd.read_csv(expected[3], parse_dates=["anchor_date"]),
        }
        for key, df in cached.items():
            assert_cached_protocol(df, cfg, f"{asset} seed={seed_index} {key}")
        return cached

    log, score = simulate_behavior_log(asset, market, seed_index, cfg, paths, cfg.baseline_cost, 0.0, cache=True)
    qhat = rolling_origin_qhat(asset, log, seed_index, cfg, paths, cfg.baseline_cost, 0.0, cache=True)
    baseline = experiment_baseline_ope(asset, log, score, qhat, seed_index, cfg)
    memory = experiment_memory(asset, log, score, qhat, seed_index, cfg)
    recency = experiment_recency_coverage(asset, log, score, qhat, seed_index, cfg)
    resolution = experiment_policy_resolution(asset, log, score, qhat, seed_index, cfg)
    baseline.to_csv(expected[0], index=False)
    memory.to_csv(expected[1], index=False)
    recency.to_csv(expected[2], index=False)
    resolution.to_csv(expected[3], index=False)
    return {"baseline": baseline, "memory": memory, "recency": recency, "resolution": resolution}


def concat_nonempty(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    good = [x for x in parts if x is not None and not x.empty]
    return pd.concat(good, ignore_index=True) if good else pd.DataFrame()


# =============================================================================
# 14. Financial robustness experiments
# =============================================================================


def run_reward_robustness_seed(
    asset: str,
    market: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    paths: Mapping[str, Path],
    transaction_cost: float,
    risk_lambda: float,
    tag: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    outdir = seed_result_dir(paths, asset, seed_index, tag)
    mem_path = outdir / "memory.csv"
    res_path = outdir / "resolution.csv"
    if mem_path.exists() and res_path.exists():
        mem_cached = pd.read_csv(mem_path, parse_dates=["anchor_date"])
        res_cached = pd.read_csv(res_path, parse_dates=["anchor_date"])
        assert_cached_protocol(mem_cached, cfg, f"{asset} seed={seed_index} {tag} memory")
        assert_cached_protocol(res_cached, cfg, f"{asset} seed={seed_index} {tag} resolution")
        return mem_cached, res_cached
    log, score = simulate_behavior_log(asset, market, seed_index, cfg, paths, transaction_cost, risk_lambda, cache=True)
    qhat = rolling_origin_qhat(asset, log, seed_index, cfg, paths, transaction_cost, risk_lambda, cache=True)
    mem = experiment_memory(asset, log, score, qhat, seed_index, cfg)
    res = experiment_policy_resolution(asset, log, score, qhat, seed_index, cfg)
    mem["transaction_cost"] = transaction_cost
    mem["risk_lambda"] = risk_lambda
    res["transaction_cost"] = transaction_cost
    res["risk_lambda"] = risk_lambda
    mem.to_csv(mem_path, index=False)
    res.to_csv(res_path, index=False)
    return mem, res


def summarize_reward_robustness(memory_detail: pd.DataFrame, resolution_detail: pd.DataFrame) -> pd.DataFrame:
    if memory_detail.empty or resolution_detail.empty:
        return pd.DataFrame()
    mem = (
        memory_detail[~memory_detail["memory_rule"].str.startswith("decay_")]
        .groupby(["asset", "transaction_cost", "risk_lambda", "memory_rule"], as_index=False)
        .agg(MAE_bp=("abs_error_bp", "mean"))
    )
    rows: List[Dict[str, object]] = []
    for (asset, c, lam), gm in mem.groupby(["asset", "transaction_cost", "risk_lambda"]):
        gr = resolution_detail[
            (resolution_detail["asset"] == asset)
            & np.isclose(resolution_detail["transaction_cost"], c)
            & np.isclose(resolution_detail["risk_lambda"], lam)
        ]
        all_mae = gm.loc[gm["memory_rule"] == "all_history", "MAE_bp"]
        best_rule = gm.loc[gm["MAE_bp"].idxmin(), "memory_rule"] if len(gm) else ""
        rows.append(
            {
                "asset": asset,
                "transaction_cost": c,
                "risk_lambda": lam,
                "All_History_MAE_bp": float(all_mae.iloc[0]) if len(all_mae) else np.nan,
                "Lowest_MAE_Memory_Rule": best_rule,
                "Median_Abs_DeltaV_bp": float(np.median(np.abs(gr["delta_v_counterfactual_bp"]))) if len(gr) else np.nan,
                "Median_SE_bp": float(np.median(gr["bootstrap_se_bp"])) if len(gr) else np.nan,
                "Point_Selection_Accuracy": float(gr["point_selection_correct"].mean()) if len(gr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def run_alternative_policy_contrast_seed(
    asset: str,
    market: pd.DataFrame,
    seed_index: int,
    cfg: Protocol,
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    outdir = seed_result_dir(paths, asset, seed_index, "alt_policy")
    p = outdir / "resolution_alt.csv"
    if p.exists():
        cached = pd.read_csv(p, parse_dates=["anchor_date"])
        assert_cached_protocol(cached, cfg, f"{asset} seed={seed_index} alternative-policy resolution")
        return cached
    log, score = simulate_behavior_log(asset, market, seed_index, cfg, paths, cfg.baseline_cost, 0.0, cache=True)
    qhat = rolling_origin_qhat(asset, log, seed_index, cfg, paths, cfg.baseline_cost, 0.0, cache=True)
    res = experiment_policy_resolution(
        asset, log, score, qhat, seed_index, cfg, candidate_history=cfg.alternative_candidate_history
    )
    res.to_csv(p, index=False)
    return res


# =============================================================================
# 15. Optional supplementary experiments
# =============================================================================


def experiment_exploration_supplement(
    asset: str,
    market: pd.DataFrame,
    cfg: Protocol,
    paths: Mapping[str, Path],
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    # Use a fixed target policy built from a baseline-epsilon log. This is a
    # logging-design diagnostic, not a causal theorem about epsilon.
    for eps in cfg.exploration_epsilons:
        for s in range(cfg.exploration_seed_count):
            local_cfg = replace(cfg, force_recompute_logs=True, force_recompute_qhat=True)
            log, score = simulate_behavior_log(
                asset,
                market,
                s,
                local_cfg,
                paths,
                cfg.baseline_cost,
                0.0,
                epsilon=float(eps),
                cache=False,
            )
            qhat = rolling_origin_qhat(
                asset,
                log,
                s,
                local_cfg,
                paths,
                cfg.baseline_cost,
                0.0,
                cache=False,
            )
            # Summarize DR error for candidate policy over main forward anchors.
            for i in forward_anchors(len(log), cfg):
                history = log.iloc[:i]
                future = log.iloc[i : i + cfg.forward_horizon]
                _, new, _ = fit_policy_pair(history, cfg)
                ev, eq, w = evidence_slice(history, qhat, "all_history", cfg)
                res = evaluate_ope_from_qhat(ev, eq, new, "dr", w)
                truth = counterfactual_value(new, future, score)
                rows.append(
                    {
                        "asset": asset,
                        "epsilon": eps,
                        "supp_seed": s,
                        "anchor_date": pd.Timestamp(future.index[0]),
                        "abs_error_bp": abs(res.estimate - truth) * 1e4,
                        "ess": res.ess,
                    }
                )
    return pd.DataFrame(rows)


def synthetic_lcb_stress(cfg: Protocol) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.base_seed + 909090)
    deltas = [-0.0004, -0.0002, -0.0001, 0.0, 0.0001, 0.0002, 0.0004]
    n = 252
    noise_std = 0.001
    phi = 0.35
    rows = []
    z = float(norm.ppf(cfg.confidence_level))
    for true_delta in deltas:
        for rep in range(cfg.synthetic_lcb_repeats):
            eps = rng.normal(0, noise_std, size=n)
            x = np.empty(n)
            x[0] = eps[0]
            for t in range(1, n):
                x[t] = phi * x[t - 1] + eps[t]
            obs = true_delta + x
            point = float(np.mean(obs))
            boot = np.empty(200)
            for b in range(200):
                idx = moving_block_indices(n, cfg.bootstrap_block_length, rng)
                boot[b] = float(np.mean(obs[idx]))
            se = float(np.std(boot, ddof=1))
            rows.append(
                {
                    "true_delta": true_delta,
                    "rep": rep,
                    "point_adopt": int(point > 0),
                    "lcb_adopt": int(point - z * se > 0),
                }
            )
    return pd.DataFrame(rows)


# =============================================================================
# 16. Publication tables and cross-ETF helpers
# =============================================================================


def make_sample_table(markets: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for asset, df in markets.items():
        rows.append(
            {
                "ETF": asset,
                "Final_sample_start": str(pd.Timestamp(df.index.min()).date()),
                "Final_sample_end": str(pd.Timestamp(df.index.max()).date()),
                "Decision_observations": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def make_cross_etf_helpers(memory_summary: pd.DataFrame, recency_curve: pd.DataFrame, resolution_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for asset in sorted(memory_summary["asset"].unique()):
        gm = memory_summary[memory_summary["asset"] == asset]
        all_mae = gm.loc[gm["memory_rule"] == "all_history", "MAE_bp"]
        shorts = gm[gm["memory_rule"].isin(["last_252", "last_126", "last_60"])]
        gc = recency_curve[recency_curve["asset"] == asset]
        base = gc[(gc["evidence_type"] == "history_only") & (gc["m"] == 0)]
        recent80 = gc[(gc["evidence_type"] == "recent_original_policy") & (gc["m"] == 80)]
        relog80 = gc[(gc["evidence_type"] == "target_aware_relogging") & (gc["m"] == 80)]
        gr = resolution_summary[resolution_summary["asset"] == asset]
        a = float(all_mae.iloc[0]) if len(all_mae) else np.nan
        rows.append(
            {
                "asset": asset,
                "all_history_mae_bp": a,
                "best_short_mae_bp": float(shorts["MAE_bp"].min()) if len(shorts) else np.nan,
                "all_short_windows_worse_than_all": bool(np.all(shorts["MAE_bp"].to_numpy() > a)) if len(shorts) and np.isfinite(a) else None,
                "recent80_mae_change_pct": 100.0 * (float(recent80["MAE_bp"].iloc[0]) / float(base["MAE_bp"].iloc[0]) - 1.0) if len(recent80) and len(base) else np.nan,
                "relog80_mae_change_pct": 100.0 * (float(relog80["MAE_bp"].iloc[0]) / float(base["MAE_bp"].iloc[0]) - 1.0) if len(relog80) and len(base) else np.nan,
                "relog80_ess_change_pct": 100.0 * (float(relog80["Mean_ESS"].iloc[0]) / float(base["Mean_ESS"].iloc[0]) - 1.0) if len(relog80) and len(base) else np.nan,
                "median_gap_bp": float(gr["Median_Abs_DeltaV_bp"].iloc[0]) if len(gr) else np.nan,
                "median_se_bp": float(gr["Median_SE_bp"].iloc[0]) if len(gr) else np.nan,
                "gap_to_se": float(gr["Gap_to_SE"].iloc[0]) if len(gr) else np.nan,
            }
        )
    return pd.DataFrame(rows)


# =============================================================================
# 17. Figures (single axes, publication-oriented)
# =============================================================================


def finish_figure(path: Path, cfg: Protocol) -> None:
    plt.tight_layout()
    if cfg.save_figures:
        plt.savefig(path, dpi=180, bbox_inches="tight")
    if cfg.show_figures:
        plt.show(block=False)
    else:
        plt.close()


def plot_baseline_ope(summary: pd.DataFrame, paths: Mapping[str, Path], cfg: Protocol) -> None:
    if summary.empty:
        return
    pivot = summary.pivot(index="estimator", columns="asset", values="MAE_bp")
    ax = pivot.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Mean absolute OPE error (bp)")
    ax.set_xlabel("Estimator")
    ax.set_title("Baseline OPE accuracy across ETFs")
    finish_figure(paths["figures"] / "F1_baseline_ope_mae_by_etf.png", cfg)


def plot_memory(summary: pd.DataFrame, paths: Mapping[str, Path], cfg: Protocol) -> None:
    if summary.empty:
        return
    order = ["all_history", "last_252", "last_126", "last_60"]
    p = summary[summary["memory_rule"].isin(order)].pivot(index="memory_rule", columns="asset", values="MAE_bp").reindex(order)
    ax = p.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("DR mean absolute error (bp)")
    ax.set_xlabel("Historical evidence rule")
    ax.set_title("Historical-memory comparison")
    finish_figure(paths["figures"] / "F2_memory_mae_by_etf.png", cfg)


def plot_recency_coverage(curve: pd.DataFrame, paths: Mapping[str, Path], cfg: Protocol) -> None:
    if curve.empty:
        return
    for asset in cfg.assets:
        g = curve[curve["asset"] == asset]
        plt.figure(figsize=(7, 5))
        for etype, ge in g[g["evidence_type"] != "history_only"].groupby("evidence_type"):
            plt.plot(ge["m"], ge["MAE_bp"], marker="o", label=etype)
        base = g[(g["evidence_type"] == "history_only") & (g["m"] == 0)]
        if len(base):
            plt.axhline(float(base["MAE_bp"].iloc[0]), linestyle="--", label="history_only")
        plt.xlabel("Number of recent observations")
        plt.ylabel("DR mean absolute error (bp)")
        plt.title(f"Recency versus coverage: {asset}")
        plt.legend()
        finish_figure(paths["figures"] / f"F3_recency_coverage_{asset}.png", cfg)


def plot_gap_vs_se(detail: pd.DataFrame, paths: Mapping[str, Path], cfg: Protocol) -> None:
    if detail.empty:
        return
    plt.figure(figsize=(7, 5))
    for asset, g in detail.groupby("asset"):
        plt.scatter(np.abs(g["delta_v_counterfactual_bp"]), g["bootstrap_se_bp"], alpha=0.6, label=asset)
    lim = max(float(np.nanmax(np.abs(detail["delta_v_counterfactual_bp"]))), float(np.nanmax(detail["bootstrap_se_bp"])))
    plt.plot([0, lim], [0, lim], linestyle="--")
    plt.xlabel("Absolute counterfactual policy gap (bp)")
    plt.ylabel("Bootstrap SE of estimated gap (bp)")
    plt.title("Policy gap versus statistical uncertainty")
    plt.legend()
    finish_figure(paths["figures"] / "F4_policy_gap_vs_uncertainty.png", cfg)


def plot_cost_robustness(summary: pd.DataFrame, paths: Mapping[str, Path], cfg: Protocol) -> None:
    if summary.empty:
        return
    g = summary[np.isclose(summary["risk_lambda"], 0.0)]
    plt.figure(figsize=(7, 5))
    for asset, ga in g.groupby("asset"):
        plt.plot(ga["transaction_cost"], ga["Median_Abs_DeltaV_bp"], marker="o", label=asset)
    plt.xlabel("Transaction-cost parameter")
    plt.ylabel("Median absolute policy gap (bp)")
    plt.title("Transaction-cost robustness")
    plt.legend()
    finish_figure(paths["figures"] / "F5_transaction_cost_robustness.png", cfg)


# =============================================================================
# 18. Output validation
# =============================================================================


REQUIRED_TABLES = [
    "T01_etf_sample.csv",
    "T02_baseline_ope.csv",
    "T03_memory_comparison.csv",
    "T04_recency_vs_coverage.csv",
    "T04b_recency_coverage_full_curve.csv",
    "T05_policy_resolution.csv",
    "T06_cross_etf_helpers.csv",
    "T07_transaction_cost_robustness.csv",
    "T08_risk_reward_robustness.csv",
    "T09_alternative_policy_contrast.csv",
]


def validate_final_outputs(paths: Mapping[str, Path], cfg: Protocol) -> pd.DataFrame:
    checks: List[Dict[str, object]] = []
    for name in REQUIRED_TABLES:
        p = paths["tables"] / name
        checks.append({"item": name, "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    for asset in cfg.assets:
        p = paths["processed"] / f"{asset.lower()}_features.csv"
        checks.append({"item": f"processed_{asset}", "exists": p.exists(), "size_bytes": p.stat().st_size if p.exists() else 0})
    out = pd.DataFrame(checks)
    out.to_csv(paths["metadata"] / "final_output_validation.csv", index=False)
    if not bool(out["exists"].all()):
        missing = out.loc[~out["exists"], "item"].tolist()
        raise RuntimeError(f"Final output validation failed. Missing: {missing}")
    return out


# =============================================================================
# 19. Orchestration
# =============================================================================


def write_detail(df: pd.DataFrame, path: Path) -> None:
    if not df.empty:
        df.to_csv(path, index=False)


def aggregate_main(markets: Mapping[str, pd.DataFrame], cfg: Protocol, paths: Mapping[str, Path], seed_count: int) -> Dict[str, pd.DataFrame]:
    baseline_parts: List[pd.DataFrame] = []
    memory_parts: List[pd.DataFrame] = []
    recency_parts: List[pd.DataFrame] = []
    resolution_parts: List[pd.DataFrame] = []
    sanity_rows: List[Dict[str, object]] = []

    for asset in cfg.assets:
        print(f"\n=== MAIN {asset} ===")
        for seed_index in range(seed_count):
            t0 = time.time()
            print(f"{asset}: seed {seed_index+1}/{seed_count}", flush=True)
            result = run_main_seed(asset, markets[asset], seed_index, cfg, paths)
            baseline_parts.append(result["baseline"])
            memory_parts.append(result["memory"])
            recency_parts.append(result["recency"])
            resolution_parts.append(result["resolution"])
            log, _ = simulate_behavior_log(asset, markets[asset], seed_index, cfg, paths, cfg.baseline_cost, 0.0, cache=True)
            r = run_log_sanity_checks(asset, log, cfg)
            r["behavior_seed"] = seed_index
            r["runtime_seconds_seed"] = time.time() - t0
            sanity_rows.append(r)

    baseline = concat_nonempty(baseline_parts)
    memory = concat_nonempty(memory_parts)
    recency = concat_nonempty(recency_parts)
    resolution = concat_nonempty(resolution_parts)
    write_detail(baseline, paths["detail"] / "main_baseline_ope_detail.csv")
    write_detail(memory, paths["detail"] / "main_memory_detail.csv")
    write_detail(recency, paths["detail"] / "main_recency_coverage_detail.csv")
    write_detail(resolution, paths["detail"] / "main_policy_resolution_detail.csv")
    pd.DataFrame(sanity_rows).to_csv(paths["metadata"] / "logger_sanity_by_seed.csv", index=False)
    return {"baseline": baseline, "memory": memory, "recency": recency, "resolution": resolution}


def aggregate_financial_robustness(markets: Mapping[str, pd.DataFrame], cfg: Protocol, paths: Mapping[str, Path], seed_count: int) -> Dict[str, pd.DataFrame]:
    cost_mem: List[pd.DataFrame] = []
    cost_res: List[pd.DataFrame] = []
    risk_mem: List[pd.DataFrame] = []
    risk_res: List[pd.DataFrame] = []
    alt_parts: List[pd.DataFrame] = []

    for asset in cfg.assets:
        for seed_index in range(seed_count):
            # Cost grid. Baseline cost is allowed to reuse cached main log/qhat.
            for c in cfg.cost_grid:
                print(f"ROBUST cost {asset} seed={seed_index} c={c}", flush=True)
                mem, res = run_reward_robustness_seed(
                    asset, markets[asset], seed_index, cfg, paths, float(c), 0.0, f"cost_{c:.6f}"
                )
                cost_mem.append(mem)
                cost_res.append(res)

            print(f"ROBUST risk {asset} seed={seed_index} lambda={cfg.risk_lambda}", flush=True)
            mem, res = run_reward_robustness_seed(
                asset,
                markets[asset],
                seed_index,
                cfg,
                paths,
                cfg.baseline_cost,
                cfg.risk_lambda,
                f"risk_{cfg.risk_lambda:.6f}",
            )
            risk_mem.append(mem)
            risk_res.append(res)

            print(f"ROBUST alt-policy {asset} seed={seed_index}", flush=True)
            alt_parts.append(run_alternative_policy_contrast_seed(asset, markets[asset], seed_index, cfg, paths))

    return {
        "cost_memory": concat_nonempty(cost_mem),
        "cost_resolution": concat_nonempty(cost_res),
        "risk_memory": concat_nonempty(risk_mem),
        "risk_resolution": concat_nonempty(risk_res),
        "alt_resolution": concat_nonempty(alt_parts),
    }


def save_main_tables(markets: Mapping[str, pd.DataFrame], main: Mapping[str, pd.DataFrame], cfg: Protocol, paths: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    t01 = make_sample_table(markets)
    t02 = summarize_baseline_ope(main["baseline"])
    t03 = summarize_memory(main["memory"], main_only=True)
    curve, t04 = summarize_recency_coverage(main["recency"])
    t05 = summarize_policy_resolution(main["resolution"])
    t06 = make_cross_etf_helpers(t03, curve, t05)
    t01.to_csv(paths["tables"] / "T01_etf_sample.csv", index=False)
    t02.to_csv(paths["tables"] / "T02_baseline_ope.csv", index=False)
    t03.to_csv(paths["tables"] / "T03_memory_comparison.csv", index=False)
    t04.to_csv(paths["tables"] / "T04_recency_vs_coverage.csv", index=False)
    curve.to_csv(paths["tables"] / "T04b_recency_coverage_full_curve.csv", index=False)
    t05.to_csv(paths["tables"] / "T05_policy_resolution.csv", index=False)
    t06.to_csv(paths["tables"] / "T06_cross_etf_helpers.csv", index=False)
    # Fixed-decay is saved separately if enabled; never mixed into the main table.
    if cfg.include_fixed_decay_supplement:
        summarize_memory(main["memory"], main_only=False).to_csv(
            paths["supplement"] / "fixed_decay_memory_summary.csv", index=False
        )
    return {"T01": t01, "T02": t02, "T03": t03, "T04": t04, "curve": curve, "T05": t05, "T06": t06}


def save_robustness_tables(rob: Mapping[str, pd.DataFrame], cfg: Protocol, paths: Mapping[str, Path]) -> Dict[str, pd.DataFrame]:
    write_detail(rob["cost_memory"], paths["detail"] / "robust_cost_memory_detail.csv")
    write_detail(rob["cost_resolution"], paths["detail"] / "robust_cost_resolution_detail.csv")
    write_detail(rob["risk_memory"], paths["detail"] / "robust_risk_memory_detail.csv")
    write_detail(rob["risk_resolution"], paths["detail"] / "robust_risk_resolution_detail.csv")
    write_detail(rob["alt_resolution"], paths["detail"] / "robust_alternative_policy_resolution_detail.csv")

    t07 = summarize_reward_robustness(rob["cost_memory"], rob["cost_resolution"])
    t08 = summarize_reward_robustness(rob["risk_memory"], rob["risk_resolution"])
    alt = summarize_policy_resolution(rob["alt_resolution"])
    alt["candidate_history"] = cfg.alternative_candidate_history
    alt["baseline_history"] = cfg.baseline_policy_history
    t07.to_csv(paths["tables"] / "T07_transaction_cost_robustness.csv", index=False)
    t08.to_csv(paths["tables"] / "T08_risk_reward_robustness.csv", index=False)
    alt.to_csv(paths["tables"] / "T09_alternative_policy_contrast.csv", index=False)
    return {"T07": t07, "T08": t08, "T09": alt}


def run_supplements(markets: Mapping[str, pd.DataFrame], cfg: Protocol, paths: Mapping[str, Path]) -> None:
    if cfg.run_exploration_supplement:
        parts = []
        for asset in cfg.assets:
            parts.append(experiment_exploration_supplement(asset, markets[asset], cfg, paths))
        d = concat_nonempty(parts)
        d.to_csv(paths["supplement"] / "exploration_detail.csv", index=False)
        if not d.empty:
            d.groupby(["asset", "epsilon"], as_index=False).agg(
                MAE_bp=("abs_error_bp", "mean"), Mean_ESS=("ess", "mean")
            ).to_csv(paths["supplement"] / "exploration_summary.csv", index=False)
    if cfg.run_synthetic_lcb_supplement:
        d = synthetic_lcb_stress(cfg)
        d.to_csv(paths["supplement"] / "synthetic_lcb_detail.csv", index=False)
        d.groupby("true_delta", as_index=False).agg(
            Point_Adoption=("point_adopt", "mean"), LCB_Adoption=("lcb_adopt", "mean")
        ).to_csv(paths["supplement"] / "synthetic_lcb_summary.csv", index=False)


def run_full(cfg: Protocol, root: Path, mode: str) -> None:
    validate_protocol(cfg)
    paths = ensure_dirs(root)
    script_path = Path(globals().get(
        "__file__", Path.cwd() / "offline_policy_resolution_pipeline.py")).resolve()
    enforce_protocol_lock(cfg, paths, script_path, mode)
    save_environment_metadata(cfg, paths, script_path)
    print(f"Protocol hash: {protocol_hash(cfg)}")
    print(f"Output: {root}")

    markets = load_or_build_all_assets(cfg, paths)
    make_sample_table(markets).to_csv(paths["tables"] / "T01_etf_sample.csv", index=False)

    if mode == "download":
        print("Data download/feature construction complete.")
        return

    seed_count = 1 if mode == "pilot" else cfg.behavior_seed_count
    main = aggregate_main(markets, cfg, paths, seed_count)
    main_tables = save_main_tables(markets, main, cfg, paths)
    plot_baseline_ope(main_tables["T02"], paths, cfg)
    plot_memory(main_tables["T03"], paths, cfg)
    plot_recency_coverage(main_tables["curve"], paths, cfg)
    plot_gap_vs_se(main["resolution"], paths, cfg)

    if mode == "pilot":
        print("\nPILOT COMPLETE. Use runtime only; do not interpret scientific outcomes.")
        return

    if mode in {"full", "robustness"}:
        rob = aggregate_financial_robustness(markets, cfg, paths, seed_count)
        rob_tables = save_robustness_tables(rob, cfg, paths)
        plot_cost_robustness(rob_tables["T07"], paths, cfg)

    if mode == "main":
        # Main mode intentionally does not fabricate robustness tables. Validation
        # for all final outputs is therefore reserved for full mode.
        print("Main experiments complete. Run --mode full for journal robustness tables.")
    else:
        run_supplements(markets, cfg, paths)
        validate_final_outputs(paths, cfg)
        print("\nFULL OUTPUT VALIDATION PASSED.")


# =============================================================================
# 20. Synthetic smoke test (no internet, no scientific interpretation)
# =============================================================================


def synthetic_market(n: int = 1800, seed: int = 123) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    r = 0.0002 + rng.normal(0, 0.012, size=n)
    p = 100.0 * np.exp(np.cumsum(r))
    idx = pd.bdate_range("2010-01-01", periods=n)
    raw = pd.DataFrame({"Adj Close": p, "Close": p}, index=idx)
    return raw


def smoke_test() -> None:
    cfg = replace(
        Protocol(),
        assets=("QQQ",),
        min_history=400,
        baseline_policy_history=252,
        candidate_policy_history=63,
        alternative_candidate_history=126,
        forward_horizon=40,
        anchor_spacing=40,
        static_eval_window=40,
        nuisance_warmup=100,
        nuisance_prediction_block=20,
        recent_evidence_grid=(0, 20, 40),
        bootstrap_resamples=30,
        bootstrap_block_length=10,
        behavior_seed_count=1,
        ts_mc_draws=30,
        include_fixed_decay_supplement=False,
        save_figures=False,
    )
    root = Path.cwd() / "_smoke_offline_policy_resolution"
    paths = ensure_dirs(root)
    raw = synthetic_market()
    market = build_market_features("QQQ", raw, cfg, paths)
    log, score = simulate_behavior_log("QQQ", market, 0, cfg, paths, cfg.baseline_cost, 0.0, cache=False)
    qhat = rolling_origin_qhat("QQQ", log, 0, cfg, paths, cfg.baseline_cost, 0.0, cache=False)
    b = experiment_baseline_ope("QQQ", log, score, qhat, 0, cfg)
    m = experiment_memory("QQQ", log, score, qhat, 0, cfg)
    r = experiment_recency_coverage("QQQ", log, score, qhat, 0, cfg)
    d = experiment_policy_resolution("QQQ", log, score, qhat, 0, cfg)
    assert not b.empty and not m.empty and not r.empty and not d.empty
    assert set(b["estimator"]) == {"DM", "IPS", "SNIPS", "DR"}
    assert set(m["memory_rule"]) >= {"all_history", "last_252", "last_126", "last_60"}
    print("Synthetic smoke test PASSED")
    print(f"baseline rows={len(b)}, memory rows={len(m)}, recency rows={len(r)}, resolution rows={len(d)}")


# =============================================================================
# 21. Portable command-line entry point
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen cross-ETF off-policy evaluation and policy-resolution "
            "experiments used in the accompanying paper."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for downloaded data, caches, tables, metadata, and figures.",
    )
    parser.add_argument(
        "--mode",
        choices=("download", "pilot", "main", "full", "robustness", "smoke"),
        default="full",
        help=(
            "download=data only; pilot=timing-only single-seed run; "
            "main=main experiments; full=main plus robustness and validation; "
            "robustness=full robustness path; smoke=synthetic no-internet smoke test."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.mode == "smoke":
        smoke_test()
        return

    cfg = Protocol()
    root = args.output_dir.expanduser().resolve()

    print("=" * 76)
    print("OFFLINE POLICY RESOLUTION — EMPIRICAL RUN")
    print(f"Mode              : {args.mode}")
    print(f"Results directory : {root}")
    print(f"Protocol hash     : {protocol_hash(cfg)}")
    print(f"Behavior seeds    : {cfg.behavior_seed_count} per ETF")
    print(f"Temperatures      : base={cfg.base_temperature}, candidate={cfg.candidate_temperature}")
    print("=" * 76)

    run_full(cfg, root, args.mode)


if __name__ == "__main__":
    main()
