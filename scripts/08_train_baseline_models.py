"""
Train baseline ML classifiers on pre-computed ECFP4 fingerprints with Bayesian
hyperparameter optimization via BayesOpt.

Supports three model types:
  - Logistic Regression
  - Random Forest
  - XGBoost

Hyperparameters are optimized by maximizing validation AUPRC. The final model is
trained with the best hyperparameters and evaluated on the validation set with
bootstrapped metrics (AUPRC, Precision, Recall, F1, Accuracy).

Usage:
    python scripts/12_train_baseline_models.py --model_type Logistic
    python scripts/12_train_baseline_models.py --model_type Random_forest
    python scripts/12_train_baseline_models.py --model_type XGBoost
    python scripts/12_train_baseline_models.py --model_type XGBoost --max_train_samples 500000
"""

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier

# =============================================================================
# Constants
# =============================================================================

N_BITS = 2048
FP_COLS = [f"fp_{i}" for i in range(N_BITS)]
MODEL_TYPES = ["Logistic", "Random_forest", "XGBoost"]

# =============================================================================
# Data Loading
# =============================================================================


def load_fingerprints(split: str, data_dir: Path, max_samples: int = None):
    """
    Load pre-computed ECFP4 fingerprints and labels from parquet.

    Returns:
        Tuple of (fingerprint_matrix [N, 2048], labels [N,])
    """
    parquet_path = data_dir / split / f"supcon_{split}_ecfp4.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"ECFP4 parquet not found: {parquet_path}")

    print(f"  Loading {parquet_path.name}...")
    df = pd.read_parquet(parquet_path)

    if max_samples is not None and len(df) > max_samples:
        print(f"  Subsampling from {len(df):,} to {max_samples:,} rows (stratified)...")
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df, train_size=max_samples, stratify=df["label"], random_state=42
        )

    fps = df[FP_COLS].to_numpy(dtype=np.float32)
    labels = df["label"].to_numpy()
    print(f"  Loaded {len(fps):,} samples (PKS ratio: {labels.mean():.3f})")
    return fps, labels


# =============================================================================
# Objective Functions (closures for BayesOpt)
# =============================================================================


def logistic_regression_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int = 1,
):
    """
    Objective function for Logistic Regression hyperparameter optimization.
    Optimizes regularization strength C and class_weight scaling.
    """

    def objective(C, class_weight_scale):
        model = LogisticRegression(
            C=float(C),
            solver="saga",
            max_iter=1000,
            class_weight={0: 1.0, 1: float(class_weight_scale)},
            n_jobs=n_jobs,
        )
        model.fit(X_train, y_train)
        y_val_predicted_probabilities = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_val_predicted_probabilities)
        return auprc

    return objective


def random_forest_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int = 1,
):
    """
    Objective function for Random Forest hyperparameter optimization.
    """

    def objective(n_estimators, max_depth, min_samples_split, min_samples_leaf):
        model = RandomForestClassifier(
            n_estimators=int(n_estimators),
            max_depth=int(max_depth),
            min_samples_split=int(min_samples_split),
            min_samples_leaf=int(min_samples_leaf),
            class_weight="balanced",
            n_jobs=n_jobs,
            random_state=42,
        )
        model.fit(X_train, y_train)
        y_val_predicted_probabilities = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_val_predicted_probabilities)
        return auprc

    return objective


def XGBC_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_jobs: int = 1,
):
    """
    Objective function for XGBoost hyperparameter optimization via Bayesian algorithm.
    """

    def objective(
        learning_rate,
        max_leaves,
        max_depth,
        reg_alpha,
        reg_lambda,
        n_estimators,
        min_child_weight,
        colsample_bytree,
        colsample_bylevel,
        colsample_bynode,
        subsample,
        scale_pos_weight,
    ):
        params = {
            "learning_rate": learning_rate,
            "max_leaves": int(max_leaves),
            "max_depth": int(max_depth),
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "n_estimators": int(n_estimators),
            "min_child_weight": min_child_weight,
            "colsample_bytree": colsample_bytree,
            "colsample_bylevel": colsample_bylevel,
            "colsample_bynode": colsample_bynode,
            "subsample": subsample,
            "scale_pos_weight": scale_pos_weight,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "tree_method": "hist",
            "n_jobs": n_jobs,
        }

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        y_val_predicted_probabilities = model.predict_proba(X_val)[:, 1]
        auprc = average_precision_score(y_val, y_val_predicted_probabilities)
        return auprc

    return objective


# =============================================================================
# Bayesian Hyperparameter Search
# =============================================================================


def run_bayesian_hyperparameter_search(
    model_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
    init_points: int,
    n_iter: int,
    n_jobs: int = 1,
) -> dict:
    """
    Conduct hyperparameter optimization using Bayesian optimization.
    Returns the best hyperparameters as a dictionary.
    """
    if model_type == "Logistic":
        objective = logistic_regression_objective(X_train, y_train, X_val, y_val, n_jobs)
        pbounds = {
            "C": (0.01, 10),
            "class_weight_scale": (1.0, 5.0),
        }

    elif model_type == "Random_forest":
        objective = random_forest_objective(X_train, y_train, X_val, y_val, n_jobs)
        pbounds = {
            "n_estimators": (100, 1000),
            "max_depth": (5, 50),
            "min_samples_split": (2, 20),
            "min_samples_leaf": (1, 10),
        }

    elif model_type == "XGBoost":
        objective = XGBC_objective(X_train, y_train, X_val, y_val, n_jobs)
        pbounds = {
            "learning_rate": (0.01, 0.5),
            "max_leaves": (20, 300),
            "max_depth": (1, 15),
            "reg_alpha": (0, 1.0),
            "reg_lambda": (0, 1.0),
            "n_estimators": (20, 300),
            "min_child_weight": (2, 10),
            "colsample_bytree": (0.5, 1.0),
            "colsample_bylevel": (0.5, 1.0),
            "colsample_bynode": (0.5, 1.0),
            "subsample": (0.4, 1.0),
            "scale_pos_weight": (1, 5),
        }

    else:
        raise ValueError(f"Unknown model_type: {model_type}. Choose from {MODEL_TYPES}")

    optimizer = BayesianOptimization(
        f=objective, pbounds=pbounds, random_state=random_state
    )

    optimizer.maximize(init_points=init_points, n_iter=n_iter)

    best_params = optimizer.max["params"]
    best_score = optimizer.max["target"]
    print(f"\nBest AUPRC: {best_score:.4f} achieved with {best_params}")

    return best_params


# =============================================================================
# Model Training with Optimized Hyperparameters
# =============================================================================


def train_model(
    model_type: str,
    opt_hyperparams: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int,
    n_jobs: int,
):
    """
    Train a model using the optimized hyperparameters from Bayesian search.
    """
    if model_type == "Logistic":
        model = LogisticRegression(
            C=opt_hyperparams["C"],
            solver="saga",
            max_iter=1000,
            class_weight={0: 1.0, 1: opt_hyperparams["class_weight_scale"]},
            random_state=random_state,
            n_jobs=n_jobs,
        )

    elif model_type == "Random_forest":
        model = RandomForestClassifier(
            n_estimators=int(opt_hyperparams["n_estimators"]),
            max_depth=int(opt_hyperparams["max_depth"]),
            min_samples_split=int(opt_hyperparams["min_samples_split"]),
            min_samples_leaf=int(opt_hyperparams["min_samples_leaf"]),
            class_weight="balanced",
            random_state=random_state,
            n_jobs=n_jobs,
        )

    elif model_type == "XGBoost":
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=n_jobs,
            learning_rate=opt_hyperparams["learning_rate"],
            max_leaves=int(opt_hyperparams["max_leaves"]),
            max_depth=int(opt_hyperparams["max_depth"]),
            reg_alpha=opt_hyperparams["reg_alpha"],
            reg_lambda=opt_hyperparams["reg_lambda"],
            n_estimators=int(opt_hyperparams["n_estimators"]),
            min_child_weight=opt_hyperparams["min_child_weight"],
            colsample_bytree=opt_hyperparams["colsample_bytree"],
            colsample_bylevel=opt_hyperparams["colsample_bylevel"],
            colsample_bynode=opt_hyperparams["colsample_bynode"],
            subsample=opt_hyperparams["subsample"],
            scale_pos_weight=opt_hyperparams["scale_pos_weight"],
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    model.fit(X_train, y_train)
    return model


# =============================================================================
# Bootstrap Evaluation
# =============================================================================


def _single_bootstrap(y_true, y_scores, metric_fn, rng_seed):
    """Compute a single bootstrap resample metric. Designed for parallel dispatch."""
    rng = np.random.RandomState(rng_seed)
    n_samples = len(y_true)
    indices = rng.choice(n_samples, n_samples, replace=True)
    try:
        return metric_fn(y_true[indices], y_scores[indices])
    except ValueError:
        return 0.0


def bootstrap_metric(
    y_true, y_scores, metric_fn, n_iterations=1000, n_jobs=-1, random_state=42
):
    """
    Calculate a bootstrapped metric with 95% confidence intervals.
    Uses joblib for parallel bootstrap resampling.

    Args:
        y_true: True labels.
        y_scores: Predicted probabilities or binary labels.
        metric_fn: Scoring function (y_true, y_scores) -> float.
        n_iterations: Number of bootstrap iterations.
        n_jobs: Number of parallel workers (-1 = all CPUs).
        random_state: Base seed for reproducible bootstrap samples.

    Returns:
        Tuple of (mean_score, lower_ci, upper_ci).
    """
    # Generate unique seeds for each bootstrap iteration for reproducibility
    seed_rng = np.random.RandomState(random_state)
    seeds = seed_rng.randint(0, 2**31, size=n_iterations)

    scores = Parallel(n_jobs=n_jobs)(
        delayed(_single_bootstrap)(y_true, y_scores, metric_fn, seed)
        for seed in seeds
    )

    scores = np.array(scores)
    mean_score = np.mean(scores)
    lower = np.percentile(scores, 2.5)
    upper = np.percentile(scores, 97.5)

    return mean_score, lower, upper


def evaluate_model(model, X_val, y_val, n_bootstrap=1000, n_jobs=-1):
    """
    Evaluate a trained model on validation data with parallelized bootstrapped metrics.
    """
    y_pred_proba = model.predict_proba(X_val)[:, 1]
    y_pred_binary = (y_pred_proba >= 0.5).astype(int)

    print("\nBootstrapped validation metrics (95% CI):")
    print("-" * 60)

    results = {}

    # AUPRC
    mean, lower, upper = bootstrap_metric(
        y_val, y_pred_proba, average_precision_score, n_bootstrap, n_jobs
    )
    print(f"  AUPRC:     {mean:.4f}  ({lower:.4f}, {upper:.4f})")
    results["mean AUPRC"] = mean
    results["AUPRC lower CI"] = lower
    results["AUPRC upper CI"] = upper

    # Precision
    mean, lower, upper = bootstrap_metric(
        y_val, y_pred_binary, precision_score, n_bootstrap, n_jobs
    )
    print(f"  Precision: {mean:.4f}  ({lower:.4f}, {upper:.4f})")
    results["mean Precision"] = mean
    results["Precision lower CI"] = lower
    results["Precision upper CI"] = upper

    # Recall
    mean, lower, upper = bootstrap_metric(
        y_val,
        y_pred_binary,
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        n_bootstrap,
        n_jobs,
    )
    print(f"  Recall:    {mean:.4f}  ({lower:.4f}, {upper:.4f})")
    results["mean Recall"] = mean
    results["Recall lower CI"] = lower
    results["Recall upper CI"] = upper

    # F1
    mean, lower, upper = bootstrap_metric(
        y_val,
        y_pred_binary,
        lambda yt, yp: f1_score(yt, yp, zero_division=0),
        n_bootstrap,
        n_jobs,
    )
    print(f"  F1:        {mean:.4f}  ({lower:.4f}, {upper:.4f})")
    results["mean F1"] = mean
    results["F1 lower CI"] = lower
    results["F1 upper CI"] = upper

    # Accuracy
    mean, lower, upper = bootstrap_metric(
        y_val, y_pred_binary, accuracy_score, n_bootstrap, n_jobs
    )
    print(f"  Accuracy:  {mean:.4f}  ({lower:.4f}, {upper:.4f})")
    results["mean Accuracy"] = mean
    results["Accuracy lower CI"] = lower
    results["Accuracy upper CI"] = upper

    return results


# =============================================================================
# Main
# =============================================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train baseline ML classifiers on ECFP4 fingerprints with "
        "Bayesian hyperparameter optimization."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        choices=MODEL_TYPES,
        help="Model type to train: Logistic, Random_forest, or XGBoost",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to data/ directory (default: auto-detect from script location)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Path to output directory (default: models/baselines/)",
    )
    parser.add_argument(
        "--random_state", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers (default: -1, all CPUs)",
    )
    parser.add_argument(
        "--init_points",
        type=int,
        default=5,
        help="Number of random exploration points for BayesOpt (default: 5)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=20,
        help="Number of Bayesian optimization iterations (default: 20)",
    )
    parser.add_argument(
        "--n_bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap iterations for evaluation (default: 1000)",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Subsample training data to this many rows (default: use all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve directories
    project_root = Path(__file__).resolve().parent.parent
    data_dir = Path(args.data_dir) if args.data_dir else project_root / "data"
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "baselines"

    # Create output subdirectories
    params_dir = output_dir / "params"
    results_dir = output_dir / "results"
    for d in [output_dir, params_dir, results_dir]:
        d.mkdir(parents=True, exist_ok=True)

    model_type = args.model_type
    print(f"\n{'='*60}")
    print(f"Training {model_type} baseline on ECFP4 fingerprints")
    print(f"{'='*60}")

    # --- Load data ---
    print("\nLoading data...")
    train_fps, train_labels = load_fingerprints(
        "train", data_dir, max_samples=args.max_train_samples
    )
    val_fps, val_labels = load_fingerprints("val", data_dir)

    # --- Bayesian hyperparameter optimization ---
    print(f"\nStarting Bayesian hyperparameter optimization for {model_type}")
    print(f"  init_points={args.init_points}, n_iter={args.n_iter}")
    start_time = time.time()

    opt_params = run_bayesian_hyperparameter_search(
        model_type=model_type,
        X_train=train_fps,
        y_train=train_labels,
        X_val=val_fps,
        y_val=val_labels,
        random_state=args.random_state,
        init_points=args.init_points,
        n_iter=args.n_iter,
        n_jobs=args.n_jobs,
    )

    elapsed = time.time() - start_time
    print(f"Hyperparameter optimization completed in {elapsed:.1f}s")

    # Save optimized hyperparameters
    params_path = params_dir / f"{model_type}_ecfp4_best_params.json"
    with open(params_path, "w") as f:
        json.dump(opt_params, f, indent=4)
    print(f"Saved best params to {params_path}")

    # --- Train final model with optimized hyperparameters ---
    print(f"\nTraining final {model_type} model with optimized hyperparameters...")
    start_time = time.time()

    model = train_model(
        model_type=model_type,
        opt_hyperparams=opt_params,
        X_train=train_fps,
        y_train=train_labels,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    elapsed = time.time() - start_time
    print(f"Model training completed in {elapsed:.1f}s")

    # Save trained model
    model_path = output_dir / f"{model_type}_ecfp4.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved trained model to {model_path}")

    # --- Evaluate on validation set ---
    print(f"\nEvaluating {model_type} on validation set...")
    results = evaluate_model(
        model, val_fps, val_labels, n_bootstrap=args.n_bootstrap, n_jobs=args.n_jobs
    )

    # Save validation results
    results_path = results_dir / f"{model_type}_ecfp4_val_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved validation results to {results_path}")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"  Model:       {model_path}")
    print(f"  Params:      {params_path}")
    print(f"  Val results: {results_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
