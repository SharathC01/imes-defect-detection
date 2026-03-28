"""
src/models/train.py

Train all defect-detection models, log runs to MLflow, save best estimators.

Run with:
    python -m src.models.train

Inputs:
    data/processed/features_p5.csv   (54 rows)
    data/processed/features_fixed.csv (53 rows)

Train / test split (LOCKED):
    Train: groups 1-7
    Test:  groups 8-9
    CV:    LeaveOneGroupOut on groups 1-7 only
"""

from __future__ import annotations

import joblib
import warnings
from pathlib import Path
from typing import Any

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, LeaveOneGroupOut, RandomizedSearchCV
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "data" / "processed" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TRAIN_GROUPS = list(range(1, 8))   # groups 1-7
TEST_GROUPS  = [8, 9]

EXPERIMENT_NAME = "imes-defect-detection"


# ---------------------------------------------------------------------------
# Parameter spaces
# ---------------------------------------------------------------------------

PARAM_SPACES: dict[str, dict[str, list[Any]]] = {
    "LogisticRegression": {
        "model__C":       [0.001, 0.01, 0.1, 1, 10, 100],
        "model__penalty": ["l1", "l2"],
        "model__solver":  ["liblinear"],
    },
    "SVC": {
        "model__C":      [0.1, 1, 10, 100],
        "model__kernel": ["rbf", "linear"],
        "model__gamma":  ["scale", "auto"],
    },
    "RandomForestClassifier": {
        "model__n_estimators":    [50, 100, 200],
        "model__max_depth":       [3, 5, 10, None],
        "model__min_samples_leaf":[1, 2, 5],
        "model__max_features":    ["sqrt", "log2"],
    },
    "GradientBoostingClassifier": {
        "model__n_estimators":  [50, 100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth":     [2, 3, 5],
        "model__subsample":     [0.7, 0.8, 1.0],
    },
    "Ridge": {
        "model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "RandomForestRegressor": {
        "model__n_estimators":    [50, 100, 200],
        "model__max_depth":       [3, 5, 10, None],
        "model__min_samples_leaf":[1, 2, 5],
        "model__max_features":    ["sqrt", "log2"],
    },
    "GradientBoostingRegressor": {
        "model__n_estimators":  [50, 100, 200],
        "model__learning_rate": [0.01, 0.05, 0.1],
        "model__max_depth":     [2, 3, 5],
        "model__subsample":     [0.7, 0.8, 1.0],
    },
    # MultiOutputRegressor wraps an estimator — prefix params with 'estimator__'
    "MultiOutputRegressor_Ridge": {
        "model__estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "MultiOutputRegressor_RandomForestRegressor": {
        "model__estimator__n_estimators":    [50, 100, 200],
        "model__estimator__max_depth":       [3, 5, 10, None],
        "model__estimator__min_samples_leaf":[1, 2, 5],
        "model__estimator__max_features":    ["sqrt", "log2"],
    },
    "MultiOutputRegressor_GradientBoostingRegressor": {
        "model__estimator__n_estimators":  [50, 100, 200],
        "model__estimator__learning_rate": [0.01, 0.05, 0.1],
        "model__estimator__max_depth":     [2, 3, 5],
        "model__estimator__subsample":     [0.7, 0.8, 1.0],
    },
}


def _narrow_grid(param_space: dict[str, list[Any]], best_params: dict[str, Any]) -> dict[str, list[Any]]:
    """Build a narrow grid around the best params from Stage 1.

    For each hyperparameter, include the best value plus one neighbour on
    either side (where the list has an order).  Categorical params that have
    no natural order keep only the best value found.
    """
    narrow: dict[str, list[Any]] = {}
    for key, candidates in param_space.items():
        best_val = best_params.get(key)
        if best_val is None:
            narrow[key] = candidates
            continue
        if best_val not in candidates:
            narrow[key] = [best_val]
            continue
        idx = candidates.index(best_val)
        lo  = max(0, idx - 1)
        hi  = min(len(candidates) - 1, idx + 1)
        narrow[key] = candidates[lo : hi + 1]
    return narrow


def _lookup_param_space(model_name: str) -> dict[str, list[Any]]:
    """Return the parameter space dict for a given model name."""
    if model_name in PARAM_SPACES:
        return PARAM_SPACES[model_name]
    # MultiOutputRegressor — name is "MultiOutputRegressor_<inner>"
    for key in PARAM_SPACES:
        if model_name.startswith(key):
            return PARAM_SPACES[key]
    raise KeyError(f"No param space found for model '{model_name}'")


def _build_pipeline(estimator: Any, use_imputer: bool = False) -> Pipeline:
    """Wrap an estimator in an imputer + scaler pipeline."""
    steps: list[tuple[str, Any]] = []
    if use_imputer:
        steps.append(("imputer", SimpleImputer(strategy="median")))
    steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps=steps)


def _two_stage_tune(
    pipeline: Pipeline,
    param_space: dict[str, list[Any]],
    X_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    scoring: str,
) -> tuple[Any, dict[str, Any]]:
    """Run two-stage hyperparameter tuning (RandomizedSearchCV → GridSearchCV).

    Returns (best_estimator, best_params).
    """
    logo = LeaveOneGroupOut()

    # Stage 1: RandomizedSearchCV
    stage1 = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_space,
        n_iter=50,
        cv=logo,
        scoring=scoring,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage1.fit(X_train, y_train, groups=groups_train)

    best_params_stage1 = stage1.best_params_

    # Stage 2: GridSearchCV on narrow grid
    narrow = _narrow_grid(param_space, best_params_stage1)
    stage2 = GridSearchCV(
        estimator=pipeline,
        param_grid=narrow,
        cv=logo,
        scoring=scoring,
        n_jobs=-1,
        refit=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stage2.fit(X_train, y_train, groups=groups_train)

    return stage2.best_estimator_, stage2.best_params_


# ---------------------------------------------------------------------------
# run_classification_problem
# ---------------------------------------------------------------------------

def run_classification_problem(
    problem_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    models: list[tuple[str, Any, dict[str, list[Any]]]],
    mlflow_experiment_id: str,
    use_imputer: bool = False,
) -> pd.DataFrame:
    """Train each classifier, log to MLflow, return comparison DataFrame.

    Returns DataFrame with columns: model, cv_f1, test_f1, test_auc, best_params
    """
    print("=" * 60)
    print(f"PROBLEM: {problem_name}")
    print("=" * 60)

    records = []

    for model_name, estimator, param_space in models:
        run_name = f"{problem_name}{model_name.lower().replace(' ', '_')}"
        print(f"\n  Training: {run_name}")

        pipeline = _build_pipeline(estimator, use_imputer=use_imputer)

        best_est, best_params = _two_stage_tune(
            pipeline=pipeline,
            param_space=param_space,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            scoring="f1_weighted",
        )

        # CV score using LOGO on training set with best params
        logo = LeaveOneGroupOut()
        cv_scores: list[float] = []
        for train_idx, val_idx in logo.split(X_train, y_train, groups=groups_train):
            X_cv_tr, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_tr, y_cv_val = y_train[train_idx], y_train[val_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_est.fit(X_cv_tr, y_cv_tr)
            preds = best_est.predict(X_cv_val)
            cv_scores.append(f1_score(y_cv_val, preds, average="weighted", zero_division=0))

        cv_f1 = float(np.mean(cv_scores))

        # Refit on full training set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_est.fit(X_train, y_train)

        y_pred = best_est.predict(X_test)
        test_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # AUC (binary or multi-class OvR)
        try:
            if hasattr(best_est, "predict_proba"):
                y_score = best_est.predict_proba(X_test)
            else:
                y_score = best_est.decision_function(X_test)
            n_classes = len(np.unique(y_test))
            if n_classes == 2:
                if y_score.ndim > 1:
                    y_score = y_score[:, 1]
                test_auc = roc_auc_score(y_test, y_score)
            else:
                test_auc = roc_auc_score(y_test, y_score, multi_class="ovr", average="weighted")
        except Exception:
            test_auc = float("nan")

        # Strip pipeline prefix from logged params
        logged_params = {
            k.replace("model__", "").replace("model__estimator__", "estimator__"): v
            for k, v in best_params.items()
        }
        metrics = {
            "cv_f1_weighted": cv_f1,
            "test_f1_weighted": test_f1,
            "test_auc": test_auc,
        }

        model_path = MODELS_DIR / f"{problem_name}{model_name}.joblib"
        joblib.dump(best_est, model_path)
        print(f"    Saved model to {model_path}")

        with mlflow.start_run(run_name=run_name, experiment_id=mlflow_experiment_id):
            mlflow.log_params(logged_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_est, "model")

        print(f"    cv_f1={cv_f1:.3f}  test_f1={test_f1:.3f}  test_auc={test_auc:.3f}")
        records.append({
            "model":       model_name,
            "cv_f1":       cv_f1,
            "test_f1":     test_f1,
            "test_auc":    test_auc,
            "best_params": logged_params,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# run_regression_problem
# ---------------------------------------------------------------------------

def run_regression_problem(
    problem_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    groups_train: np.ndarray,
    models: list[tuple[str, Any, dict[str, list[Any]]]],
    mlflow_experiment_id: str,
    use_imputer: bool = False,
) -> pd.DataFrame:
    """Train each regressor, log to MLflow, return comparison DataFrame.

    Returns DataFrame with columns: model, cv_mae, test_mae, test_r2, best_params
    """
    print("=" * 60)
    print(f"PROBLEM: {problem_name}")
    print("=" * 60)

    records = []

    for model_name, estimator, param_space in models:
        run_name = f"{problem_name}{model_name.lower().replace(' ', '_')}"
        print(f"\n  Training: {run_name}")

        pipeline = _build_pipeline(estimator, use_imputer=use_imputer)

        best_est, best_params = _two_stage_tune(
            pipeline=pipeline,
            param_space=param_space,
            X_train=X_train,
            y_train=y_train,
            groups_train=groups_train,
            scoring="neg_mean_absolute_error",
        )

        # CV score using LOGO on training set with best params
        logo = LeaveOneGroupOut()
        cv_maes: list[float] = []
        for train_idx, val_idx in logo.split(X_train, y_train, groups=groups_train):
            X_cv_tr, X_cv_val = X_train[train_idx], X_train[val_idx]
            y_cv_tr, y_cv_val = y_train[train_idx], y_train[val_idx]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                best_est.fit(X_cv_tr, y_cv_tr)
            preds = best_est.predict(X_cv_val)
            # For multi-output, average MAE across outputs
            if y_cv_val.ndim > 1:
                mae_val = float(np.mean([
                    mean_absolute_error(y_cv_val[:, i], preds[:, i])
                    for i in range(y_cv_val.shape[1])
                ]))
            else:
                mae_val = mean_absolute_error(y_cv_val, preds)
            cv_maes.append(mae_val)

        cv_mae = float(np.mean(cv_maes))

        # Refit on full training set
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_est.fit(X_train, y_train)

        y_pred = best_est.predict(X_test)

        if y_test.ndim > 1:
            test_mae = float(np.mean([
                mean_absolute_error(y_test[:, i], y_pred[:, i])
                for i in range(y_test.shape[1])
            ]))
            test_r2 = float(np.mean([
                r2_score(y_test[:, i], y_pred[:, i])
                for i in range(y_test.shape[1])
            ]))
        else:
            test_mae = mean_absolute_error(y_test, y_pred)
            test_r2  = r2_score(y_test, y_pred)

        logged_params = {
            k.replace("model__", "").replace("model__estimator__", "estimator__"): v
            for k, v in best_params.items()
        }
        metrics = {
            "cv_mae":  cv_mae,
            "test_mae": test_mae,
            "test_r2":  test_r2,
        }

        model_path = MODELS_DIR / f"{problem_name}{model_name}.joblib"
        joblib.dump(best_est, model_path)
        print(f"    Saved model to {model_path}")

        with mlflow.start_run(run_name=run_name, experiment_id=mlflow_experiment_id):
            mlflow.log_params(logged_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_est, "model")

        print(f"    cv_mae={cv_mae:.3f}  test_mae={test_mae:.3f}  test_r2={test_r2:.3f}")
        records.append({
            "model":       model_name,
            "cv_mae":      cv_mae,
            "test_mae":    test_mae,
            "test_r2":     test_r2,
            "best_params": logged_params,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    p5     = pd.read_csv(PROCESSED / "features_p5.csv")
    fixed  = pd.read_csv(PROCESSED / "features_fixed.csv")

    # ------------------------------------------------------------------
    # 2. Train / test splits
    # ------------------------------------------------------------------
    p5_train  = p5[p5["group"].isin(TRAIN_GROUPS)].reset_index(drop=True)
    p5_test   = p5[p5["group"].isin(TEST_GROUPS)].reset_index(drop=True)
    fix_train = fixed[fixed["group"].isin(TRAIN_GROUPS)].reset_index(drop=True)
    fix_test  = fixed[fixed["group"].isin(TEST_GROUPS)].reset_index(drop=True)

    groups_p5_train  = p5_train["group"].to_numpy()
    groups_fix_train = fix_train["group"].to_numpy()

    # ------------------------------------------------------------------
    # 3. MLflow experiment
    # ------------------------------------------------------------------
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    exp = mlflow.set_experiment(EXPERIMENT_NAME)
    exp_id = exp.experiment_id

    # ------------------------------------------------------------------
    # 4. Problem 1 — Frequency binary detection
    # ------------------------------------------------------------------
    FREQ_FEATURES = [
        "peak_magnitude", "dominant_freq_hz",
        "freq_x", "freq_y", "freq_z",
        "mag_x", "mag_y", "mag_z",
    ]
    X_freq_tr = p5_train[FREQ_FEATURES].to_numpy()
    y_freq_tr = p5_train["freq_defect"].to_numpy()
    X_freq_te = p5_test[FREQ_FEATURES].to_numpy()
    y_freq_te = p5_test["freq_defect"].to_numpy()

    freq_models: list[tuple[str, Any, dict[str, list[Any]]]] = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
            PARAM_SPACES["LogisticRegression"],
        ),
        (
            "SVC",
            SVC(class_weight="balanced", random_state=RANDOM_STATE, probability=True),
            PARAM_SPACES["SVC"],
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            PARAM_SPACES["RandomForestClassifier"],
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            PARAM_SPACES["GradientBoostingClassifier"],
        ),
    ]

    freq_results = run_classification_problem(
        problem_name="frequency_",
        X_train=X_freq_tr,
        y_train=y_freq_tr,
        X_test=X_freq_te,
        y_test=y_freq_te,
        groups_train=groups_p5_train,
        models=freq_models,
        mlflow_experiment_id=exp_id,
        use_imputer=False,
    )

    # ------------------------------------------------------------------
    # 5. Problem 2a — Inclination binary detection
    # ------------------------------------------------------------------
    INCL_FEATURES = [
        "peak_z", "signed_peak_z", "peak_jerk",
        "impulse_energy", "rise_time", "post_impact_rms",
    ]
    X_incl_tr = p5_train[INCL_FEATURES].to_numpy()
    y_incl_tr = p5_train["incl_detectable"].to_numpy()
    X_incl_te = p5_test[INCL_FEATURES].to_numpy()
    y_incl_te = p5_test["incl_detectable"].to_numpy()

    incl_bin_models: list[tuple[str, Any, dict[str, list[Any]]]] = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
            PARAM_SPACES["LogisticRegression"],
        ),
        (
            "SVC",
            SVC(class_weight="balanced", random_state=RANDOM_STATE, probability=True),
            PARAM_SPACES["SVC"],
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            PARAM_SPACES["RandomForestClassifier"],
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            PARAM_SPACES["GradientBoostingClassifier"],
        ),
    ]

    incl_bin_results = run_classification_problem(
        problem_name="inclination_binary_",
        X_train=X_incl_tr,
        y_train=y_incl_tr,
        X_test=X_incl_te,
        y_test=y_incl_te,
        groups_train=groups_p5_train,
        models=incl_bin_models,
        mlflow_experiment_id=exp_id,
        use_imputer=True,
    )

    # ------------------------------------------------------------------
    # 6. Problem 2b — Inclination angle regression (Loc4 rows only)
    # ------------------------------------------------------------------
    p5_incl_tr = p5_train[p5_train["incl_detectable"] == 1].reset_index(drop=True)
    p5_incl_te = p5_test[p5_test["incl_detectable"] == 1].reset_index(drop=True)
    groups_incl_tr = p5_incl_tr["group"].to_numpy()

    X_incl_reg_tr = p5_incl_tr[INCL_FEATURES].to_numpy()
    y_incl_reg_tr = p5_incl_tr["incl_degree"].to_numpy()
    X_incl_reg_te = p5_incl_te[INCL_FEATURES].to_numpy()
    y_incl_reg_te = p5_incl_te["incl_degree"].to_numpy()

    incl_reg_models: list[tuple[str, Any, dict[str, list[Any]]]] = [
        (
            "Ridge",
            Ridge(random_state=RANDOM_STATE),
            PARAM_SPACES["Ridge"],
        ),
        (
            "RandomForestRegressor",
            RandomForestRegressor(random_state=RANDOM_STATE),
            PARAM_SPACES["RandomForestRegressor"],
        ),
        (
            "GradientBoostingRegressor",
            GradientBoostingRegressor(random_state=RANDOM_STATE),
            PARAM_SPACES["GradientBoostingRegressor"],
        ),
    ]

    incl_reg_results = run_regression_problem(
        problem_name="inclination_regression_",
        X_train=X_incl_reg_tr,
        y_train=y_incl_reg_tr,
        X_test=X_incl_reg_te,
        y_test=y_incl_reg_te,
        groups_train=groups_incl_tr,
        models=incl_reg_models,
        mlflow_experiment_id=exp_id,
        use_imputer=True,
    )

    # ------------------------------------------------------------------
    # 7. Problem 3 — Belt speed multi-output regression (all 54 rows)
    # ------------------------------------------------------------------
    SPEED_FEATURES = [
        "journey_duration", "duration_delta",
        "rms_full", "rms_first_half", "rms_second_half", "rms_ratio",
    ]
    X_speed_tr = p5_train[SPEED_FEATURES].to_numpy()
    y_speed_tr = p5_train[["rail2_speed", "rail3_speed"]].to_numpy()
    X_speed_te = p5_test[SPEED_FEATURES].to_numpy()
    y_speed_te = p5_test[["rail2_speed", "rail3_speed"]].to_numpy()

    speed_models: list[tuple[str, Any, dict[str, list[Any]]]] = [
        (
            "MultiOutputRegressor_Ridge",
            MultiOutputRegressor(Ridge()),
            PARAM_SPACES["MultiOutputRegressor_Ridge"],
        ),
        (
            "MultiOutputRegressor_RandomForestRegressor",
            MultiOutputRegressor(RandomForestRegressor(random_state=RANDOM_STATE)),
            PARAM_SPACES["MultiOutputRegressor_RandomForestRegressor"],
        ),
        (
            "MultiOutputRegressor_GradientBoostingRegressor",
            MultiOutputRegressor(GradientBoostingRegressor(random_state=RANDOM_STATE)),
            PARAM_SPACES["MultiOutputRegressor_GradientBoostingRegressor"],
        ),
    ]

    speed_results = run_regression_problem(
        problem_name="belt_speed_",
        X_train=X_speed_tr,
        y_train=y_speed_tr,
        X_test=X_speed_te,
        y_test=y_speed_te,
        groups_train=groups_p5_train,
        models=speed_models,
        mlflow_experiment_id=exp_id,
        use_imputer=False,
    )

    # ------------------------------------------------------------------
    # 8. Problem 4 — Damping binary detection
    #    Exclude rows where freq_defect == 1 AND damping_present == 0
    # ------------------------------------------------------------------
    DAMPING_FEATURES = [
        "rms_P1", "rms_P2", "rms_P3", "rms_P4",
        "rms_ratio_P1", "rms_ratio_P2", "rms_ratio_P3", "rms_ratio_P4",
        "min_rms_ratio",
        "rms_ratio_P3_to_P1", "rms_ratio_P4_to_P1",
        "rms_ratio_P3_to_P2", "rms_ratio_P4_to_P2",
    ]

    contaminated_mask_tr = (fix_train["freq_defect"] == 1) & (fix_train["damping_present"] == 0)
    contaminated_mask_te = (fix_test["freq_defect"] == 1)  & (fix_test["damping_present"] == 0)

    fix_train_clean = fix_train[~contaminated_mask_tr].reset_index(drop=True)
    fix_test_clean  = fix_test[~contaminated_mask_te].reset_index(drop=True)
    groups_fix_clean_tr = fix_train_clean["group"].to_numpy()

    X_damp_tr = fix_train_clean[DAMPING_FEATURES].to_numpy()
    y_damp_tr = fix_train_clean["damping_present"].to_numpy()
    X_damp_te = fix_test_clean[DAMPING_FEATURES].to_numpy()
    y_damp_te = fix_test_clean["damping_present"].to_numpy()

    damp_models: list[tuple[str, Any, dict[str, list[Any]]]] = [
        (
            "LogisticRegression",
            LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE, max_iter=1000),
            PARAM_SPACES["LogisticRegression"],
        ),
        (
            "SVC",
            SVC(class_weight="balanced", random_state=RANDOM_STATE, probability=True),
            PARAM_SPACES["SVC"],
        ),
        (
            "RandomForestClassifier",
            RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            PARAM_SPACES["RandomForestClassifier"],
        ),
        (
            "GradientBoostingClassifier",
            GradientBoostingClassifier(random_state=RANDOM_STATE),
            PARAM_SPACES["GradientBoostingClassifier"],
        ),
    ]

    damp_results = run_classification_problem(
        problem_name="damping_",
        X_train=X_damp_tr,
        y_train=y_damp_tr,
        X_test=X_damp_te,
        y_test=y_damp_te,
        groups_train=groups_fix_clean_tr,
        models=damp_models,
        mlflow_experiment_id=exp_id,
        use_imputer=False,
    )

    # ------------------------------------------------------------------
    # 9. Final summary tables
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    print("\n--- Problem 1: Frequency binary detection ---")
    print(freq_results[["model", "cv_f1", "test_f1", "test_auc"]].to_string(index=False))

    print("\n--- Problem 2a: Inclination binary detection ---")
    print(incl_bin_results[["model", "cv_f1", "test_f1", "test_auc"]].to_string(index=False))

    print("\n--- Problem 2b: Inclination angle regression ---")
    print(incl_reg_results[["model", "cv_mae", "test_mae", "test_r2"]].to_string(index=False))

    print("\n--- Problem 3: Belt speed regression ---")
    print(speed_results[["model", "cv_mae", "test_mae", "test_r2"]].to_string(index=False))

    print("\n--- Problem 4: Damping binary detection ---")
    print(damp_results[["model", "cv_f1", "test_f1", "test_auc"]].to_string(index=False))

    print("\nAll runs logged to MLflow experiment:", EXPERIMENT_NAME)
    print("View with: mlflow ui")


if __name__ == "__main__":
    main()
