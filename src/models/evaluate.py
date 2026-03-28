"""
src/models/evaluate.py

Two evaluation tasks:

TASK 1: Evaluate best models on the G8-G9 holdout set (ground truth available).
        Compute metrics and save confusion-matrix / residual plots.

TASK 2: Run predictions on professor test cases (no ground truth in code).
        Load raw XLS files, extract features, apply models, print predictions.

Run with:
    python -m src.models.evaluate
"""

from __future__ import annotations

import joblib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for all environments
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
    roc_auc_score,
)

from src.data_loader import load_file
from src.features.belt_speed import extract_belt_speed_features
from src.features.damping import extract_damping_features
from src.features.frequency import extract_dominant_frequency
from src.features.inclination import extract_inclination_features
from src.preprocessor import detect_pickup_event, resample, segment_journey

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_RAW       = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
FIGURES_DIR    = DATA_PROCESSED / "eval_figures"
MODELS_DIR     = ROOT / "data" / "processed" / "models"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPERIMENT_NAME = "imes-defect-detection"

FREQ_FEATURES = [
    "peak_magnitude", "dominant_freq_hz",
    "freq_x", "freq_y", "freq_z",
    "mag_x", "mag_y", "mag_z",
]
INCL_FEATURES = [
    "peak_z", "signed_peak_z", "peak_jerk",
    "impulse_energy", "rise_time", "post_impact_rms",
]
SPEED_FEATURES = [
    "journey_duration", "duration_delta",
    "rms_full", "rms_first_half", "rms_second_half", "rms_ratio",
]
DAMP_FEATURES = [
    "rms_P1", "rms_P2", "rms_P3", "rms_P4",
    "rms_ratio_P1", "rms_ratio_P2", "rms_ratio_P3", "rms_ratio_P4",
    "min_rms_ratio",
    "rms_ratio_P3_to_P1", "rms_ratio_P4_to_P1",
    "rms_ratio_P3_to_P2", "rms_ratio_P4_to_P2",
]


# ---------------------------------------------------------------------------
# Helper — load and resample one XLS file
# ---------------------------------------------------------------------------
def _load_and_resample(path: Path) -> pd.DataFrame:
    """Load *path* and resample to 100 Hz."""
    df, fs = load_file(str(path))
    df = resample(df, fs_original=fs, fs_target=100.0)
    return df


# ---------------------------------------------------------------------------
# load_best_models
# ---------------------------------------------------------------------------
def load_best_models() -> dict:
    """Load best models from joblib files in MODELS_DIR.

    Returns
    -------
    dict with keys:
        'frequency', 'inclination_binary', 'inclination_regression',
        'belt_speed', 'damping'
    """
    return {
        "frequency": joblib.load(
            MODELS_DIR / "frequency_SVC.joblib"),
        "inclination_binary": joblib.load(
            MODELS_DIR / "inclination_binary_SVC.joblib"),
        "inclination_regression": joblib.load(
            MODELS_DIR / "inclination_regression_GradientBoostingRegressor.joblib"),
        "belt_speed": joblib.load(
            MODELS_DIR / "belt_speed_MultiOutputRegressor_RandomForestRegressor.joblib"),
        "damping": joblib.load(
            MODELS_DIR / "damping_RandomForestClassifier.joblib"),
    }


# ---------------------------------------------------------------------------
# load_g0_references
# ---------------------------------------------------------------------------
def load_g0_references() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Load G0 P5 journey segment and G0 fixed phones P1-P4.

    Returns
    -------
    (g0_p5_journey_df, g0_phones_dict)
        g0_p5_journey_df : journey segment of G0 P5
        g0_phones_dict   : {'P1': df, 'P2': df, 'P3': df, 'P4': df}
    """
    g0_dir = DATA_RAW / "G0"

    # G0 P5
    df_g0_p5 = _load_and_resample(g0_dir / "G0_P5_case_perfect.xls")
    pickup_g0 = detect_pickup_event(df_g0_p5)
    g0_p5_journey = segment_journey(df_g0_p5, pickup_g0)["journey"]

    # G0 fixed phones
    g0_phones: dict[str, pd.DataFrame] = {}
    for p in [1, 2, 3, 4]:
        g0_phones[f"P{p}"] = _load_and_resample(g0_dir / f"G0_P{p}_case_perfect.xls")

    return g0_p5_journey, g0_phones


# ---------------------------------------------------------------------------
# extract_professor_case_features
# ---------------------------------------------------------------------------
def extract_professor_case_features(
    case_n: int,
    g0_p5_journey: pd.DataFrame,
    g0_phones: dict[str, pd.DataFrame],
) -> dict:
    """Load and extract all features for professor test case *case_n*.

    Loads ``data/raw/Test Case/case_{case_n}/phone_{p}.xls``
    for p in [1, 2, 3, 4, 5].

    Parameters
    ----------
    case_n      : int — test case number (1-5)
    g0_p5_journey : G0 P5 journey segment (for belt-speed delta feature)
    g0_phones   : dict of G0 fixed-phone DataFrames

    Returns
    -------
    dict — flat dict with all feature keys needed by every problem.

    Raises
    ------
    Exception — propagated to caller; caller handles with try/except.
    """
    case_dir = DATA_RAW / "Test Case" / f"case_{case_n}"

    # --- Phone 5 (moving) ---
    df_p5 = _load_and_resample(case_dir / "phone_5.xls")
    pickup_time = detect_pickup_event(df_p5)
    segments    = segment_journey(df_p5, pickup_time)

    freq_feats  = extract_dominant_frequency(segments["journey"])
    incl_feats  = extract_inclination_features(segments["inclination"])
    speed_feats = extract_belt_speed_features(segments["journey"], g0_p5_journey)

    # --- Fixed phones P1-P4 ---
    phones: dict[str, pd.DataFrame] = {}
    for p in [1, 2, 3, 4]:
        phones[f"P{p}"] = _load_and_resample(case_dir / f"phone_{p}.xls")

    damp_feats = extract_damping_features(phones, g0_phones)

    features: dict = {
        **freq_feats,
        **incl_feats,
        **speed_feats,
        **damp_feats,
    }
    return features


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Plot and save a confusion matrix to *out_path*."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def _save_residual_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Scatter plot of predicted vs actual with a perfect-fit diagonal."""
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(y_true, y_pred, alpha=0.7, edgecolors="k", linewidths=0.5)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "r--", linewidth=1, label="perfect fit")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


# ---------------------------------------------------------------------------
# TASK 1 — G8-G9 holdout evaluation
# ---------------------------------------------------------------------------

def task1_holdout_evaluation(models: dict) -> None:
    """Evaluate all models on the G8-G9 holdout set."""
    print("=" * 60)
    print("TASK 1: G8-G9 HOLDOUT EVALUATION")
    print("=" * 60)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    df_p5    = pd.read_csv(DATA_PROCESSED / "features_p5.csv")
    df_fixed = pd.read_csv(DATA_PROCESSED / "features_fixed.csv")

    test_p5    = df_p5[df_p5["group"] >= 8].reset_index(drop=True)
    test_fixed = df_fixed[df_fixed["group"] >= 8].reset_index(drop=True)

    results: list[dict] = []

    # ------------------------------------------------------------------
    # Problem 1 — Frequency binary
    # ------------------------------------------------------------------
    print("\n[Problem 1] Frequency binary detection")
    X_freq = test_p5[FREQ_FEATURES].to_numpy()
    y_freq = test_p5["freq_defect"].to_numpy()
    pred_freq = models["frequency"].predict(X_freq)

    f1_freq  = f1_score(y_freq, pred_freq, average="weighted", zero_division=0)
    acc_freq = float((pred_freq == y_freq).mean())
    print(f"  F1 (weighted) = {f1_freq:.3f}  Accuracy = {acc_freq:.3f}")

    _save_confusion_matrix(
        y_freq, pred_freq,
        title="Frequency detection — G8-G9",
        out_path=FIGURES_DIR / "cm_frequency.png",
    )
    results.append({"Problem": "Frequency (binary)", "Metric": "F1", "Score": f1_freq})

    # ------------------------------------------------------------------
    # Problem 2a — Inclination binary
    # ------------------------------------------------------------------
    print("\n[Problem 2a] Inclination binary detection")
    X_incl = test_p5[INCL_FEATURES].to_numpy()
    y_incl = test_p5["incl_detectable"].to_numpy()
    # The pipeline has a fitted SimpleImputer — NaN rise_time handled automatically
    pred_incl = models["inclination_binary"].predict(X_incl)

    f1_incl  = f1_score(y_incl, pred_incl, average="weighted", zero_division=0)
    acc_incl = float((pred_incl == y_incl).mean())
    print(f"  F1 (weighted) = {f1_incl:.3f}  Accuracy = {acc_incl:.3f}")

    _save_confusion_matrix(
        y_incl, pred_incl,
        title="Inclination detection — G8-G9",
        out_path=FIGURES_DIR / "cm_inclination_binary.png",
    )
    results.append({"Problem": "Inclination (binary)", "Metric": "F1", "Score": f1_incl})

    # ------------------------------------------------------------------
    # Problem 2b — Inclination angle regression (Loc4 rows only)
    # ------------------------------------------------------------------
    print("\n[Problem 2b] Inclination angle regression")
    test_incl_reg = test_p5[test_p5["incl_detectable"] == 1].reset_index(drop=True)

    if len(test_incl_reg) == 0:
        print("  No Loc4 inclination rows in G8-G9 holdout — skipping regression.")
        mae_incl_reg = float("nan")
        r2_incl_reg  = float("nan")
    else:
        X_incl_reg = test_incl_reg[INCL_FEATURES].to_numpy()
        y_incl_reg = test_incl_reg["incl_degree"].to_numpy()
        pred_incl_reg = models["inclination_regression"].predict(X_incl_reg)

        mae_incl_reg = mean_absolute_error(y_incl_reg, pred_incl_reg)
        r2_incl_reg  = r2_score(y_incl_reg, pred_incl_reg)
        print(f"  MAE = {mae_incl_reg:.3f} degrees  R² = {r2_incl_reg:.3f}")

        _save_residual_scatter(
            y_incl_reg, pred_incl_reg,
            title="Inclination angle — G8-G9",
            xlabel="Actual incl_degree",
            ylabel="Predicted incl_degree",
            out_path=FIGURES_DIR / "residuals_inclination.png",
        )

    results.append({"Problem": "Inclination (angle)", "Metric": "MAE (degrees)", "Score": mae_incl_reg})

    # ------------------------------------------------------------------
    # Problem 3 — Belt speed multi-output regression
    # ------------------------------------------------------------------
    print("\n[Problem 3] Belt speed regression")
    X_speed = test_p5[SPEED_FEATURES].to_numpy()
    y_speed = test_p5[["rail2_speed", "rail3_speed"]].to_numpy()
    pred_speed = models["belt_speed"].predict(X_speed)

    mae_rail2 = mean_absolute_error(y_speed[:, 0], pred_speed[:, 0])
    mae_rail3 = mean_absolute_error(y_speed[:, 1], pred_speed[:, 1])
    r2_rail2  = r2_score(y_speed[:, 0], pred_speed[:, 0])
    r2_rail3  = r2_score(y_speed[:, 1], pred_speed[:, 1])
    print(f"  Rail2 — MAE = {mae_rail2:.2f}  R² = {r2_rail2:.3f}")
    print(f"  Rail3 — MAE = {mae_rail3:.2f}  R² = {r2_rail3:.3f}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    for ax, actual, predicted, label in [
        (ax1, y_speed[:, 0], pred_speed[:, 0], "Rail 2"),
        (ax2, y_speed[:, 1], pred_speed[:, 1], "Rail 3"),
    ]:
        ax.scatter(actual, predicted, alpha=0.7, edgecolors="k", linewidths=0.5)
        lo = min(actual.min(), predicted.min())
        hi = max(actual.max(), predicted.max())
        margin = (hi - lo) * 0.05
        ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
                "r--", linewidth=1, label="perfect fit")
        ax.set_xlabel(f"Actual {label} speed (%)")
        ax.set_ylabel(f"Predicted {label} speed (%)")
        ax.set_title(f"{label} — G8-G9")
        ax.legend()
    fig.tight_layout()
    belt_fig_path = FIGURES_DIR / "residuals_belt_speed.png"
    fig.savefig(belt_fig_path, dpi=150)
    plt.close(fig)
    print(f"    Saved: {belt_fig_path.name}")

    results.append({"Problem": "Belt speed", "Metric": "MAE rail2", "Score": mae_rail2})
    results.append({"Problem": "Belt speed", "Metric": "MAE rail3", "Score": mae_rail3})

    # ------------------------------------------------------------------
    # Problem 4 — Damping binary (exclude freq_defect==1, damping==0)
    # ------------------------------------------------------------------
    print("\n[Problem 4] Damping binary detection")
    contaminated = (test_fixed["freq_defect"] == 1) & (test_fixed["damping_present"] == 0)
    test_damp = test_fixed[~contaminated].reset_index(drop=True)

    X_damp = test_damp[DAMP_FEATURES].to_numpy()
    y_damp = test_damp["damping_present"].to_numpy()
    pred_damp = models["damping"].predict(X_damp)

    f1_damp = f1_score(y_damp, pred_damp, average="weighted", zero_division=0)

    try:
        if hasattr(models["damping"], "predict_proba"):
            y_score = models["damping"].predict_proba(X_damp)[:, 1]
        else:
            y_score = models["damping"].decision_function(X_damp)
        auc_damp = roc_auc_score(y_damp, y_score)
    except Exception:
        auc_damp = float("nan")

    print(f"  F1 (weighted) = {f1_damp:.3f}  AUC = {auc_damp:.3f}")

    _save_confusion_matrix(
        y_damp, pred_damp,
        title="Damping detection — G8-G9",
        out_path=FIGURES_DIR / "cm_damping.png",
    )
    results.append({"Problem": "Damping", "Metric": "AUC", "Score": auc_damp})

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n=== G8-G9 HOLDOUT RESULTS ===")
    header = f"{'Problem':<22} | {'Metric':<15} | {'Score':>6}"
    sep    = "-" * 22 + "-+-" + "-" * 15 + "-+-" + "-" * 7
    print(header)
    print(sep)
    for row in results:
        score_str = f"{row['Score']:.3f}" if not (isinstance(row['Score'], float) and np.isnan(row['Score'])) else "  N/A"
        print(f"{row['Problem']:<22} | {row['Metric']:<15} | {score_str:>6}")


# ---------------------------------------------------------------------------
# TASK 2 — Professor test cases
# ---------------------------------------------------------------------------

def task2_professor_predictions(models: dict) -> None:
    """Extract features from professor test files and print predictions."""
    print("\n" + "=" * 60)
    print("TASK 2: PROFESSOR TEST CASES — PREDICTIONS")
    print("=" * 60)

    print("\nLoading G0 reference data…")
    g0_p5_journey, g0_phones = load_g0_references()
    print("G0 references loaded.\n")

    rows: list[dict] = []

    for case_n in [1, 2, 3, 4, 5]:
        print(f"  Processing case {case_n}…")
        try:
            features = extract_professor_case_features(case_n, g0_p5_journey, g0_phones)

            # Frequency
            X_freq = pd.DataFrame([features])[FREQ_FEATURES]
            freq_detected = int(models["frequency"].predict(X_freq)[0])
            freq_hz       = features["dominant_freq_hz"] if freq_detected else 0.0
            freq_magnitude = features["peak_magnitude"]

            # Inclination
            X_incl = pd.DataFrame([features])[INCL_FEATURES]
            incl_detected = int(models["inclination_binary"].predict(X_incl)[0])
            incl_degree   = 0.0
            if incl_detected:
                incl_degree = float(models["inclination_regression"].predict(X_incl)[0])

            # Belt speed
            X_speed    = pd.DataFrame([features])[SPEED_FEATURES]
            speed_pred = models["belt_speed"].predict(X_speed)[0]
            rail2_pred = float(speed_pred[0])
            rail3_pred = float(speed_pred[1])

            # Damping
            X_damp = pd.DataFrame([features])[DAMP_FEATURES]
            damp_detected = int(models["damping"].predict(X_damp)[0])

            rows.append({
                "Case":          case_n,
                "Freq detected": "Yes" if freq_detected else "No",
                "Freq Hz":       f"{freq_hz:.1f}" if freq_detected else "—",
                "Incl detected": "Yes" if incl_detected else "No",
                "Incl degree":   f"{incl_degree:+.2f}" if incl_detected else "—",
                "Damping":       "Yes" if damp_detected else "No",
                "Rail2 (%)":     f"{rail2_pred:.0f}",
                "Rail3 (%)":     f"{rail3_pred:.0f}",
            })

        except Exception as e:
            print(f"  Case {case_n}: ERROR — {e}")
            rows.append({
                "Case":          case_n,
                "Freq detected": "ERROR",
                "Freq Hz":       "—",
                "Incl detected": "ERROR",
                "Incl degree":   "—",
                "Damping":       "ERROR",
                "Rail2 (%)":     "—",
                "Rail3 (%)":     "—",
            })

    # Print table
    print("\n=== PROFESSOR TEST CASES — MODEL PREDICTIONS ===\n")
    col_widths = {
        "Case":          4,
        "Freq detected": 13,
        "Freq Hz":       8,
        "Incl detected": 13,
        "Incl degree":   11,
        "Damping":       8,
        "Rail2 (%)":     7,
        "Rail3 (%)":     7,
    }
    headers = list(col_widths.keys())
    header_line = " | ".join(h.center(col_widths[h]) for h in headers)
    sep_line    = "-+-".join("-" * col_widths[h] for h in headers)
    print(header_line)
    print(sep_line)
    for row in rows:
        line = " | ".join(str(row[h]).center(col_widths[h]) for h in headers)
        print(line)

    print(
        "\nNote: Inclination detection is limited to Loc4 (end of Conveyor 1).\n"
        " Cases where inclination is at Loc2 or Loc5 will not be detected."
    )


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Loading best models from MLflow…")
    models = load_best_models()
    print("Models loaded.\n")

    task1_holdout_evaluation(models)
    task2_professor_predictions(models)


if __name__ == "__main__":
    main()
