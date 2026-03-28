"""
build_features.py — Extract features from all 270 XLS files.

Outputs:
  data/processed/features_p5.csv    — P5 frequency / inclination / belt-speed features
  data/processed/features_fixed.csv — fixed-phone (P1–P4) damping features
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import pandas as pd

from src.data_loader import load_file
from src.preprocessor import resample, detect_pickup_event, segment_journey
from src.features.frequency import extract_dominant_frequency
from src.features.inclination import extract_inclination_features
from src.features.belt_speed import extract_belt_speed_features
from src.features.damping import extract_damping_features

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
LABELS_PATH = ROOT / "data" / "labels.csv"


# ---------------------------------------------------------------------------
# Helper — load and resample one file
# ---------------------------------------------------------------------------
def load_and_resample(path: Path) -> pd.DataFrame:
    """Load *path* and resample to 100 Hz. Raises if path does not exist."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df, fs = load_file(str(path))
    df = resample(df, fs_original=fs, fs_target=100.0)
    return df


# ---------------------------------------------------------------------------
# Step 2 — Load G0 reference data
# ---------------------------------------------------------------------------
def load_g0_references() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return (g0_p5_journey_df, g0_phones_dict).

    g0_p5_journey_df  — journey segment of G0 P5 (used by belt-speed features)
    g0_phones_dict    — {'P1': df, 'P2': df, 'P3': df, 'P4': df} full resampled DFs
    """
    # --- G0 P5 ---
    g0_p5_path = DATA_RAW / "G0" / "G0_P5_case_perfect.xls"
    df_g0_p5 = load_and_resample(g0_p5_path)
    pickup_g0 = detect_pickup_event(df_g0_p5)
    segments_g0 = segment_journey(df_g0_p5, pickup_g0)
    g0_p5_journey = segments_g0["journey"]

    # --- G0 fixed phones P1-P4 ---
    g0_phones: dict[str, pd.DataFrame] = {}
    for p in [1, 2, 3, 4]:
        path = DATA_RAW / "G0" / f"G0_P{p}_case_perfect.xls"
        g0_phones[f"P{p}"] = load_and_resample(path)

    return g0_p5_journey, g0_phones


# ---------------------------------------------------------------------------
# Step 3 — Extract P5 features for one file
# ---------------------------------------------------------------------------
def extract_p5_row(
    group: int,
    case: int,
    label_row: pd.Series,
    g0_p5_journey: pd.DataFrame,
) -> dict:
    """Load and extract all P5 features for a single (group, case).

    Returns a flat dict with label columns + feature values.
    Raises on any processing error.
    """
    filepath = DATA_RAW / f"G{group}" / f"G{group}_P5_case{group}{case}.xls"
    df = load_and_resample(filepath)

    pickup_time = detect_pickup_event(df)
    segments = segment_journey(df, pickup_time)

    freq_feats = extract_dominant_frequency(segments["journey"])
    incl_feats = extract_inclination_features(segments["inclination"])
    speed_feats = extract_belt_speed_features(segments["journey"], g0_p5_journey)

    row: dict = {
        # --- label columns ---
        "group": group,
        "case": case,
        "freq_hz": label_row["freq_hz"],
        "freq_location": label_row["freq_location"],
        "incl_degree": label_row["incl_degree"],
        "incl_location": label_row["incl_location"],
        "incl_detectable": label_row["incl_detectable"],
        "damping_present": label_row["damping_present"],
        "damping_location": label_row["damping_location"],
        "rail2_speed": label_row["rail2_speed"],
        "rail3_speed": label_row["rail3_speed"],
        "freq_defect": label_row["freq_defect"],
        "speed_defect": label_row["speed_defect"],
        # --- features ---
        **freq_feats,
        **incl_feats,
        **speed_feats,
    }
    return row


# ---------------------------------------------------------------------------
# Step 4 — Extract fixed-phone damping features for one (group, case)
# ---------------------------------------------------------------------------
def extract_fixed_row(
    group: int,
    case: int,
    label_row: pd.Series,
    g0_phones: dict[str, pd.DataFrame],
) -> dict:
    """Load P1-P4 for (group, case) and extract damping features.

    Returns a flat dict with label columns + feature values.
    Raises on any processing error.
    """
    phones: dict[str, pd.DataFrame] = {}
    for p in [1, 2, 3, 4]:
        path = DATA_RAW / f"G{group}" / f"G{group}_P{p}_case{group}{case}.xls"
        phones[f"P{p}"] = load_and_resample(path)

    damp_feats = extract_damping_features(phones, g0_phones)

    row: dict = {
        # --- label columns ---
        "group": group,
        "case": case,
        "damping_present": label_row["damping_present"],
        "damping_location": label_row["damping_location"],
        "freq_defect": label_row["freq_defect"],
        # --- features ---
        **damp_feats,
    }
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 1 — Load labels
    # ------------------------------------------------------------------
    df_labels = pd.read_csv(LABELS_PATH)

    # ------------------------------------------------------------------
    # Step 2 — Load G0 reference data once
    # ------------------------------------------------------------------
    log.info("Loading G0 reference data …")
    g0_p5_journey, g0_phones = load_g0_references()
    log.info("G0 reference data loaded.")

    # ------------------------------------------------------------------
    # Step 3 — Extract P5 features
    # ------------------------------------------------------------------
    log.info("Extracting P5 features …")
    p5_rows: list[dict] = []
    df_p5_labels = df_labels[df_labels["phone"] == 5].copy()

    for _, label_row in df_p5_labels.iterrows():
        g = int(label_row["group"])
        c = int(label_row["case"])
        print(f"Processing G{g} P5 case{c}…")
        try:
            row = extract_p5_row(g, c, label_row, g0_p5_journey)
            p5_rows.append(row)
        except Exception as e:
            warnings.warn(f"Skipping G{g} P5 case{c}: {e}")

    df_p5 = pd.DataFrame(p5_rows)
    out_p5 = DATA_PROCESSED / "features_p5.csv"
    df_p5.to_csv(out_p5, index=False)
    log.info(f"Saved {len(df_p5)} rows → {out_p5}")

    # ------------------------------------------------------------------
    # Step 4 — Extract fixed-phone (damping) features
    # ------------------------------------------------------------------
    log.info("Extracting fixed-phone damping features …")
    fixed_rows: list[dict] = []

    # Use phone==1 rows to iterate unique (group, case) combinations
    df_p1_labels = df_labels[df_labels["phone"] == 1].copy()

    for _, label_row in df_p1_labels.iterrows():
        g = int(label_row["group"])
        c = int(label_row["case"])
        print(f"Processing G{g} fixed phones case{c}…")
        try:
            row = extract_fixed_row(g, c, label_row, g0_phones)
            fixed_rows.append(row)
        except Exception as e:
            warnings.warn(f"Skipping G{g} fixed phones case{c}: {e}")

    df_fixed = pd.DataFrame(fixed_rows)
    out_fixed = DATA_PROCESSED / "features_fixed.csv"
    df_fixed.to_csv(out_fixed, index=False)
    log.info(f"Saved {len(df_fixed)} rows → {out_fixed}")

    # ------------------------------------------------------------------
    # Step 5 — Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nfeatures_p5.csv  — {len(df_p5)} rows")
    print(f"Columns ({len(df_p5.columns)}): {list(df_p5.columns)}")

    if "case" in df_p5.columns:
        print("\nRows per case type (features_p5):")
        print(df_p5["case"].value_counts().sort_index().to_string())

    print(f"\nfeatures_fixed.csv — {len(df_fixed)} rows")
    print(f"Columns ({len(df_fixed.columns)}): {list(df_fixed.columns)}")

    if "damping_present" in df_fixed.columns:
        counts = df_fixed["damping_present"].value_counts().sort_index()
        print("\ndamping_present counts (features_fixed):")
        for val, cnt in counts.items():
            label = "present" if val == 1 else "absent"
            print(f"  {val} ({label}): {cnt}")

    print("=" * 60)


if __name__ == "__main__":
    main()
