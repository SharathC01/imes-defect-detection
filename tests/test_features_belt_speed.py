"""Tests for src/features/belt_speed.py — extract_belt_speed_features."""

import numpy as np
import pandas as pd
import pytest

from src.features.belt_speed import extract_belt_speed_features

FS = 100.0
_EXPECTED_KEYS = {
    "journey_duration",
    "duration_delta",
    "rms_full",
    "rms_first_half",
    "rms_second_half",
    "rms_ratio",
}


def _make_journey(t_start: float, t_end: float, acc_abs_value: float = 1.0) -> pd.DataFrame:
    """Return a uniform-amplitude journey DataFrame from t_start to t_end at fs=100."""
    n = round((t_end - t_start) * FS) + 1
    time = np.linspace(t_start, t_end, n)
    acc_abs = np.full(n, acc_abs_value)
    return pd.DataFrame(
        {
            "time": time,
            "acc_x": acc_abs,
            "acc_y": acc_abs,
            "acc_z": acc_abs,
            "acc_abs": acc_abs,
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — returned dict has exactly the expected keys
# ---------------------------------------------------------------------------

def test_extract_belt_speed_features_returns_all_keys():
    df_journey = _make_journey(4.0, 44.0)
    df_g0 = _make_journey(4.0, 43.0)

    result = extract_belt_speed_features(df_journey, df_g0, fs=FS)

    assert set(result.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Test 2 — journey_duration and duration_delta computed correctly
# ---------------------------------------------------------------------------

def test_journey_duration_is_correct():
    # journey spans exactly 40 s; G0 spans exactly 39 s
    df_journey = _make_journey(4.0, 44.0)
    df_g0 = _make_journey(4.0, 43.0)

    result = extract_belt_speed_features(df_journey, df_g0, fs=FS)

    assert abs(result["journey_duration"] - 40.0) < 0.1
    assert abs(result["duration_delta"] - 1.0) < 0.1


# ---------------------------------------------------------------------------
# Test 3 — rms_ratio < 1.0 when first half is less energetic than second half
# ---------------------------------------------------------------------------

def test_rms_ratio_reflects_half_energies():
    # Build a 40-second journey: first half amplitude=1.0, second half amplitude=2.0.
    # The split is at t_mid = t[0] + duration/2 = 20.0.
    n = 4001
    time = np.linspace(0.0, 40.0, n)
    t_mid = 20.0
    acc_abs = np.where(time < t_mid, 1.0, 2.0)
    df_journey = pd.DataFrame(
        {"time": time, "acc_x": acc_abs, "acc_y": acc_abs, "acc_z": acc_abs, "acc_abs": acc_abs}
    )
    df_g0 = _make_journey(0.0, 39.0)

    result = extract_belt_speed_features(df_journey, df_g0, fs=FS)

    assert result["rms_ratio"] < 1.0


# ---------------------------------------------------------------------------
# Test 4 — rms_ratio is np.nan when the second half is all zeros
# ---------------------------------------------------------------------------

def test_rms_ratio_returns_nan_for_zero_second_half():
    # Build a 40-second journey: first half amplitude=1.0, second half=0.0.
    n = 4001
    time = np.linspace(0.0, 40.0, n)
    t_mid = 20.0
    acc_abs = np.where(time < t_mid, 1.0, 0.0)
    df_journey = pd.DataFrame(
        {"time": time, "acc_x": acc_abs, "acc_y": acc_abs, "acc_z": acc_abs, "acc_abs": acc_abs}
    )
    df_g0 = _make_journey(0.0, 39.0)

    result = extract_belt_speed_features(df_journey, df_g0, fs=FS)

    assert np.isnan(result["rms_ratio"])
