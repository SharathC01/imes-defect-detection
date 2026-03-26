"""Tests for src/features/inclination.py — extract_inclination_features.

All signals are synthetic, built with numpy.
fs=100.0 Hz, t=30s to t=37s (700 samples).
No real data files are referenced anywhere in this module.
"""

import numpy as np
import pandas as pd
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from features.inclination import extract_inclination_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FS = 100.0
N = 700  # 7 s × 100 Hz = 700 samples (t = 30.00 s … 36.99 s)
IMPACT_IDX = 300  # t = 33.0 s  (300 samples after t=30.0 s)

EXPECTED_KEYS = {
    "peak_z",
    "signed_peak_z",
    "peak_jerk",
    "impulse_energy",
    "rise_time",
    "post_impact_rms",
}


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(acc_z: np.ndarray, acc_abs: np.ndarray | None = None) -> pd.DataFrame:
    """Return a DataFrame with the five columns expected by
    extract_inclination_features.

    Parameters
    ----------
    acc_z : np.ndarray, shape (N,)
        Z-axis acceleration signal.
    acc_abs : np.ndarray, shape (N,) or None
        If None, ``np.abs(acc_z)`` is used.
    """
    n = len(acc_z)
    t = np.arange(n) / FS + 30.0
    if acc_abs is None:
        acc_abs = np.abs(acc_z)
    return pd.DataFrame(
        {
            "time": t,
            "acc_x": np.zeros(n, dtype=float),
            "acc_y": np.zeros(n, dtype=float),
            "acc_z": acc_z.astype(float),
            "acc_abs": acc_abs.astype(float),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — returned dict contains exactly the six expected keys
# ---------------------------------------------------------------------------

def test_extract_inclination_features_returns_all_keys():
    """Returned dict must contain exactly the six documented feature keys."""
    acc_z = np.zeros(N)
    acc_z[IMPACT_IDX] = 10.0
    df = _make_df(acc_z)

    result = extract_inclination_features(df, fs=FS)

    assert isinstance(result, dict)
    assert set(result.keys()) == EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Test 2 — signed_peak_z < 0 for a downward impact
# ---------------------------------------------------------------------------

def test_signed_peak_z_is_negative_for_downward_impact():
    """A sharp negative acc_z spike at t=33 s (downward drop) must yield
    signed_peak_z < 0.

    signed_peak_z is acc_z[argmax(acc_abs)], so acc_abs must peak at the
    same index as the negative acc_z spike.
    """
    acc_z = np.zeros(N)
    acc_z[IMPACT_IDX] = -20.0  # downward spike

    # acc_abs peaks at the same index so argmax(acc_abs) == IMPACT_IDX
    acc_abs = np.zeros(N)
    acc_abs[IMPACT_IDX] = 20.0

    df = _make_df(acc_z, acc_abs)
    result = extract_inclination_features(df, fs=FS)

    assert result["signed_peak_z"] < 0


# ---------------------------------------------------------------------------
# Test 3 — signed_peak_z > 0 for an upward impact
# ---------------------------------------------------------------------------

def test_signed_peak_z_is_positive_for_upward_impact():
    """A sharp positive acc_z spike at t=33 s (upward ramp) must yield
    signed_peak_z > 0.
    """
    acc_z = np.zeros(N)
    acc_z[IMPACT_IDX] = 20.0  # upward spike

    acc_abs = np.zeros(N)
    acc_abs[IMPACT_IDX] = 20.0

    df = _make_df(acc_z, acc_abs)
    result = extract_inclination_features(df, fs=FS)

    assert result["signed_peak_z"] > 0


# ---------------------------------------------------------------------------
# Test 4 — peak_jerk is higher for a sharp impact than a gradual one
# ---------------------------------------------------------------------------

def test_peak_jerk_is_higher_for_sharp_impact():
    """peak_jerk must be strictly larger for a fast-rising transient than for
    a slow-rising one of equal amplitude.

    peak_jerk = max(|diff(acc_abs_filtered)| * fs).
    Both signals peak at 20 m/s².  The sharp signal rises in 3 samples
    (~30 ms); the gradual signal rises over 100 samples (~1 s).
    The 20 Hz low-pass filter preserves the relative ordering even after
    smoothing.
    """
    # --- Sharp impact: step from 0 → 20 m/s² over 3 samples at t=33 s ---
    acc_abs_sharp = np.zeros(N)
    acc_abs_sharp[IMPACT_IDX - 1] = 0.0
    acc_abs_sharp[IMPACT_IDX]     = 10.0
    acc_abs_sharp[IMPACT_IDX + 1] = 20.0
    acc_abs_sharp[IMPACT_IDX + 2 : IMPACT_IDX + 30] = 20.0  # hold plateau

    acc_z_sharp = np.zeros(N)  # direction irrelevant for peak_jerk
    df_sharp = _make_df(acc_z_sharp, acc_abs_sharp)

    # --- Gradual impact: linear ramp from 0 → 20 m/s² over 100 samples ---
    acc_abs_gradual = np.zeros(N)
    ramp_start = IMPACT_IDX - 50
    ramp_end   = IMPACT_IDX + 50
    acc_abs_gradual[ramp_start:ramp_end] = np.linspace(0.0, 20.0, 100)

    acc_z_gradual = np.zeros(N)
    df_gradual = _make_df(acc_z_gradual, acc_abs_gradual)

    result_sharp   = extract_inclination_features(df_sharp,   fs=FS)
    result_gradual = extract_inclination_features(df_gradual, fs=FS)

    assert result_sharp["peak_jerk"] > result_gradual["peak_jerk"]


# ---------------------------------------------------------------------------
# Test 5 — rise_time is np.nan for a flat, low-amplitude signal
# ---------------------------------------------------------------------------

def test_rise_time_returns_nan_for_flat_signal():
    """A perfectly flat signal with no impact transient must return
    rise_time = np.nan, because no 10 %→90 % rise can be resolved.
    """
    # Constant low-amplitude signal — no impact, no transient
    acc_z   = np.full(N, 0.05)   # 0.05 m/s²  (far below any real impact)
    acc_abs = np.full(N, 0.05)

    df = _make_df(acc_z, acc_abs)
    result = extract_inclination_features(df, fs=FS)

    assert np.isnan(result["rise_time"])
