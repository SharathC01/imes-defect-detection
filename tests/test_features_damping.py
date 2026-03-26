"""Tests for src/features/damping.py — extract_damping_features."""

import numpy as np
import pandas as pd

from src.features.damping import extract_damping_features

_EXPECTED_KEYS = {
    "rms_P1", "rms_P2", "rms_P3", "rms_P4",
    "rms_ratio_P1", "rms_ratio_P2", "rms_ratio_P3", "rms_ratio_P4",
    "min_rms_ratio", "min_rms_ratio_phone",
}

_PHONE_NAMES = ["P1", "P2", "P3", "P4"]


def make_phone_df(duration=50.0, fs=100.0, value=1.0):
    n = int(duration * fs)
    t = np.arange(n) / fs
    acc = np.full(n, value)
    return pd.DataFrame({
        "time": t, "acc_x": acc * 0.1, "acc_y": acc * 0.1,
        "acc_z": acc * 0.9, "acc_abs": acc
    })


def _uniform_phones(value: float) -> dict[str, pd.DataFrame]:
    """Return a phones dict where all four phones have a constant acc_abs value."""
    return {p: make_phone_df(value=value) for p in _PHONE_NAMES}


# ---------------------------------------------------------------------------
# Test 1 — returned dict has exactly the expected keys
# ---------------------------------------------------------------------------

def test_extract_damping_features_returns_all_keys():
    phones = _uniform_phones(1.0)
    g0_phones = _uniform_phones(1.0)

    result = extract_damping_features(phones, g0_phones)

    assert set(result.keys()) == _EXPECTED_KEYS


# ---------------------------------------------------------------------------
# Test 2 — rms_ratio == 1.0 when case and G0 signals are identical
# ---------------------------------------------------------------------------

def test_rms_ratio_equals_one_when_case_matches_g0():
    phones = _uniform_phones(1.0)
    g0_phones = _uniform_phones(1.0)

    result = extract_damping_features(phones, g0_phones)

    for phone in _PHONE_NAMES:
        assert abs(result[f"rms_ratio_{phone}"] - 1.0) < 1e-6, (
            f"rms_ratio_{phone} = {result[f'rms_ratio_{phone}']} expected ~1.0"
        )


# ---------------------------------------------------------------------------
# Test 3 — rms_ratio ≈ 0.5 when case amplitude is half G0
# ---------------------------------------------------------------------------

def test_rms_ratio_less_than_one_when_damped():
    phones = _uniform_phones(0.5)
    g0_phones = _uniform_phones(1.0)

    result = extract_damping_features(phones, g0_phones)

    for phone in _PHONE_NAMES:
        assert abs(result[f"rms_ratio_{phone}"] - 0.5) < 1e-3, (
            f"rms_ratio_{phone} = {result[f'rms_ratio_{phone}']} expected ~0.5"
        )


# ---------------------------------------------------------------------------
# Test 4 — min_rms_ratio_phone identifies the phone with lowest ratio
# ---------------------------------------------------------------------------

def test_min_rms_ratio_phone_identifies_lowest():
    phones = {
        "P1": make_phone_df(value=1.0),
        "P2": make_phone_df(value=1.0),
        "P3": make_phone_df(value=0.1),
        "P4": make_phone_df(value=1.0),
    }
    g0_phones = _uniform_phones(1.0)

    result = extract_damping_features(phones, g0_phones)

    assert result["min_rms_ratio_phone"] == "P3"
    assert abs(result["min_rms_ratio"] - 0.1) < 1e-3, (
        f"min_rms_ratio = {result['min_rms_ratio']} expected ~0.1"
    )


# ---------------------------------------------------------------------------
# Test 5 — handoff window spike (t=18–38 s) does not contaminate RMS
# ---------------------------------------------------------------------------

def test_handoff_window_excluded():
    # Build a DataFrame where the signal is 1.0 everywhere but spikes to 1000.0
    # strictly inside the handoff window (18 < t < 38).  The boundaries t=18.0
    # and t=38.0 are kept at 1.0 so they are safely included in the C1/C2 mask.
    duration = 50.0
    fs = 100.0
    n = int(duration * fs)
    t = np.arange(n) / fs
    acc = np.where((t > 18.0) & (t < 38.0), 1000.0, 1.0)

    df = pd.DataFrame({
        "time": t, "acc_x": acc * 0.1, "acc_y": acc * 0.1,
        "acc_z": acc * 0.9, "acc_abs": acc
    })

    phones = {p: df.copy() for p in _PHONE_NAMES}
    g0_phones = {p: df.copy() for p in _PHONE_NAMES}

    result = extract_damping_features(phones, g0_phones)

    for phone in _PHONE_NAMES:
        assert abs(result[f"rms_ratio_{phone}"] - 1.0) < 0.1, (
            f"rms_ratio_{phone} = {result[f'rms_ratio_{phone}']} "
            f"— handoff spike appears to be contaminating RMS"
        )
