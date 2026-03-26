"""Tests for src/features/frequency.py — all signals are synthetic (numpy only)."""

import numpy as np
import pandas as pd
import pytest

from src.features.frequency import extract_dominant_frequency, locate_frequency_in_time

FS = 100.0
N = 3000  # 30 s at 100 Hz


def _make_journey_df(acc_x, acc_y, acc_z, fs=FS):
    """Build a minimal journey DataFrame from three numpy arrays."""
    t = np.arange(len(acc_x)) / fs
    return pd.DataFrame(
        {
            "time": t,
            "acc_x": acc_x,
            "acc_y": acc_y,
            "acc_z": acc_z,
            "acc_abs": np.sqrt(acc_x**2 + acc_y**2 + acc_z**2),
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — dominant frequency detection
# ---------------------------------------------------------------------------


def test_extract_dominant_frequency_detects_correct_frequency():
    """Strong 40 Hz on acc_z; 10 Hz (below search band) on acc_x and acc_y."""
    t = np.arange(N) / FS

    # acc_z: large-amplitude 40 Hz — should dominate in the 25–50 Hz band
    acc_z = 2.0 * np.sin(2 * np.pi * 40.0 * t)
    # acc_x, acc_y: 10 Hz only — entirely below the 25 Hz search cutoff
    acc_x = 0.5 * np.sin(2 * np.pi * 10.0 * t)
    acc_y = 0.5 * np.sin(2 * np.pi * 10.0 * t)

    result = extract_dominant_frequency(_make_journey_df(acc_x, acc_y, acc_z), fs=FS)

    assert abs(result["dominant_freq_hz"] - 40.0) <= 1.0
    assert result["dominant_axis"] == "acc_z"


# ---------------------------------------------------------------------------
# Test 2 — return dict has exactly the expected keys
# ---------------------------------------------------------------------------


def test_extract_dominant_frequency_returns_all_keys():
    """Return dict must contain exactly the nine documented keys."""
    t = np.arange(N) / FS
    acc_z = 1.0 * np.sin(2 * np.pi * 35.0 * t)
    acc_x = np.zeros(N)
    acc_y = np.zeros(N)

    result = extract_dominant_frequency(_make_journey_df(acc_x, acc_y, acc_z), fs=FS)

    expected_keys = {
        "dominant_freq_hz",
        "dominant_axis",
        "peak_magnitude",
        "freq_x",
        "freq_y",
        "freq_z",
        "mag_x",
        "mag_y",
        "mag_z",
    }
    assert set(result.keys()) == expected_keys


# ---------------------------------------------------------------------------
# Test 3 — search band respects 25 Hz lower cutoff
# ---------------------------------------------------------------------------


def test_extract_dominant_frequency_respects_search_band():
    """Strong 10 Hz component (below cutoff) must be ignored; 35 Hz should win."""
    t = np.arange(N) / FS

    # acc_z: dominant energy at 10 Hz but also a weaker 35 Hz component
    acc_z = 5.0 * np.sin(2 * np.pi * 10.0 * t) + 1.0 * np.sin(2 * np.pi * 35.0 * t)
    acc_x = np.zeros(N)
    acc_y = np.zeros(N)

    result = extract_dominant_frequency(_make_journey_df(acc_x, acc_y, acc_z), fs=FS)

    # 10 Hz is below the 25 Hz lower bound — the function must return ~35 Hz
    assert abs(result["dominant_freq_hz"] - 35.0) <= 1.0


# ---------------------------------------------------------------------------
# Test 4 — STFT window centred on the burst
# ---------------------------------------------------------------------------


def test_locate_frequency_in_time_returns_valid_window():
    """40 Hz burst only in the middle third (t=10–20 s) of a 30 s signal."""
    t = np.arange(N) / FS

    # Build acc_z: silence outside t=10–20 s, 40 Hz sine inside
    acc_z = np.zeros(N)
    burst_start, burst_end = 1000, 2000  # sample indices for t=10s to t=20s
    acc_z[burst_start:burst_end] = 2.0 * np.sin(2 * np.pi * 40.0 * t[burst_start:burst_end])

    acc_x = np.zeros(N)
    acc_y = np.zeros(N)

    df = _make_journey_df(acc_x, acc_y, acc_z)
    result = locate_frequency_in_time(df, dominant_freq=40.0, fs=FS)

    # Centre must land somewhere inside (or very close to) the burst window
    assert 8.0 <= result["defect_time_centre"] <= 22.0
    # Window must be ordered correctly
    assert result["defect_time_start"] < result["defect_time_centre"] < result["defect_time_end"]
