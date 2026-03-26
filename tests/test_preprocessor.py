"""
Unit tests for src/preprocessor.py.

All tests use synthetic DataFrames built with numpy — no real files are read.
Column names match the output of data_loader.load_file():
    time, acc_x, acc_y, acc_z, acc_abs
"""

import numpy as np
import pandas as pd
import pytest

from src.preprocessor import detect_pickup_event, resample, segment_journey


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(duration: float, fs: float, acc_amplitude: float = 1.0) -> pd.DataFrame:
    """Return a synthetic accelerometer DataFrame at uniform sampling rate *fs*."""
    t = np.arange(0.0, duration, 1.0 / fs)
    n = len(t)
    rng = np.random.default_rng(0)
    noise = rng.uniform(-acc_amplitude, acc_amplitude, n)
    return pd.DataFrame(
        {
            "time": t,
            "acc_x": noise * 0.3,
            "acc_y": noise * 0.5,
            "acc_z": noise * 0.7,
            "acc_abs": np.abs(noise),
        }
    )


# ---------------------------------------------------------------------------
# 1. resample — skip when fs_original == fs_target
# ---------------------------------------------------------------------------

def test_resample_skips_when_rates_equal():
    df = _make_df(duration=10.0, fs=100.0)
    result = resample(df, fs_original=100.0, fs_target=100.0)
    # Must be the identical object — no copy, no resampling
    assert result is df


# ---------------------------------------------------------------------------
# 2. resample — row count correct when downsampling from 400 Hz to 100 Hz
# ---------------------------------------------------------------------------

def test_resample_reduces_row_count_400_to_100():
    duration = 10.0
    fs_orig = 400.0
    fs_target = 100.0
    df = _make_df(duration=duration, fs=fs_orig)

    result = resample(df, fs_original=fs_orig, fs_target=fs_target)

    # Output rows ≈ duration * fs_target (np.arange may drop last step)
    expected_rows = int(duration * fs_target)
    assert abs(len(result) - expected_rows) <= 1

    # Time step must be uniform at 1/fs_target
    dt = np.diff(result["time"].to_numpy())
    assert np.allclose(dt, 1.0 / fs_target, atol=1e-9)

    # All acc columns must be present
    for col in ("acc_x", "acc_y", "acc_z", "acc_abs"):
        assert col in result.columns


# ---------------------------------------------------------------------------
# 3. detect_pickup_event — returns correct timestamp for known peak
# ---------------------------------------------------------------------------

def test_detect_pickup_event_returns_correct_timestamp():
    duration = 46.0
    fs = 100.0
    df = _make_df(duration=duration, fs=fs, acc_amplitude=0.5)

    # Plant a large spike (35 m/s²) at t = 44.0 s — well inside the last 30 %
    spike_time = 44.0
    spike_idx = int(spike_time * fs)
    df.loc[spike_idx, "acc_abs"] = 35.0

    result = detect_pickup_event(df, fs=fs)

    assert abs(result - spike_time) < 0.05  # within one sample at 100 Hz


# ---------------------------------------------------------------------------
# 4. detect_pickup_event — raises ValueError when no peak exceeds 10.0 m/s²
# ---------------------------------------------------------------------------

def test_detect_pickup_event_raises_when_no_clear_peak():
    # acc_abs is always ≤ 0.5 m/s² — well below the 10.0 m/s² threshold
    df = _make_df(duration=46.0, fs=100.0, acc_amplitude=0.5)

    with pytest.raises(ValueError, match="acc_abs"):
        detect_pickup_event(df, fs=100.0)


# ---------------------------------------------------------------------------
# 5. segment_journey — correct time bounds for all three segments
# ---------------------------------------------------------------------------

def test_segment_journey_correct_time_bounds():
    duration = 50.0
    fs = 100.0
    df = _make_df(duration=duration, fs=fs)
    pickup_time = 45.0

    segments = segment_journey(df, pickup_time=pickup_time)

    # Keys present
    assert set(segments.keys()) == {"full", "journey", "handoff"}

    # full — entire recording
    assert len(segments["full"]) == len(df)

    # journey — t in [4.0, pickup_time - 2.0]
    t_journey = segments["journey"]["time"]
    assert t_journey.iloc[0] >= 4.0
    assert t_journey.iloc[-1] <= pickup_time - 2.0

    # handoff — t in [18.0, 38.0]
    t_handoff = segments["handoff"]["time"]
    assert t_handoff.iloc[0] >= 18.0
    assert t_handoff.iloc[-1] <= 38.0

    # Segments must be non-empty
    assert len(segments["journey"]) > 0
    assert len(segments["handoff"]) > 0
