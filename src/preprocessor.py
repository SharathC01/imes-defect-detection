"""
Preprocessing utilities for IMES accelerometer data.

Handles resampling (fixed phones vary from ~71–400 Hz), arm-pickup landmark
detection on P5 signals, and journey segmentation into analysis windows.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.signal import find_peaks

# Accelerometer column names as they appear in the loaded XLS files
_ACC_COLS = ["acc_x", "acc_y", "acc_z", "acc_abs"]


def resample(
    df: pd.DataFrame,
    fs_original: float,
    fs_target: float = 100.0,
) -> pd.DataFrame:
    """Resample accelerometer columns to a uniform target sampling rate.

    Uses cubic interpolation (``scipy.interpolate.interp1d`` with
    ``kind='cubic'``) so that the resampled signal is smooth and artefact-free.
    Resampling is skipped when the original rate is already within ±1 Hz of the
    target to avoid unnecessary processing on P5 files (~100 Hz).

    Parameters
    ----------
    df : pd.DataFrame
        Raw accelerometer DataFrame with columns ``time``, ``acc_x``,
        ``acc_y``, ``acc_z``, ``acc_abs`` as produced by ``data_loader.load``.
    fs_original : float
        Nominal sampling rate of the input recording in Hz.
    fs_target : float, optional
        Desired output sampling rate in Hz.  Default is 100.0.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same column layout as *df* but with uniformly spaced
        time steps at *fs_target*.  When resampling is skipped the original
        DataFrame is returned unchanged.

    Notes
    -----
    Extrapolation beyond the original time range is disabled
    (``bounds_error=True``).  The output time vector is constructed with
    ``np.arange`` so that the step size is exactly ``1 / fs_target`` seconds.
    """
    if abs(fs_original - fs_target) <= 1.0:
        return df

    t_orig = df["time"].to_numpy(dtype=float)
    t_new = np.arange(t_orig[0], t_orig[-1], 1.0 / fs_target)

    resampled: dict[str, np.ndarray] = {"time": t_new}
    for col in _ACC_COLS:
        if col not in df.columns:
            continue
        interpolator = interp1d(
            t_orig,
            df[col].to_numpy(dtype=float),
            kind="cubic",
            bounds_error=True,
        )
        resampled[col] = interpolator(t_new)

    # Preserve any extra columns that are not accelerometer channels
    extra_cols = [c for c in df.columns if c not in ("time", *_ACC_COLS)]
    result = pd.DataFrame(resampled)
    for col in extra_cols:
        interpolator = interp1d(
            t_orig,
            df[col].to_numpy(dtype=float),
            kind="cubic",
            bounds_error=True,
        )
        result[col] = interpolator(t_new)

    return result


def detect_pickup_event(df: pd.DataFrame, fs: float = 100.0) -> float:
    """Detect the timestamp of the Arm 2 pickup event in a P5 recording.

    The pickup event is the moment Arm 2 lifts P5 off Conveyor 2.  It produces
    the largest acceleration peak in the recording (~37–39 m/s²) and always
    occurs in the final portion of the signal.  Only the last 30 % of the
    recording is searched to avoid false positives from earlier transients
    (e.g. the handoff zone at ~22–34 s).

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed P5 DataFrame containing ``time`` and ``acc_abs``
        columns sampled at *fs* Hz.
    fs : float, optional
        Sampling rate of *df* in Hz.  Used to set the minimum peak separation
        to 0.5 s.  Default is 100.0.

    Returns
    -------
    float
        Timestamp (seconds, relative to recording start) of the arm pickup
        peak.

    Raises
    ------
    ValueError
        If no peak with ``acc_abs`` ≥ 10.0 m/s² is found in the search window,
        which indicates either the wrong phone file or a corrupted recording.

    Notes
    -----
    Peak detection uses ``scipy.signal.find_peaks`` with a minimum height of
    10.0 m/s² and a minimum separation of 0.5 s (``distance = 0.5 * fs``
    samples).  Among all qualifying peaks the one with the highest amplitude
    is selected as the pickup event.
    """
    t = df["time"].to_numpy(dtype=float)
    acc_abs = df["acc_abs"].to_numpy(dtype=float)

    # Restrict search to the last 30 % of the recording
    search_start_idx = int(len(t) * 0.70)
    t_window = t[search_start_idx:]
    acc_window = acc_abs[search_start_idx:]

    min_distance = max(1, int(0.5 * fs))
    peak_indices, _ = find_peaks(
        acc_window,
        height=10.0,
        distance=min_distance,
    )

    if len(peak_indices) == 0:
        raise ValueError(
            "No clear arm pickup peak found (acc_abs < 10.0 m/s²) in the last "
            "30 %% of the recording.  Check that the correct P5 file was passed."
        )

    # Pick the highest peak in the search window
    best_local_idx = peak_indices[np.argmax(acc_window[peak_indices])]
    pickup_time = float(t_window[best_local_idx])
    return pickup_time


def segment_journey(df: pd.DataFrame, pickup_time: float) -> dict:
    """Slice a P5 recording into the three canonical analysis windows.

    All time boundaries are relative to the recording start (``t = 0``).

    Parameters
    ----------
    df : pd.DataFrame
        Full P5 DataFrame with a ``time`` column.  The DataFrame should
        already be resampled to a uniform rate before calling this function.
    pickup_time : float
        Timestamp (seconds) of the arm pickup event as returned by
        :func:`detect_pickup_event`.

    Returns
    -------
    dict
        A dictionary with three entries:

        ``'full'`` : pd.DataFrame
            The entire recording, unchanged.
        ``'journey'`` : pd.DataFrame
            Conveyor journey from ``t = 4 s`` to ``pickup_time − 2 s``.
            Covers P5's travel on Conveyor 1 and Conveyor 2, excluding the
            initial arm-deposit transient and the final pickup impulse.
        ``'handoff'`` : pd.DataFrame
            Fixed handoff analysis window from ``t = 18 s`` to ``t = 38 s``.
            Captures both handoff transients (C1→transfer at ~22–25 s and
            transfer→C2 at ~33–34 s) with a small guard margin on each side.

    Notes
    -----
    Segment boundaries are applied with boolean indexing on ``time``, so
    the returned DataFrames share the original index values of *df*.  Callers
    that need a reset index should call ``.reset_index(drop=True)`` on the
    returned segments.
    """
    t = df["time"]

    journey_start = 4.0
    journey_end = pickup_time - 2.0
    handoff_start = 18.0
    handoff_end = 38.0

    segments = {
        "full": df,
        "journey": df.loc[(t >= journey_start) & (t <= journey_end)],
        "handoff": df.loc[(t >= handoff_start) & (t <= handoff_end)],
        "inclination": df.loc[(t >= 30.0) & (t <= 37.0)],
    }
    return segments
