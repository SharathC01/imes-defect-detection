"""Inclination feature extraction for IMES defect detection.

Extracts six physically motivated features from the inclination window
(t = 30–37 s in the P5 recording).  All features are derived from the
impact transient produced when P5 drops off or lands onto a tilted section
of the conveyor at the handoff zone (Location 4).

Note: the accelerometer records *linear* acceleration only (gravity removed).
Static tilt cannot be detected from mean axis values.  Inclination is inferred
from the character of the impact transient — its magnitude, sharpness, energy,
and direction (sign of acc_z at peak).
"""

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt


def extract_inclination_features(
    df_inclination: pd.DataFrame,
    fs: float = 100.0,
) -> dict:
    """Extract impact-transient features from the inclination window.

    Processes the 30–37 s segment of a P5 recording, which captures the
    handoff transient at Location 4.  Six features characterise the
    magnitude, direction, sharpness, and energy of the impact event.

    Location 5 inclination cases are not detectable in this window and are
    treated as label = 0 in the binary detection stage.

    Parameters
    ----------
    df_inclination : pd.DataFrame
        Inclination segment produced by ``preprocessor.segment_journey``
        (the ``'inclination'`` key).  Must contain columns ``time``,
        ``acc_x``, ``acc_y``, ``acc_z``, ``acc_abs``.
    fs : float, optional
        Sampling rate of *df_inclination* in Hz.  Default is 100.0.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'peak_z'`` : float
            Maximum absolute value of ``acc_z`` in the window.  Captures
            vertical impact magnitude regardless of impact direction.
        ``'signed_peak_z'`` : float
            Value of ``acc_z`` (not absolute) at the sample of peak
            ``acc_abs``.  Negative = downward impact (P5 drops off a raised
            section); positive = upward impact (P5 hits an upward ramp).
        ``'peak_jerk'`` : float
            Maximum jerk magnitude in the window, where jerk is defined as
            ``np.diff(acc_abs) * fs``.  High jerk indicates a sharp,
            sudden impact event.
        ``'impulse_energy'`` : float
            Energy of a 0.5 s window centred on the peak ``acc_abs`` sample,
            computed as ``np.trapezoid(acc_abs_window ** 2, dx=1/fs)``.
        ``'rise_time'`` : float
            Time in seconds from 10 % to 90 % of peak ``acc_abs`` within
            the 0.5 s impact window.  Measures how abruptly the impact
            builds.  Returns ``np.nan`` if no clear rise can be resolved.
        ``'post_impact_rms'`` : float
            RMS of ``acc_abs`` in the 0.5 s window immediately following
            the impact peak.  Captures residual vibration after the drop.

    Notes
    -----
    A 4th-order Butterworth low-pass filter (cutoff=20 Hz) is applied
    before extracting magnitude features to remove frequency-defect
    contamination (30-50 Hz) from combined defect cases.

    The 0.5 s impact window is centred on the sample of peak ``acc_abs``
    and spans ``± half_samples = int(0.25 * fs)`` samples.  All window
    boundaries are clamped to the valid index range of *df_inclination*.
    """
    acc_z = df_inclination["acc_z"].to_numpy(dtype=float)
    acc_abs = df_inclination["acc_abs"].to_numpy(dtype=float)
    n = len(acc_abs)
    half_samples = int(0.25 * fs)  # 0.5 s window → ±0.25 s each side

    # Low-pass filter: remove 30–50 Hz frequency-defect contamination
    b, a = butter(4, 20.0 / (fs / 2), btype="low")
    acc_abs_filtered = filtfilt(b, a, acc_abs)
    acc_z_filtered = filtfilt(b, a, acc_z)

    # ------------------------------------------------------------------
    # 1. peak_z — raw signal, direction-agnostic vertical magnitude
    # ------------------------------------------------------------------
    peak_z = float(np.max(np.abs(acc_z)))

    # ------------------------------------------------------------------
    # 2. signed_peak_z — raw acc_z value at peak index of UNFILTERED signal
    # ------------------------------------------------------------------
    peak_abs_idx = int(np.argmax(acc_abs))        # anchor: unfiltered peak
    signed_peak_z = float(acc_z[peak_abs_idx])

    # ------------------------------------------------------------------
    # 3. peak_jerk — filtered signal
    # ------------------------------------------------------------------
    jerk = np.diff(acc_abs_filtered) * fs
    peak_jerk = float(np.max(np.abs(jerk)))

    # ------------------------------------------------------------------
    # 4. impulse_energy — trapz integral of filtered acc_abs² over 0.5 s window
    # ------------------------------------------------------------------
    win_start = max(0, peak_abs_idx - half_samples)
    win_end = min(n, peak_abs_idx + half_samples)
    acc_abs_window = acc_abs_filtered[win_start:win_end]
    impulse_energy = float(np.trapezoid(acc_abs_window ** 2, dx=1.0 / fs))

    # ------------------------------------------------------------------
    # 5. rise_time — 10 %→90 % of peak filtered acc_abs within the impact window
    # ------------------------------------------------------------------
    peak_in_window = float(np.max(acc_abs_window))
    low_thresh = 0.10 * peak_in_window
    high_thresh = 0.90 * peak_in_window

    rise_time: float = np.nan
    low_idx = np.where(acc_abs_window >= low_thresh)[0]
    high_idx = np.where(acc_abs_window >= high_thresh)[0]
    if low_idx.size > 0 and high_idx.size > 0:
        t_low = float(low_idx[0]) / fs
        t_high = float(high_idx[0]) / fs
        if t_high >= t_low:
            rise_time = t_high - t_low

    # ------------------------------------------------------------------
    # 6. post_impact_rms — RMS of filtered acc_abs in 0.5 s after peak
    # ------------------------------------------------------------------
    post_start = peak_abs_idx
    post_end = min(n, peak_abs_idx + int(0.5 * fs))
    post_window = acc_abs_filtered[post_start:post_end]
    if post_window.size > 0:
        post_impact_rms = float(np.sqrt(np.mean(post_window ** 2)))
    else:
        post_impact_rms = np.nan

    return {
        "peak_z": peak_z,
        "signed_peak_z": signed_peak_z,
        "peak_jerk": peak_jerk,
        "impulse_energy": impulse_energy,
        "rise_time": rise_time,
        "post_impact_rms": post_impact_rms,
    }
