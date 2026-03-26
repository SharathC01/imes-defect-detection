"""Belt-speed feature extraction for IMES defect detection.

Extracts six features from the full journey segment of the P5 recording
that capture how belt speed deviates from the G0 baseline.  Journey
duration is the single most discriminative feature (normal ~39 s, slow
cases up to ~67 s).  RMS features over the first and second halves of the
journey proxy Conveyor 1 (Rail 2) and Conveyor 2 (Rail 3) behaviour
respectively.

Note: the fixed handoff window (18–38 s) is intentionally not used here.
At non-standard belt speeds the handoff transient shifts in time, so any
feature tied to a fixed time window would be confounded by belt speed
rather than measuring it.
"""

import numpy as np
import pandas as pd


def extract_belt_speed_features(
    df_journey: pd.DataFrame,
    df_g0_journey: pd.DataFrame,
    fs: float = 100.0,
) -> dict:
    """Extract belt-speed features from a P5 journey segment.

    Computes six features that characterise how belt speed (Rail 2 and
    Rail 3) differs from the G0 reference recording.  All features are
    derived from the full journey segment; no fixed-time sub-window is
    applied.

    Parameters
    ----------
    df_journey : pd.DataFrame
        Journey segment for the case being analysed, as produced by
        ``preprocessor.segment_journey`` (the ``'journey'`` key).  Must
        contain columns ``time``, ``acc_x``, ``acc_y``, ``acc_z``,
        ``acc_abs``.
    df_g0_journey : pd.DataFrame
        Journey segment from the G0 baseline recording for the same phone
        position.  Same column requirements as *df_journey*.  Used to
        compute the duration deviation feature.
    fs : float, optional
        Sampling rate of both DataFrames in Hz.  Default is 100.0.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'journey_duration'`` : float
            Total duration of the journey segment in seconds::

                df_journey['time'].iloc[-1] - df_journey['time'].iloc[0]

        ``'duration_delta'`` : float
            Deviation from the G0 baseline journey duration in seconds::

                journey_duration - g0_journey_duration

            Positive values indicate a slower-than-normal belt; negative
            values indicate a faster-than-normal belt.

        ``'rms_full'`` : float
            RMS of ``acc_abs`` over the entire journey segment.

        ``'rms_first_half'`` : float
            RMS of ``acc_abs`` over the first half of the journey segment
            by time.  Proxies Rail 2 / Conveyor 1 (Loc 5 → Loc 4).

        ``'rms_second_half'`` : float
            RMS of ``acc_abs`` over the second half of the journey segment
            by time.  Proxies Rail 3 / Conveyor 2 (Loc 3 → Loc 2).

        ``'rms_ratio'`` : float
            Ratio ``rms_first_half / rms_second_half``.  Captures the
            relative energy balance between the two conveyors.  Returns
            ``np.nan`` if *rms_second_half* is zero.

    Notes
    -----
    The first/second half split is made at the temporal midpoint of the
    journey segment::

        t_mid = df_journey['time'].iloc[0] + journey_duration / 2

    Samples with ``time < t_mid`` form the first half; the rest form the
    second half.  This time-based split is used rather than a sample-count
    split so that the boundary is stable even if the DataFrame is
    non-uniformly sampled.
    """
    time = df_journey["time"].to_numpy(dtype=float)
    acc_abs = df_journey["acc_abs"].to_numpy(dtype=float)

    # ------------------------------------------------------------------
    # 1. journey_duration
    # ------------------------------------------------------------------
    journey_duration = float(time[-1] - time[0])

    # ------------------------------------------------------------------
    # 2. duration_delta
    # ------------------------------------------------------------------
    g0_time = df_g0_journey["time"].to_numpy(dtype=float)
    g0_journey_duration = float(g0_time[-1] - g0_time[0])
    duration_delta = journey_duration - g0_journey_duration

    # ------------------------------------------------------------------
    # 3. rms_full
    # ------------------------------------------------------------------
    rms_full = float(np.sqrt(np.mean(acc_abs ** 2)))

    # ------------------------------------------------------------------
    # 4 & 5. rms_first_half, rms_second_half — split at temporal midpoint
    # ------------------------------------------------------------------
    t_mid = time[0] + journey_duration / 2.0
    first_half = acc_abs[time < t_mid]
    second_half = acc_abs[time >= t_mid]

    rms_first_half = float(np.sqrt(np.mean(first_half ** 2))) if first_half.size > 0 else np.nan
    rms_second_half = float(np.sqrt(np.mean(second_half ** 2))) if second_half.size > 0 else np.nan

    # ------------------------------------------------------------------
    # 6. rms_ratio
    # ------------------------------------------------------------------
    if rms_second_half == 0.0 or np.isnan(rms_second_half):
        rms_ratio: float = np.nan
    else:
        rms_ratio = rms_first_half / rms_second_half

    return {
        "journey_duration": journey_duration,
        "duration_delta": duration_delta,
        "rms_full": rms_full,
        "rms_first_half": rms_first_half,
        "rms_second_half": rms_second_half,
        "rms_ratio": rms_ratio,
    }
