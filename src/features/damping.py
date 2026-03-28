"""Damping feature extraction for IMES defect detection.

Extracts RMS energy features from the four fixed phones (P1–P4) to detect
whether a sponge block is placed under a conveyor frame leg.  The sponge
attenuates structural vibration energy; the phone nearest the sponge will
show the largest reduction in RMS relative to the G0 baseline.

Phone 5 (the moving phone) is intentionally excluded — the sponge is
located under the frame leg, not on the belt, so P5 never physically
interacts with it.

The handoff window (t = 18 s to t = 38 s) is excluded from all RMS
calculations.  Fixed phones show large transient spikes during handoff
events (P5 dropping between conveyors) that are unrelated to damping and
would dominate the features if included.  Only samples in the C1 segment
(t = 4–18 s) and C2 segment (t = 38–43 s) are used.
"""

import numpy as np
import pandas as pd

_PHONES = ["P1", "P2", "P3", "P4"]

# Time mask boundaries (seconds)
_T_C1_START = 4.0
_T_C1_END = 18.0
_T_C2_START = 38.0
_T_C2_END = 43.0


def _rms_excluding_handoff(df: pd.DataFrame) -> float:
    """Compute RMS of acc_abs for the C1 and C2 segments only.

    Parameters
    ----------
    df : pd.DataFrame
        Full resampled DataFrame for one fixed phone.  Must contain
        columns ``time`` and ``acc_abs``.

    Returns
    -------
    float
        RMS value, or ``np.nan`` if no samples fall within the mask.
    """
    time = df["time"].to_numpy(dtype=float)
    acc_abs = df["acc_abs"].to_numpy(dtype=float)

    mask = (
        ((time >= _T_C1_START) & (time <= _T_C1_END))
        | ((time >= _T_C2_START) & (time <= _T_C2_END))
    )
    values = acc_abs[mask]

    if values.size == 0:
        return np.nan

    return float(np.sqrt(np.mean(values ** 2)))


def extract_damping_features(
    phones: dict[str, pd.DataFrame],
    g0_phones: dict[str, pd.DataFrame],
    fs: float = 100.0,
) -> dict:
    """Extract damping features from the four fixed phones.

    Computes per-phone RMS energy (excluding the handoff window) and the
    ratio of each phone's RMS to its G0 baseline value.  A ratio below 1.0
    indicates that the sponge is attenuating vibration energy at that
    location.  Two summary features identify which phone shows the greatest
    attenuation.

    Parameters
    ----------
    phones : dict[str, pd.DataFrame]
        Mapping of phone name to its full resampled DataFrame for the case
        being analysed.  Expected keys: ``'P1'``, ``'P2'``, ``'P3'``,
        ``'P4'``.  Each DataFrame must contain columns ``time``, ``acc_x``,
        ``acc_y``, ``acc_z``, ``acc_abs``.
    g0_phones : dict[str, pd.DataFrame]
        Same structure as *phones* but for the G0 baseline recording.
        Used to normalise per-phone RMS values against the reference.
    fs : float, optional
        Sampling rate of all DataFrames in Hz.  Default is 100.0.
        (Not used in computation but included for API consistency.)

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'rms_P1'``, ``'rms_P2'``, ``'rms_P3'``, ``'rms_P4'`` : float
            RMS of ``acc_abs`` for each phone, computed over the C1 segment
            (t = 4–18 s) and C2 segment (t = 38–43 s), excluding the
            handoff window (t = 18–38 s).

        ``'rms_ratio_P1'``, ``'rms_ratio_P2'``, ``'rms_ratio_P3'``,
        ``'rms_ratio_P4'`` : float
            Ratio of each phone's RMS to the G0 baseline RMS::

                rms_ratio_P = rms_P / g0_rms_P

            Values below 1.0 indicate damping at that location.  Returns
            ``np.nan`` if the G0 RMS for that phone is zero.

        ``'min_rms_ratio'`` : float
            The minimum ``rms_ratio`` across all four phones.  The phone
            with the lowest ratio is closest to the sponge.  ``np.nan``
            values are ignored when computing the minimum.

        ``'min_rms_ratio_phone'`` : str
            Name of the phone (``'P1'``–``'P4'``) whose ``rms_ratio`` is
            the minimum.  If all ratios are ``np.nan``, returns
            ``'unknown'``.

        ``'rms_ratio_P3_to_P1'``, ``'rms_ratio_P4_to_P1'``,
        ``'rms_ratio_P3_to_P2'``, ``'rms_ratio_P4_to_P2'`` : float
            Phone-to-phone RMS ratios::

                rms_ratio_P3_to_P1 = rms_P3 / rms_P1
                rms_ratio_P4_to_P1 = rms_P4 / rms_P1
                rms_ratio_P3_to_P2 = rms_P3 / rms_P2
                rms_ratio_P4_to_P2 = rms_P4 / rms_P2

            Phone-to-phone ratios are robust to frequency defect
            contamination.  When a frequency generator is active, it
            inflates RMS of all fixed phones proportionally.  Taking
            ratios between phones cancels this common-mode inflation,
            leaving only the spatial damping signal.  Returns ``np.nan``
            if the denominator phone RMS is zero.

    Notes
    -----
    The handoff exclusion mask is::

        (time >= 4) & (time <= 18) | (time >= 38) & (time <= 43)

    This matches the guidance in PROJECT_CONTEXT.md (Defect 4 — Damping):
    use only t = 4–18 s and t = 38–43 s for fixed-phone RMS features.
    """
    rms_per_phone: dict[str, float] = {}
    rms_ratio_per_phone: dict[str, float] = {}

    for phone in _PHONES:
        rms = _rms_excluding_handoff(phones[phone])
        g0_rms = _rms_excluding_handoff(g0_phones[phone])

        rms_per_phone[phone] = rms

        if g0_rms == 0.0 or np.isnan(g0_rms):
            rms_ratio_per_phone[phone] = np.nan
        else:
            rms_ratio_per_phone[phone] = rms / g0_rms

    # ------------------------------------------------------------------
    # Phone-to-phone relative ratios (robust to frequency defect)
    # ------------------------------------------------------------------
    rms_P1 = rms_per_phone["P1"]
    rms_P2 = rms_per_phone["P2"]
    rms_P3 = rms_per_phone["P3"]
    rms_P4 = rms_per_phone["P4"]

    rms_ratio_P3_to_P1 = rms_P3 / rms_P1 if rms_P1 != 0.0 and not np.isnan(rms_P1) else np.nan
    rms_ratio_P4_to_P1 = rms_P4 / rms_P1 if rms_P1 != 0.0 and not np.isnan(rms_P1) else np.nan
    rms_ratio_P3_to_P2 = rms_P3 / rms_P2 if rms_P2 != 0.0 and not np.isnan(rms_P2) else np.nan
    rms_ratio_P4_to_P2 = rms_P4 / rms_P2 if rms_P2 != 0.0 and not np.isnan(rms_P2) else np.nan

    # ------------------------------------------------------------------
    # Summary features: which phone shows the greatest attenuation
    # ------------------------------------------------------------------
    ratios = np.array([rms_ratio_per_phone[p] for p in _PHONES], dtype=float)

    if np.all(np.isnan(ratios)):
        min_rms_ratio: float = np.nan
        min_rms_ratio_phone: str = "unknown"
    else:
        min_idx = int(np.nanargmin(ratios))
        min_rms_ratio = float(ratios[min_idx])
        min_rms_ratio_phone = _PHONES[min_idx]

    return {
        "rms_P1": rms_per_phone["P1"],
        "rms_P2": rms_per_phone["P2"],
        "rms_P3": rms_per_phone["P3"],
        "rms_P4": rms_per_phone["P4"],
        "rms_ratio_P1": rms_ratio_per_phone["P1"],
        "rms_ratio_P2": rms_ratio_per_phone["P2"],
        "rms_ratio_P3": rms_ratio_per_phone["P3"],
        "rms_ratio_P4": rms_ratio_per_phone["P4"],
        "min_rms_ratio": min_rms_ratio,
        "min_rms_ratio_phone": min_rms_ratio_phone,
        "rms_ratio_P3_to_P1": rms_ratio_P3_to_P1,
        "rms_ratio_P4_to_P1": rms_ratio_P4_to_P1,
        "rms_ratio_P3_to_P2": rms_ratio_P3_to_P2,
        "rms_ratio_P4_to_P2": rms_ratio_P4_to_P2,
    }
