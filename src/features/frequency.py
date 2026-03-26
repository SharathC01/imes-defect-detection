"""Frequency-domain feature extraction for IMES defect detection.

Uses FFT to detect the dominant vibration frequency on the full journey segment,
then STFT to localise that frequency in time.  Both functions operate on
individual axes (acc_x, acc_y, acc_z) and never on acc_abs, which would
introduce non-linear combination artefacts.
"""

import numpy as np
import pandas as pd
from scipy.signal import stft


def extract_dominant_frequency(
    df_journey: pd.DataFrame,
    fs: float = 100.0,
) -> dict:
    """Detect the dominant defect frequency from the full journey segment.

    Runs a real-valued FFT independently on ``acc_x``, ``acc_y``, and
    ``acc_z``.  Only the 25–50 Hz band is considered; this excludes the
    mechanical conveyor resonance that appears below 25 Hz in all cases.
    The axis whose peak magnitude is highest is declared the dominant axis.

    Parameters
    ----------
    df_journey : pd.DataFrame
        Journey segment produced by ``preprocessor.segment_journey``
        (the ``'journey'`` key).  Must contain columns ``acc_x``,
        ``acc_y``, ``acc_z``.
    fs : float, optional
        Sampling rate of *df_journey* in Hz.  Default is 100.0.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'dominant_freq_hz'`` : float
            Peak frequency (Hz) on the dominant axis.
        ``'dominant_axis'`` : str
            Name of the axis with the highest peak magnitude —
            one of ``'acc_x'``, ``'acc_y'``, ``'acc_z'``.
        ``'peak_magnitude'`` : float
            FFT magnitude at the dominant peak.
        ``'freq_x'`` : float
            Peak frequency (Hz) in the 25–50 Hz band on ``acc_x``.
        ``'freq_y'`` : float
            Peak frequency (Hz) in the 25–50 Hz band on ``acc_y``.
        ``'freq_z'`` : float
            Peak frequency (Hz) in the 25–50 Hz band on ``acc_z``.
        ``'mag_x'`` : float
            FFT magnitude at ``freq_x``.
        ``'mag_y'`` : float
            FFT magnitude at ``freq_y``.
        ``'mag_z'`` : float
            FFT magnitude at ``freq_z``.

    Notes
    -----
    The one-sided magnitude spectrum is computed as::

        magnitude = (2 / N) * |rfft(signal)|

    where N is the number of samples.  Dividing by N normalises for signal
    length; the factor of 2 compensates for the discarded negative frequencies
    (except the DC and Nyquist bins, which are not in the search range).

    The search is restricted to the closed interval [25 Hz, 50 Hz].
    """
    axes = ["acc_x", "acc_y", "acc_z"]

    freq_results: dict[str, float] = {}
    mag_results: dict[str, float] = {}

    for axis in axes:
        signal = df_journey[axis].to_numpy(dtype=float)
        n = len(signal)

        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        magnitude = (2.0 / n) * np.abs(np.fft.rfft(signal))

        # Restrict to defect search band: 25–50 Hz
        band_mask = (freqs >= 25.0) & (freqs <= 50.0)
        band_freqs = freqs[band_mask]
        band_mags = magnitude[band_mask]

        peak_idx = int(np.argmax(band_mags))
        freq_results[axis] = float(band_freqs[peak_idx])
        mag_results[axis] = float(band_mags[peak_idx])

    dominant_axis = max(axes, key=lambda a: mag_results[a])

    return {
        "dominant_freq_hz": freq_results[dominant_axis],
        "dominant_axis": dominant_axis,
        "peak_magnitude": mag_results[dominant_axis],
        "freq_x": freq_results["acc_x"],
        "freq_y": freq_results["acc_y"],
        "freq_z": freq_results["acc_z"],
        "mag_x": mag_results["acc_x"],
        "mag_y": mag_results["acc_y"],
        "mag_z": mag_results["acc_z"],
    }


def locate_frequency_in_time(
    df_journey: pd.DataFrame,
    dominant_freq: float,
    fs: float = 100.0,
) -> dict:
    """Find the time window of peak energy at the dominant defect frequency.

    Runs a Short-Time Fourier Transform (STFT) on the axis identified as
    dominant by :func:`extract_dominant_frequency` and finds the time
    segment where spectral power at *dominant_freq* is highest.

    Parameters
    ----------
    df_journey : pd.DataFrame
        Journey segment produced by ``preprocessor.segment_journey``
        (the ``'journey'`` key).  Must contain columns ``acc_x``,
        ``acc_y``, ``acc_z``.
    dominant_freq : float
        The defect frequency in Hz as returned in
        ``extract_dominant_frequency(...)['dominant_freq_hz']``.
    fs : float, optional
        Sampling rate of *df_journey* in Hz.  Default is 100.0.

    Returns
    -------
    dict
        A dictionary with the following keys:

        ``'defect_time_start'`` : float
            Start time (seconds, relative to recording start) of the STFT
            window with the highest energy at *dominant_freq*.
        ``'defect_time_end'`` : float
            End time (seconds) of that window.
        ``'defect_time_centre'`` : float
            Centre time (seconds) of that window,
            i.e. ``(defect_time_start + defect_time_end) / 2``.

    Notes
    -----
    STFT parameters: ``nperseg=256``, ``noverlap=128``, ``window='hann'``
    (scipy default).  The STFT time vector returned by
    ``scipy.signal.stft`` is relative to the start of the signal array;
    it is shifted by the first timestamp in *df_journey* so that the
    returned times align with the original recording clock.

    The frequency bin closest to *dominant_freq* is selected using
    ``np.argmin(np.abs(stft_freqs - dominant_freq))``.  The time index of
    the maximum magnitude at that bin is used to identify the peak window.
    The window boundaries are half a segment width on either side of the
    STFT time point, i.e.::

        half_width = (nperseg / 2) / fs   # seconds
        defect_time_start = t_peak - half_width
        defect_time_end   = t_peak + half_width
    """
    nperseg = 256
    noverlap = 128

    # Determine dominant axis (re-run FFT to avoid requiring the caller to pass it)
    fft_result = extract_dominant_frequency(df_journey, fs=fs)
    dominant_axis = fft_result["dominant_axis"]

    signal = df_journey[dominant_axis].to_numpy(dtype=float)
    t_offset = float(df_journey["time"].iloc[0])

    stft_freqs, stft_times, Zxx = stft(
        signal,
        fs=fs,
        nperseg=nperseg,
        noverlap=noverlap,
    )

    # Shift STFT times to match the recording clock
    stft_times_abs = stft_times + t_offset

    # Find the frequency bin closest to the dominant defect frequency
    freq_bin = int(np.argmin(np.abs(stft_freqs - dominant_freq)))

    # Power at the selected frequency bin across all time windows
    power_at_freq = np.abs(Zxx[freq_bin, :])
    peak_time_idx = int(np.argmax(power_at_freq))
    t_peak = float(stft_times_abs[peak_time_idx])

    half_width = (nperseg / 2.0) / fs
    defect_time_start = t_peak - half_width
    defect_time_end = t_peak + half_width
    defect_time_centre = t_peak

    return {
        "defect_time_start": defect_time_start,
        "defect_time_end": defect_time_end,
        "defect_time_centre": defect_time_centre,
    }
