"""Data loader for IMES accelerometer XLS files."""

import pandas as pd


_RAW_SHEET = "Raw Data"
_META_SHEET = "Metadata Device"
_MINDELAY_ROW_LABEL = "accelerometer MinDelay"
_DEFAULT_FS = 100.0
_COLUMN_NAMES = ["time", "acc_x", "acc_y", "acc_z", "acc_abs"]


def load_file(filepath: str) -> tuple[pd.DataFrame, float]:
    """Load a single IMES accelerometer XLS file.

    Parameters
    ----------
    filepath : str
        Path to the .xls file (e.g. ``data/raw/G1/G1_P5_case11.xls``).

    Returns
    -------
    df : pd.DataFrame
        Accelerometer data with columns
        ``['time', 'acc_x', 'acc_y', 'acc_z', 'acc_abs']``.
    fs : float
        Device sampling rate in Hz, derived from the ``accelerometer MinDelay``
        field in the ``Metadata Device`` sheet (microseconds → Hz).
        Falls back to ``100.0`` if the sheet is absent or the row is missing.

    Notes
    -----
    The ``accelerometer MinDelay`` field stores the minimum time between two
    sensor events in **microseconds**.  Sampling rate is therefore::

        fs = 1_000_000 / MinDelay
    """
    df = pd.read_excel(filepath, sheet_name=_RAW_SHEET, header=0)
    df.columns = _COLUMN_NAMES

    fs = _read_sampling_rate(filepath)
    return df, fs


def _read_sampling_rate(filepath: str) -> float:
    """Extract sampling rate from the Metadata Device sheet.

    Parameters
    ----------
    filepath : str
        Path to the .xls file.

    Returns
    -------
    float
        Sampling rate in Hz, or ``100.0`` as a safe default.
    """
    try:
        meta = pd.read_excel(filepath, sheet_name=_META_SHEET, header=None)
    except Exception:
        return _DEFAULT_FS

    # Find the row whose first column matches the label (case-insensitive strip)
    mask = meta.iloc[:, 0].astype(str).str.strip().str.lower() == _MINDELAY_ROW_LABEL.lower()
    matching = meta[mask]

    if matching.empty:
        return _DEFAULT_FS

    try:
        min_delay_us = float(matching.iloc[0, 1])
        if min_delay_us <= 0:
            return _DEFAULT_FS
        return 1_000_000.0 / min_delay_us
    except (ValueError, TypeError, IndexError):
        return _DEFAULT_FS
