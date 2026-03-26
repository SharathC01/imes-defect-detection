"""Tests for src/data_loader.py."""

from unittest.mock import patch
import pandas as pd
import pytest

from src.data_loader import load_file, _COLUMN_NAMES, _DEFAULT_FS


# ---------------------------------------------------------------------------
# Helpers — minimal fake data returned by pd.read_excel
# ---------------------------------------------------------------------------

def _make_raw_df() -> pd.DataFrame:
    """Five-column raw accelerometer data (10 rows)."""
    return pd.DataFrame(
        {
            0: range(10),          # time
            1: [0.1] * 10,         # acc_x
            2: [0.2] * 10,         # acc_y
            3: [0.3] * 10,         # acc_z
            4: [0.4] * 10,         # acc_abs
        }
    )


def _make_meta_df_valid() -> pd.DataFrame:
    """Metadata sheet with accelerometer MinDelay = 10000 µs → 100 Hz."""
    return pd.DataFrame(
        {
            0: ["accelerometer MinDelay", "some other field"],
            1: [10000, 999],
        }
    )


def _make_meta_df_missing_row() -> pd.DataFrame:
    """Metadata sheet that does NOT contain the MinDelay row."""
    return pd.DataFrame(
        {
            0: ["accelerometer Range", "accelerometer Resolution"],
            1: [39.2, 0.001],
        }
    )


# ---------------------------------------------------------------------------
# Test 1 — valid file: correct column names returned
# ---------------------------------------------------------------------------

def test_load_file_returns_correct_column_names():
    """load_file renames raw columns to the standard five-name schema."""

    def fake_read_excel(filepath, sheet_name, header=0):
        if sheet_name == "Raw Data":
            return _make_raw_df()
        return _make_meta_df_valid()

    with patch("src.data_loader.pd.read_excel", side_effect=fake_read_excel):
        df, _ = load_file("fake/path/G1_P5_case11.xls")

    assert list(df.columns) == _COLUMN_NAMES


# ---------------------------------------------------------------------------
# Test 2 — sampling rate is a float
# ---------------------------------------------------------------------------

def test_load_file_returns_float_for_fs():
    """fs must be a Python float regardless of what the metadata contains."""

    def fake_read_excel(filepath, sheet_name, header=0):
        if sheet_name == "Raw Data":
            return _make_raw_df()
        return _make_meta_df_valid()

    with patch("src.data_loader.pd.read_excel", side_effect=fake_read_excel):
        _, fs = load_file("fake/path/G1_P5_case11.xls")

    assert isinstance(fs, float)
    assert fs == pytest.approx(100.0)


# ---------------------------------------------------------------------------
# Test 3 — missing metadata sheet falls back gracefully
# ---------------------------------------------------------------------------

def test_load_file_handles_missing_metadata_sheet():
    """load_file must not crash when the Metadata Device sheet is absent."""

    def fake_read_excel(filepath, sheet_name, header=0):
        if sheet_name == "Raw Data":
            return _make_raw_df()
        raise ValueError(f"Sheet '{sheet_name}' not found")

    with patch("src.data_loader.pd.read_excel", side_effect=fake_read_excel):
        df, fs = load_file("fake/path/G1_P5_case11.xls")

    assert list(df.columns) == _COLUMN_NAMES
    assert fs == pytest.approx(_DEFAULT_FS)
