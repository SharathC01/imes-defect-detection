# IMES Project — Corrections and Findings
# These override or supplement PROJECT_CONTEXT.md where they conflict.
# Read this ALONGSIDE PROJECT_CONTEXT.md before writing any code.

---

## 1. NumPy Version
`np.trapz` is removed in NumPy 2.0.
Use `np.trapezoid` everywhere instead.

---

## 2. G0 Filename Convention
G0 files are named: `G0_P{n}_case_perfect.xls`
NOT `G0_P{n}_case01.xls` like other groups.

---

## 3. segment_journey() Returns 4 Keys [LOCKED]
`segment_journey()` returns a dict with exactly these keys:
  'full', 'journey', 'handoff', 'inclination'

The 'inclination' segment covers t=30s to t=37s.
The 'handoff' segment covers t=18s to t=38s.
These are two different segments — do not confuse them.

---

## 4. Inclination Window is 30-37s [LOCKED]
`extract_inclination_features()` takes `segments['inclination']` (t=30-37s).
NOT `segments['handoff']` (t=18-38s).
The 30-37s window was validated from MATLAB labelling across all groups.
Three outlier groups (G62, G82, G85) have slightly shifted windows but
all fall within 30-37s with margin.

---

## 5. Inclination Target is Signed Angle [LOCKED]
Inclination target is signed angle (-5 to +5 degrees).
Use regression only — NOT classification.
Gravity is removed from accelerometer data so static tilt is NOT detectable.
Direction is encoded in the raw acc_z value at the moment of peak impact.

Two-stage prediction:
  Stage 1: Binary detection (Loc4=present, Loc5/none=absent)
  Stage 2: Signed angle regression on Loc4 cases only (-5 to +5 degrees)

Location 5 inclination cases are NOT detectable in the 30-37s window.
These are treated as label=0 for the binary detection stage.

---

## 6. Low-Pass Filter in Inclination Features [LOCKED]
`extract_inclination_features()` applies a 4th-order Butterworth low-pass
filter at 20 Hz cutoff BEFORE computing magnitude features.
Reason: removes frequency defect contamination (30-50 Hz) from combined cases.

Implementation detail:
- Peak index is found using UNFILTERED acc_abs argmax
- Filtered signal is used only for: peak_jerk, impulse_energy, rise_time,
  post_impact_rms
- signed_peak_z uses raw acc_z value at the unfiltered peak index

---

## 7. Inclination Feature: signed_peak_z [LOCKED]
signed_peak_z = raw acc_z value (NOT absolute) at the unfiltered peak index.
Negative = downward impact (phone drops) = negative inclination angle.
Positive = upward impact (phone hits ramp) = positive inclination angle.
This is the key direction feature for signed angle regression.

---

## 8. Damping Function Signature [LOCKED]
```python
extract_damping_features(
    phones: dict[str, pd.DataFrame],     # full DataFrames, NOT pre-computed RMS
    g0_phones: dict[str, pd.DataFrame],  # full DataFrames, NOT pre-computed RMS
    fs: float = 100.0
) -> dict
```

Both arguments take full resampled DataFrames, not pre-computed RMS values.
Column names: time, acc_x, acc_y, acc_z, acc_abs

---

## 9. Damping Return Keys [LOCKED]
Exact return keys from extract_damping_features():
  rms_P1, rms_P2, rms_P3, rms_P4,
  rms_ratio_P1, rms_ratio_P2, rms_ratio_P3, rms_ratio_P4,
  min_rms_ratio, min_rms_ratio_phone

NOT: P1_rms_ratio, spatial_gradient_c1, rms_low, rms_mid, peak_freq
The implementation is simpler than originally planned — frequency band
features and spatial gradient were not implemented in this version.

---

## 10. Damping Handoff Exclusion [LOCKED]
Fixed phone RMS calculations exclude the handoff window (t=18s to t=38s).
Valid time zones for damping features:
  C1 segment: t=4s to t=18s
  C2 segment: t=38s to t=43s

Reason: fixed phones show large spikes during handoff events that are
unrelated to damping and dominate the RMS if included.

Boundary samples at exactly t=18.0 and t=38.0 ARE included in the
valid zone (mask uses <= and >=, not strict inequalities).

---

## 11. Belt Speed — No Fixed Sub-Windows [LOCKED]
The fixed 18-38s handoff window is NOT valid for belt speed cases.
At slow speeds the actual handoff occurs outside this window.
Belt speed features use the FULL journey segment only.
Do not use segments['handoff'] or segments['inclination'] for belt speed.

---

## 12. FFT/STFT — Use Individual Axes [LOCKED]
Do NOT run FFT or STFT on acc_abs.
acc_abs is a non-linear combination (sqrt of sum of squares) that produces
intermodulation artifacts at incorrect frequencies.

Always run on acc_x, acc_y, acc_z individually.
acc_z has the highest SNR for frequency defect detection (validated on G7).
Search band: 25-50 Hz only (excludes conveyor resonance below 25 Hz).

STFT showed wrong frequency (~21 Hz) when run on acc_abs and on a short
window. Full journey FFT on acc_z correctly detected 45 Hz for G7 case71.

---

## 13. Frequency Detection is Signal Processing Only [LOCKED]
No ML model for frequency detection.
FFT on full journey segment identifies the frequency value.
STFT on full journey segment identifies the time/location of the defect.
These generalise perfectly without any training.

---

## 14. Landmark Event Terminology Correction
The large spike at ~45s (normal speed) is P5 being DROPPED INTO THE BIN
by Arm 2 — NOT the arm pickup event.
The function detect_pickup_event() is a misnomer — it detects the bin drop.
Consider renaming to detect_end_event() in a future refactor.
Do not rename now as it would break existing tests.

---

## 15. Column Names After load_file() [LOCKED]
Always: time, acc_x, acc_y, acc_z, acc_abs
Never use original XLS column names in any downstream module.

---

## 16. Test Count
22/22 tests passing as of end of Step 4.
Test files:
  tests/test_data_loader.py        — 3 tests
  tests/test_preprocessor.py       — 5 tests
  tests/test_features_frequency.py — 4 tests
  tests/test_features_inclination.py — 5 tests
  tests/test_features_belt_speed.py — 4 tests
  tests/test_features_damping.py   — 5 tests

---

## 17. Sanity Check Pattern
Always test new modules with a scratch check.py on real G7 data
before writing tests. Delete check.py after verification.
Never commit check.py to the repo.
---

## 18. Frequency defect contaminates fixed phone RMS ratios
case71 (frequency defect only) shows rms_ratio_P1=7.2, rms_ratio_P2=18.9
against G0 baseline. The 45 Hz generator transmits energy through the frame
to fixed phones P1 and P2, massively inflating their RMS ratios.
Damping model must exclude cases with active frequency defects OR use
phone-to-phone relative ratios (P3/P1, P4/P2) rather than absolute G0 ratios.
