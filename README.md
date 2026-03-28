# Intelligent Monitoring of Engineering Systems
## Defect Detection in Conveyor Systems Using Smartphone Accelerometers

RWTH Aachen University — IMES Course — SS 2025
Rebuilt in Python as a portfolio project demonstrating professional ML engineering practices.

![Python](https://img.shields.io/badge/Python-3.13-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-latest-orange)
![MLflow](https://img.shields.io/badge/MLflow-tracked-green)
![Tests](https://img.shields.io/badge/tests-22%20passing-brightgreen)

---

## Problem

This project detects and characterises four types of defects in a mock conveyor production system using only smartphone accelerometer data. Five phones were deployed per recording — one riding with the conveyor (P5) and four fixed on the frame (P1–P4) — across 270 recordings from nine experimental groups.

---

## 🏭 Physical Setup

A conveyor system with two rails and a robotic arm at each end. Phone 5 makes a complete journey through the system on every run:

```
Arm 1 (Loc8 → Loc6) → Conveyor 1 (Loc5 → Loc4) → handoff → Conveyor 2 (Loc3 → Loc2) → Arm 2 → bin
```

**Sensors by role:**

| Sensor | Role |
|---|---|
| P5 (moving) | Frequency detection, inclination, belt speed |
| P1–P4 (fixed, frame-mounted) | Damping detection only |

P5 never physically contacts the sponge (which is under the frame leg), so it cannot sense the damping defect. The four fixed phones sense structural vibration transmitted through the frame — which is the correct signal for detecting sponge attenuation.

---

## 📊 Dataset

| Property | Value |
|---|---|
| Groups | 9 (G1–G9) |
| Cases per group | 6 |
| Phones per case | 5 |
| Total files | 270 |
| Train set | G1–G7 |
| Test set | G8–G9 (held out entirely) |
| Sampling rate | ~100 Hz (P5), ~71–400 Hz (fixed phones, resampled to 100 Hz) |

Each group contains six case types: frequency only, inclination only, damping only, belt speed only, frequency + inclination, and frequency + damping. Defect parameters (frequency value, inclination angle, belt speed percentages) vary across groups.

---

## 🔬 Approach

**Frequency detection.**
Full-journey FFT computed on `acc_x`, `acc_y`, `acc_z` individually (never on `acc_abs`, which produces intermodulation artefacts). The dominant peak in the 25–50 Hz search band is extracted. This is pure signal processing — no ML model is required. The FFT peak generalises perfectly across all groups.

**Inclination detection.**
Two-stage approach: (1) binary detection using SVC — is inclination present at Loc 4? (2) signed angle regression using GBM — what is the angle in degrees? A 4th-order Butterworth low-pass filter at 20 Hz is applied before computing magnitude features to remove 30–50 Hz frequency contamination in combined-defect cases. Inclination at Loc 5 (Arm 1 deposit point) is physically non-detectable in the 30–37 s window because gravity is removed from the accelerometer data; these cases are correctly treated as label = 0.

**Belt speed detection.**
Multi-output regression predicting Rail 2 and Rail 3 speeds simultaneously. Journey duration is the primary feature (normal speed ~46 s, slow speed ~71 s), supplemented by duration deviation from the G0 reference and RMS statistics over the full journey segment.

**Damping detection.**
Binary classification using fixed phones P1–P4 only. Raw G0-normalised RMS ratios are contaminated when a frequency generator is active (e.g. `rms_ratio_P1 = 7.2` for a frequency-only case). Phone-to-phone relative ratios (`rms_P3 / rms_P1`, `rms_P4 / rms_P1`, etc.) cancel this common-mode inflation because the frequency generator inflates all fixed phones proportionally. The handoff window (t = 18–38 s) is excluded from all fixed-phone RMS calculations — fixed phones show large transient spikes during handoff events that are unrelated to damping.

---

## 📈 Results

Evaluated on completely held-out groups G8–G9, never seen during training or hyperparameter tuning.

| Problem | Best Model | CV Score | Test Score | Metric |
|---|---|---|---|---|
| Frequency detection | SVC | 1.000 | 1.000 | F1 |
| Inclination detection | SVC | 0.914 | 1.000 | F1 |
| Inclination angle | GBM | 0.758° | 0.179° | MAE |
| Belt speed Rail 2 | RF | 4.24% | 3.90% | MAE |
| Belt speed Rail 3 | RF | — | 5.92% | MAE |
| Damping detection | RF | 0.648 | 1.000 | AUC |

---

## ⚙️ Key Engineering Decisions

- **Leave-One-Group-Out CV** prevents session-level data leakage — k-fold CV on pooled windowed data is misleading (the original MATLAB model achieved 3% CV error but generalised poorly for this reason)
- **Physically motivated features** over raw signal windows — FFT peaks, impact transient jerk, journey duration, and frame RMS encode the underlying physics
- **Low-pass filter at 20 Hz** isolates the inclination impact transient from frequency defect noise in combined cases
- **Phone-to-phone RMS ratios** remove common-mode frequency contamination from damping features (improved AUC from 0.938 to 1.000)
- **Two-stage inclination pipeline** — detect presence first, then regress angle only on confirmed positives — avoids conflating detection and measurement
- **G8–G9 held out entirely** — never seen during training or tuning; the test set is touched exactly once

---

## 📁 Project Structure

```
imes-defect-detection/
├── data/
│   ├── raw/                  # XLS files — not tracked in git
│   └── processed/            # Extracted features, saved models, eval figures
├── src/
│   ├── data_loader.py        # XLS → DataFrame, normalises column names
│   ├── preprocessor.py       # Resample, segment journey, detect landmarks
│   ├── build_features.py     # Batch feature extraction → CSV
│   ├── features/
│   │   ├── frequency.py      # FFT feature extraction
│   │   ├── inclination.py    # Impact transient features
│   │   ├── belt_speed.py     # Journey duration + RMS features
│   │   └── damping.py        # Fixed-phone RMS ratio features
│   └── models/
│       ├── train.py          # Train all models, log to MLflow
│       └── evaluate.py       # Holdout evaluation + professor test cases
├── notebooks/
│   ├── 01_eda.ipynb                  # Signal visualisation, landmark validation
│   ├── 02_feature_engineering.ipynb  # Feature distributions, physical validation
│   ├── 03_model_experiments.ipynb    # Model comparison, feature importance
│   └── 04_results_report.ipynb       # Final results presentation
├── tests/                    # 22 tests, all passing
└── PROJECT_CONTEXT.md        # Design decisions and physical context
```

---

## 🚀 How to Run

```bash
# Clone and set up environment
git clone https://github.com/SharathC01/imes-defect-detection.git
cd imes-defect-detection
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

```bash
# Place raw XLS files in data/raw/G0/ ... data/raw/G9/

# Extract features from all groups
python -m src.build_features

# Train all models (logs runs to MLflow)
python -m src.models.train

# Evaluate on G8-G9 holdout and professor test cases
python -m src.models.evaluate

# View experiment runs
mlflow ui
```

```bash
# Run test suite
python -m pytest tests/ -v
```

---

## Requirements

Python 3.13 with the following packages:

```
pandas
numpy
scipy
scikit-learn
matplotlib
seaborn
mlflow
xlrd
openpyxl
joblib
jupyter
pytest
```

---

## Limitations

- Inclination at Loc 5 (Arm 1 deposit) and Loc 2 (Conveyor 2 end) is not detectable — gravity is removed from the accelerometer data so static tilt produces no measurable signal; only the dynamic impact transient at Loc 4 is used
- Damping at Loc 6 (arm track) is not reliably detected — the attenuation signal is too weak for the fixed phones to sense at that distance
- Belt speed MAE of approximately 5% limits precise RPM prediction, particularly for speed combinations not present in the training groups
- Fixed phone placement varies between groups, which affects the absolute RMS magnitudes and introduces group-to-group variability in the damping features
