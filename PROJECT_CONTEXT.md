# IMES Defect Detection — Project Context for Claude Code

This document captures all design decisions made during ideation. Read this before
writing any code. Do not deviate from decisions marked **[LOCKED]** without discussion.

---

## What This Project Is

A portfolio-quality ML project that detects and characterises defects in a conveyor
production system using smartphone accelerometer data. Originally done in MATLAB as
a university coursework (RWTH Aachen, IMES course, SS 2025). Being rebuilt in Python
with proper ML engineering practices for GitHub portfolio purposes.

**Not just a homework redo.** The goal is to demonstrate:
- Physically motivated feature engineering
- Correct problem framing (right model type per defect)
- Honest generalisation evaluation (leave-group-out)
- Experiment tracking with MLflow
- Clean, reproducible, well-documented code

---

## Physical Setup **[LOCKED]**

A conveyor + robotic arm system. Phone 5 (P5) rides the system as a passenger.
Phones 1–4 (P1–P4) are fixed at static locations on the table/frame.

**Phone 5 journey (in order):**
```
Loc 8 → [Arm 1 track] → Loc 6 → drops onto Conveyor 1 at Loc 5
→ Loc 4 (end of C1) → [handoff] → Loc 3 (start of C2) → Loc 2 (end of C2)
→ [Arm 2 picks up] → bin
```

**Location reference:**
- Loc 8: Arm 1 start (picks up P5)
- Loc 6: End of Arm 1 track (P5 deposited onto C1)
- Loc 5: Start of Conveyor 1
- Loc 4: End of Conveyor 1 — **handoff zone start**
- Loc 3: Start of Conveyor 2 — **handoff zone end**
- Loc 2: End of Conveyor 2 (Arm 2 picks P5 up)

**Defect devices:**
- Frequency defect: vibration frequency generator clamped to frame
- Inclination defect: scissor lift raises a section of the conveyor
- Damping defect: sponge block placed **under the conveyor frame leg** (not on belt)
- Belt speed defect: motor controller set to different % of max RPM

---

## Dataset Structure **[LOCKED]**

**Files:** `G{g}_P{p}_case{g}{c}.xls` where g=group(1–9), p=phone(1–5), c=case(1–6)

**G0 — Perfect reference dataset:**
- No defects, normal belt speed (Rail2=70, Rail3=100)
- Same 5 phone positions as all other groups
- Used as baseline for feature normalisation:
  - Damping: feature = case_RMS / G0_RMS per phone (normalises device sensitivity)
  - Belt speed: feature = journey_duration - G0_duration (deviation from reference)
  - Inclination: G0 handoff transient = zero-angle reference
  - Frequency: G0 STFT = clean "no vibration defect" reference for EDA

**Case types (same structure across all 9 groups):**
| Case suffix | Defects present | Belt speed (Rail2/Rail3) |
|---|---|---|
| _1 | Frequency only | 70/100 (normal) |
| _2 | Inclination only | 70/100 (normal) |
| _3 | Damping only | 70/100 (normal) |
| _4 | Belt speed only | Varies per group (abnormal) |
| _5 | Frequency + Inclination | 70/100 (normal) |
| _6 | Frequency + Damping | 70/100 (normal) |

**Belt speed values in case _4 per group (Rail2 / Rail3):**
- G1: 70/100, G2: 80/100, G3: 60/30, G4: 40/50, G5: 100/70
- G6: 30/60, G7: 30/30, G8: 100/60, G9: 40/70
- Normal reference: Rail2=70, Rail3=100

**Defect parameters vary by group:**
- Frequency: 30–50 Hz, at locations 2, 3, 5, or 6
- Inclination: 1°–5°, at locations 4 or 5
- Damping: always present/absent binary, at locations 4, 5, or 6

**Total: 9 groups × 6 cases × 5 phones = 270 files**

---

## Signal Characteristics (from G7 P5 sample analysis) **[LOCKED]**

**Phone 5 (moving):**
- Device: Samsung SM-A202F
- Sampling rate: ~100 Hz (consistent, no resampling needed for P5-only analysis)
- Accelerometer range: ±39.2 m/s²
- Duration: ~46s (normal speed), ~71s (slow speed — case74 example)
- Columns: Time(s), Acc_x, Acc_y, Acc_z, Abs_acc

**Fixed phones (P1–P4):**
- Different devices per group (e.g., IV2201 at 398 Hz)
- Sampling rates vary: ~71–400 Hz → **resampling to 100 Hz required**
- Accelerometer range varies (e.g., ±78.5 m/s²)
- Much lower amplitude (structural vibration only, max ~0.76 m/s²)

**Landmark events in P5 signal (approximate, consistent across cases):**
- t ≈ 0–5s: Arm 1 deposits P5 onto C1 (small/no large peak)
- t ≈ 22–25s: First handoff transient (C1 → transfer zone)
- t ≈ 33–34s: Second handoff transient (transfer zone → C2)
- t ≈ 45s: Arm 2 picks up P5 — **always the largest peak (~37–39 m/s²)**
- For slow speed cases: pickup event shifts to ~70s

---

## The Four Defect Problems **[LOCKED]**

### Defect 1 — Frequency
- **Approach: Signal processing only, NO ML**
- Tool: STFT (scipy.signal.stft)
- Output: dominant frequency (Hz) + time of occurrence → maps to location
- Why no ML: physics gives the answer directly; FFT/STFT already generalises perfectly
- Sensor: P5 (primary) + fixed phones (corroboration)

### Defect 2 — Inclination **[PRIMARY ML SHOWCASE]**
- **Approach: Both classification AND regression (compare explicitly)**
- Sensor: P5 only
- Detection window: handoff zone ~20–36s in P5 signal
- Physical mechanism:
  - Location 4 (end of C1): P5 drops from raised conveyor → detectable impact
  - Location 5 (start of C1): P5 lands onto raised section → different impact signature
- Features (physically motivated):
  - Peak z-acceleration in handoff window
  - Jerk magnitude (np.diff of acceleration) at impact
  - Impulse energy (integral of |acc|² over impact window, ~0.5s)
  - Rise time of impact transient
  - RMS of 0.5s window post-impact
- Models: Logistic Regression (baseline) → SVM → RF → GBM
- Targets:
  - Classification: {0: no incl, 1: incl present} then {0,1,2,3,4,5 degrees}
  - Regression: predict angle in degrees (0.0–5.0)

### Defect 3 — Belt Speed **[REGRESSION SHOWCASE]**
- **Approach: Multi-output regression**
- Sensor: P5 (primary)
- Physical mechanism: slower belt → longer journey duration; different belt speeds
  produce different low-frequency vibration signatures
- Features:
  - Total journey duration (strongest single feature)
  - RMS acceleration on C1 segment (Loc5→Loc4)
  - RMS acceleration on C2 segment (Loc3→Loc2)
  - Dominant low-frequency peak (<5 Hz) on each segment
  - RMS ratio C1/C2
- Models: Linear Regression (baseline) → RF Regressor → GBM (multi-output)
- Targets: Rail2_speed (%), Rail3_speed (%)

### Defect 4 — Damping **[ANOMALY DETECTION / IMBALANCED CLASSIFICATION SHOWCASE]**
- **Approach: Binary classification with explicit imbalance handling**
- Sensor: Fixed phones P1–P4 (P5 is NOT the right sensor here)
- Physical mechanism: sponge absorbs structural vibration through frame leg;
  fixed phones sense the change in transmitted vibration energy
- Features (per fixed phone):
  - RMS energy across full recording
  - RMS in low-frequency band (0–10 Hz)
  - RMS in mid-frequency band (10–50 Hz)
  - Peak frequency from FFT
  - Spatial energy gradient (ratio of nearest/furthest phone from sponge location)
- Models: Logistic Regression → SVM → RF → GBM
- Key challenge: only case_3 and case_6 have damping; must handle imbalance
  with class_weight='balanced' and evaluate with F1/AUC not accuracy

---

## ML Engineering Decisions **[LOCKED]**

### Train/Test Split Strategy
```
Train:  Groups G1–G7  (features extracted from all cases)
Test:   Groups G8–G9  (held out entirely, never seen during training or tuning)
CV:     Leave-One-Group-Out CV on G1–G7 only
        (each fold holds out one full group — prevents session-level data leakage)
```
**Why:** k-fold CV on pooled windowed data leaks information between windows from
the same recording session, making CV scores falsely optimistic. This was the primary
reason the original MATLAB model got 3% CV error but failed on the test set.

### Hyperparameter Tuning Strategy
```
Stage 1: RandomizedSearchCV (50 iterations, wide parameter space)
         → identifies the promising region efficiently
Stage 2: GridSearchCV (narrow grid around best Stage 1 result)
         → precision tuning
Both stages use Leave-One-Group-Out CV on training set only.
Test set is NEVER touched until final model evaluation.
```

### Model Scope
- scikit-learn only (no deep learning for this project)
- Models per problem: Logistic/Linear Regression → SVM → Random Forest → GBM
- Each model compared on same held-out test set, same metrics

### Evaluation Metrics
- Inclination classification: F1-macro, confusion matrix (not accuracy — imbalanced)
- Inclination regression: MAE (degrees), R²
- Belt speed regression: MAE per rail, R² per rail
- Damping classification: F1, AUC-ROC, precision-recall curve (severely imbalanced)

### Experiment Tracking
- MLflow for all model runs
- Log: all hyperparameters, all metrics, confusion matrix plots, feature importance plots
- Every run reproducible via logged random seeds

---

## Repository Structure **[LOCKED]**

```
imes-defect-detection/
├── data/
│   ├── raw/              ← original XLS files, NEVER modified
│   │   ├── G1/
│   │   ├── G2/ ...
│   │   └── G9/
│   └── processed/        ← extracted feature CSVs (generated by pipeline)
├── src/
│   ├── data_loader.py        ← XLS → pandas DataFrame, handles all devices
│   ├── preprocessor.py       ← resample, segment, landmark detection
│   ├── features/
│   │   ├── __init__.py
│   │   ├── frequency.py      ← FFT/STFT signal processing
│   │   ├── inclination.py    ← transient features from handoff window
│   │   ├── belt_speed.py     ← journey duration + low-freq features
│   │   └── damping.py        ← RMS energy features from fixed phones
│   └── models/
│       ├── __init__.py
│       ├── train.py          ← trains all models, logs to MLflow
│       ├── evaluate.py       ← metrics, plots, final test set evaluation
│       └── predict.py        ← inference on new files
├── notebooks/
│   ├── 01_eda.ipynb                  ← signal visualisation, landmark validation
│   ├── 02_feature_engineering.ipynb  ← feature distributions, correlation
│   ├── 03_model_experiments.ipynb    ← training runs, tuning story
│   └── 04_results_report.ipynb       ← final results, discussion
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessor.py
│   └── test_features.py
├── mlruns/                   ← MLflow experiment logs
├── PROJECT_CONTEXT.md        ← THIS FILE — read at session start
├── README.md                 ← written last, after real results
└── pyproject.toml            ← dependencies with pinned versions
```

---

## Coding Order (Build in This Sequence)

```
Step 1: data_loader.py + test_data_loader.py
        → can load any XLS file from any phone/group reliably

Step 2: preprocessor.py + test_preprocessor.py
        → resampling, landmark detection (arm pickup event), segmentation

Step 3: notebooks/01_eda.ipynb
        → visualise raw signals, validate segmentation on G7 data
        *** APPROVAL GATE: confirm segmentation is correct before features ***

Step 4: src/features/*.py + test_features.py
        → extract all features, save to data/processed/

Step 5: notebooks/02_feature_engineering.ipynb
        → feature distributions, correlation matrix, sanity checks
        *** APPROVAL GATE: confirm features make physical sense before models ***

Step 6: src/models/train.py + evaluate.py
        → full training pipeline with MLflow logging

Step 7: notebooks/03_model_experiments.ipynb
        → run experiments, tuning, comparison

Step 8: notebooks/04_results_report.ipynb + README.md
        → final write-up with real numbers
```

---

## Key Insights Worth Preserving

1. **Why the original model failed:** k-fold CV on pooled windowed data from the same
   sessions made CV error falsely low (3%). Leave-group-out CV is the correct approach.

2. **Why inclination RFC failed:** Mixed detectable (Loc 4) and non-detectable (Loc 5)
   cases in one model. Used raw signal windows instead of physically motivated transient
   features. Severe class imbalance masked by accuracy metric.

3. **Why P5 is wrong for damping:** Sponge is under the frame leg, not on the belt.
   P5 never physically interacts with the sponge. Fixed phones sense the structural
   attenuation. P5 damping ratios were near-zero and physically meaningless.

4. **Belt speed primary feature:** Journey duration is the single most informative
   feature (46s normal vs 71s slow). This should be feature #1 in the speed model.

5. **Inclination at Loc 5 is detectable:** P5 is deposited by Arm 1 onto a raised
   conveyor section. The landing impact signature encodes the angle differently than
   the mid-journey drop at Loc 4. Two sub-signatures, same feature extraction approach.
```

---

## How to Start a Claude Code Session

Paste this at the start of your first message in Claude Code:

> "Read PROJECT_CONTEXT.md in the project root. This is a portfolio ML project for
> defect detection in a conveyor system. All design decisions are documented there.
> We are currently at Step [X] of the coding order. Do not deviate from locked
> decisions without discussion. Let's begin."

Then tell it which step you're starting from.
