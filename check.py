from src.data_loader import load_file
from src.preprocessor import resample, detect_pickup_event, segment_journey
from src.features.inclination import extract_inclination_features

cases = {
    '71 (freq only, no incl)':      'data/raw/G7/G7_P5_case71.xls',
    '72 (4deg Loc4, incl)':         'data/raw/G7/G7_P5_case72.xls',
    '73 (damping only, no incl)':   'data/raw/G7/G7_P5_case73.xls',
    '75 (freq+incl, 4deg Loc4)':    'data/raw/G7/G7_P5_case75.xls',
    '76 (freq+damping, no incl)':   'data/raw/G7/G7_P5_case76.xls',
}

print(f"{'Case':<35} {'signed_peak_z':>15} {'peak_z':>10} {'peak_jerk':>12}")
print("-" * 75)

for label, path in cases.items():
    df, fs = load_file(path)
    df = resample(df, fs_original=fs, fs_target=100.0)
    pickup = detect_pickup_event(df)
    segments = segment_journey(df, pickup)
    feats = extract_inclination_features(segments['inclination'])
    print(f"{label:<35} {feats['signed_peak_z']:>15.3f} "
          f"{feats['peak_z']:>10.3f} {feats['peak_jerk']:>12.3f}")