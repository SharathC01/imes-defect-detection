from src.data_loader import load_file
from src.preprocessor import resample, detect_pickup_event, segment_journey
from src.features.belt_speed import extract_belt_speed_features

# Load G0 baseline first
df_g0, fs_g0 = load_file('data/raw/G0/G0_P5_case_perfect.xls')
df_g0 = resample(df_g0, fs_original=fs_g0, fs_target=100.0)
pickup_g0 = detect_pickup_event(df_g0)
segments_g0 = segment_journey(df_g0, pickup_g0)

# Case74 — belt speed defect (Rail2=30, Rail3=30, both slow)
df74, fs74 = load_file('data/raw/G7/G7_P5_case74.xls')
df74 = resample(df74, fs_original=fs74, fs_target=100.0)
pickup74 = detect_pickup_event(df74)
segments74 = segment_journey(df74, pickup74)

features74 = extract_belt_speed_features(segments74['journey'], segments_g0['journey'])
print("--- Belt Speed Features (case74, Rail2=30 Rail3=30) ---")
for k, v in features74.items():
    print(f"  {k}: {v}")

print()

# Case71 — normal speed (Rail2=70, Rail3=100)
df71, fs71 = load_file('data/raw/G7/G7_P5_case71.xls')
df71 = resample(df71, fs_original=fs71, fs_target=100.0)
pickup71 = detect_pickup_event(df71)
segments71 = segment_journey(df71, pickup71)

features71 = extract_belt_speed_features(segments71['journey'], segments_g0['journey'])
print("--- Belt Speed Features (case71, normal speed) ---")
for k, v in features71.items():
    print(f"  {k}: {v}")