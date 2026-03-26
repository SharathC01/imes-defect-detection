from src.data_loader import load_file
from src.preprocessor import resample, detect_pickup_event, segment_journey
from src.features.frequency import extract_dominant_frequency, locate_frequency_in_time

df, fs = load_file('data/raw/G7/G7_P5_case71.xls')
df = resample(df, fs_original=fs, fs_target=100.0)
pickup = detect_pickup_event(df)
segments = segment_journey(df, pickup)

freq_features = extract_dominant_frequency(segments['journey'])
print("--- Frequency Features ---")
for k, v in freq_features.items():
    print(f"  {k}: {v}")

loc_features = locate_frequency_in_time(segments['journey'], 
                                         freq_features['dominant_freq_hz'])
print("\n--- Location Features ---")
for k, v in loc_features.items():
    print(f"  {k}: {v}")