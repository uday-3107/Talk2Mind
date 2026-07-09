import os
import csv

AUDIO_DIR = "data/raw/speech/audio"
OUTPUT_CSV = "data/raw/speech/labels.csv"

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

rows = []
for fname in os.listdir(AUDIO_DIR):
    if not fname.endswith(".wav"):
        continue
    parts = fname.split("_")
    code = parts[2]  # e.g. 1001_IWW_DIS_XX.wav -> DIS
    emotion = EMOTION_MAP.get(code)
    if emotion is None:
        print(f"Skipping unrecognized code: {fname}")
        continue
    rows.append([os.path.join(AUDIO_DIR, fname), emotion])

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filepath", "emotion"])
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
