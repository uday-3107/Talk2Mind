import librosa
import numpy as np
import pandas as pd
import os

LABELS_CSV = "data/raw/speech/labels.csv"
OUTPUT_DIR = "data/processed/speech_features"
MAX_FRAMES = 200  # pad/truncate for fixed-length BiLSTM input

os.makedirs(OUTPUT_DIR, exist_ok=True)


def extract_features(filepath):
    y, sr = librosa.load(filepath, sr=16000)

    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)          # (13, T)
    if mfcc.shape[1] < 9:
        mfcc = np.pad(mfcc, ((0, 0), (0, 9 - mfcc.shape[1])), mode="constant")
    mfcc_delta = librosa.feature.delta(mfcc)                     # (13, T)
    rms = librosa.feature.rms(y=y)                               # (1, T)
    pitch, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    pitch = np.nan_to_num(pitch).reshape(1, -1)                  # (1, T)

    min_t = min(mfcc.shape[1], mfcc_delta.shape[1], rms.shape[1], pitch.shape[1])
    mfcc, mfcc_delta = mfcc[:, :min_t], mfcc_delta[:, :min_t]
    rms, pitch = rms[:, :min_t], pitch[:, :min_t]

    features = np.vstack([mfcc, mfcc_delta, rms, pitch])  # (28, T)
    features = features.T  # (T, 28)

    # pad/truncate to MAX_FRAMES
    if features.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - features.shape[0], features.shape[1]))
        features = np.vstack([features, pad])
    else:
        features = features[:MAX_FRAMES]

    return features  # (MAX_FRAMES, 28)


if __name__ == "__main__":
    df = pd.read_csv(LABELS_CSV)
    failed = []
    for idx, row in df.iterrows():
        try:
            feat = extract_features(row["filepath"])
        except Exception as e:
            print(f"Skipping {row['filepath']}: {e}")
            failed.append(row["filepath"])
            continue
        out_path = os.path.join(OUTPUT_DIR, f"{idx}.npy")
        np.save(out_path, feat)
        if idx % 500 == 0:
            print(f"Processed {idx}/{len(df)}")

    if failed:
        with open("data/raw/speech/failed_files.txt", "w") as f:
            f.write("\n".join(failed))
        print(f"Done. {len(failed)} files failed, see failed_files.txt")
    else:
        print("Done. No failures.")