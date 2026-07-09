import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

LABELS_CSV = "data/raw/speech/labels.csv"
OUTPUT_DIR = "data/processed/speech_features"

EMOTION_TO_IDX = {
    "angry": 0, "disgust": 1, "fear": 2,
    "happy": 3, "neutral": 4, "sad": 5,
}

df = pd.read_csv(LABELS_CSV)
df["idx"] = df.index
df["label"] = df["emotion"].map(EMOTION_TO_IDX)

# Extract actor ID from filename: {actorID}_{sentence}_{emotion}_{intensity}.wav
df["actor"] = df["filepath"].apply(lambda p: p.split("/")[-1].split("_")[0])

# Split by actor, not by file — no actor leaks between splits
actors = df["actor"].unique()
train_actors, temp_actors = train_test_split(
    actors, test_size=0.30, random_state=42
)
val_actors, test_actors = train_test_split(
    temp_actors, test_size=0.50, random_state=42
)

train_df = df[df["actor"].isin(train_actors)]
val_df = df[df["actor"].isin(val_actors)]
test_df = df[df["actor"].isin(test_actors)]

train_df.to_csv("data/processed/speech_train.csv", index=False)
val_df.to_csv("data/processed/speech_val.csv", index=False)
test_df.to_csv("data/processed/speech_test.csv", index=False)

for name, split_df, split_actors in [
    ("Train", train_df, train_actors),
    ("Val", val_df, val_actors),
    ("Test", test_df, test_actors),
]:
    print(f"{name}: {len(split_df)} files, {len(split_actors)} actors")
    print(f"  Label distribution: {split_df['label'].value_counts().sort_index().to_dict()}")
