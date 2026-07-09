import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import os
import sys
from sklearn.metrics import classification_report, confusion_matrix
from torchvision import models
from torch.utils.data import DataLoader, Dataset

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.facial.preprocess_facial import get_dataset, CLASSES as FACIAL_CLASSES

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def evaluate_facial():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, len(FACIAL_CLASSES)))
    model.load_state_dict(torch.load("models/facial_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    test_ds = get_dataset("test")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=FACIAL_CLASSES, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    with open(os.path.join(RESULTS_DIR, "facial_report.txt"), "w") as f:
        f.write("FACIAL MODEL EVALUATION\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(f"Classes: {FACIAL_CLASSES}\n")
        f.write(str(cm))

    print("=== FACIAL ===")
    print(report)
    print(cm)


class SpeechDataset(Dataset):
    def __init__(self, csv_path, features_dir="data/processed/speech_features"):
        self.df = pd.read_csv(csv_path)
        self.features_dir = features_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        feat = np.load(os.path.join(self.features_dir, f"{row['idx']}.npy"))
        mean = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True) + 1e-6
        if feat.shape[1] != 28:
            raise ValueError(f"Expected 28 features, got {feat.shape[1]} — re-run preprocess_speech.py")
        feat = (feat - mean) / std
        return torch.tensor(feat, dtype=torch.float32), int(row["label"])


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(self.dropout(out))


SPEECH_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]


def evaluate_speech():
    model = BiLSTMClassifier()
    model.load_state_dict(torch.load("models/speech_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()

    test_ds = SpeechDataset("data/processed/speech_test.csv")
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for feats, labels in loader:
            feats = feats.to(DEVICE)
            out = model(feats)
            preds = out.argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    report = classification_report(all_labels, all_preds, target_names=SPEECH_CLASSES, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    with open(os.path.join(RESULTS_DIR, "speech_report.txt"), "w") as f:
        f.write("SPEECH MODEL EVALUATION\n\n")
        f.write(report)
        f.write("\n\nConfusion Matrix (rows=true, cols=pred):\n")
        f.write(f"Classes: {SPEECH_CLASSES}\n")
        f.write(str(cm))

    print("=== SPEECH ===")
    print(report)
    print(cm)


if __name__ == "__main__":
    evaluate_facial()
    evaluate_speech()
