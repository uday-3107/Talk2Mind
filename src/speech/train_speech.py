import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from collections import Counter
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
FEATURES_DIR = "data/processed/speech_features"
BATCH_SIZE = 32
EPOCHS = 25
LR = 5e-4
MODEL_SAVE_PATH = "models/speech_model.pt"
NUM_CLASSES = 6


class SpeechDataset(Dataset):
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        feat = np.load(os.path.join(FEATURES_DIR, f"{row['idx']}.npy"))
        # per-feature-column standardization (z-score), avoid div by zero
        mean = feat.mean(axis=0, keepdims=True)
        std = feat.std(axis=0, keepdims=True) + 1e-6
        feat = (feat - mean) / std
        if feat.shape[1] != 28:
            raise ValueError(f"Expected 28 features, got {feat.shape[1]} — re-run preprocess_speech.py")
        return torch.tensor(feat, dtype=torch.float32), int(row["label"])


class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)          # (batch, seq_len, hidden*2)
        out = out[:, -1, :]             # last timestep
        out = self.dropout(out)
        return self.fc(out)


def compute_class_weights(df):
    counts = Counter(df["label"])
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(NUM_CLASSES)]
    return torch.tensor(weights, dtype=torch.float)


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, weight=self.alpha, reduction='none'
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()


def train():
    train_ds = SpeechDataset("data/processed/speech_train.csv")
    val_ds = SpeechDataset("data/processed/speech_val.csv")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    class_weights = compute_class_weights(train_ds.df).to(DEVICE)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0)

    model = BiLSTMClassifier().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * feats.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/total:.4f} - train_acc: {train_acc:.4f}")

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for feats, labels in val_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                outputs = model(feats)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        val_acc = val_correct / val_total
        print(f"  val_acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs("models", exist_ok=True)
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  New best val_acc {val_acc:.4f} — checkpoint saved")

    print(f"Training done. Best val_acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    train()