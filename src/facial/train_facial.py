import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from facial.preprocess_facial import get_dataset, compute_class_weights, CLASSES

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
EPOCHS = 15
LR = 1e-4
MODEL_SAVE_PATH = "models/facial_model.pt"


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer4.parameters():
        param.requires_grad = True
    model.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(model.fc.in_features, num_classes)
    )
    return model.to(DEVICE)


def train():
    train_ds = get_dataset("train")
    test_ds = get_dataset("test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=1)

    class_weights = compute_class_weights(train_ds).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    model = build_model(len(CLASSES))
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=1e-4
    )
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        total_loss, correct, total = 0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f"Epoch {epoch+1}/{EPOCHS} - loss: {total_loss/total:.4f} - train_acc: {train_acc:.4f}")

        # quick validation pass
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in test_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
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
    print(f"Using device: {DEVICE}")
    train()