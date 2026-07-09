import os
from collections import Counter
import numpy as np
from torchvision import transforms, datasets
import torch

DATA_DIR = "data/raw/facial"
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "neutral"]  # surprise dropped

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])

test_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225]),
])


def get_dataset(split):
    """split = 'train' or 'test'"""
    full_dir = os.path.join(DATA_DIR, split)
    t = train_transform if split == "train" else test_transform
    dataset = datasets.ImageFolder(full_dir, transform=t)
    keep_idx = [i for i, (_, label) in enumerate(dataset.samples)
                if dataset.classes[label] in CLASSES]
    dataset.samples = [dataset.samples[i] for i in keep_idx]
    dataset.targets = [dataset.targets[i] for i in keep_idx]
    return dataset


def compute_class_weights(dataset):
    counts = Counter(dataset.targets)
    total = sum(counts.values())
    weights = [total / counts[i] for i in range(len(counts))]
    return torch.tensor(weights, dtype=torch.float)


if __name__ == "__main__":
    train_ds = get_dataset("train")
    print(f"Train samples: {len(train_ds)}")
    print(f"Class counts: {Counter(train_ds.targets)}")
    weights = compute_class_weights(train_ds)
    print(f"Class weights: {weights}")

