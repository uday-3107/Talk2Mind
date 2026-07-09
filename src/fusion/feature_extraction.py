import torch
import torch.nn as nn
from torchvision import models, transforms

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

facial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


# ---------- Facial embedding extractor ----------
def build_facial_embedder():
    """Loads trained ResNet18, strips classifier to output 512-d embedding + softmax probs."""
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, 6))
    model.load_state_dict(torch.load("models/facial_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def get_facial_embedding(model, image_tensor):
    """
    image_tensor: preprocessed (1, 3, 224, 224) tensor.
    Returns: (embedding_512, class_probs_6, predicted_idx)
    """
    with torch.no_grad():
        # forward through everything except final fc to get 512-d embedding
        x = model.conv1(image_tensor)
        x = model.bn1(x); x = model.relu(x); x = model.maxpool(x)
        x = model.layer1(x); x = model.layer2(x)
        x = model.layer3(x); x = model.layer4(x)
        x = model.avgpool(x)
        embedding = torch.flatten(x, 1)  # (1, 512)

        logits = model.fc(embedding)
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()

    return embedding.cpu().numpy(), probs.cpu().numpy(), idx


# ---------- Speech embedding extractor ----------
class BiLSTMClassifier(nn.Module):
    def __init__(self, input_dim=28, hidden_dim=128, num_classes=6):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True,
                             bidirectional=True, num_layers=2, dropout=0.3)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward_embedding(self, x):
        out, _ = self.lstm(x)
        return out[:, -1, :]  # (batch, hidden*2) = 256-d embedding

    def forward(self, x):
        emb = self.forward_embedding(x)
        return self.fc(self.dropout(emb))


def build_speech_embedder():
    model = BiLSTMClassifier()
    model.load_state_dict(torch.load("models/speech_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


def get_speech_embedding(model, feature_tensor):
    """
    feature_tensor: preprocessed (1, 200, 28) tensor, z-score normalized.
    Returns: (embedding_256, class_probs_6, predicted_idx)
    """
    with torch.no_grad():
        embedding = model.forward_embedding(feature_tensor)
        logits = model.fc(embedding)
        probs = torch.softmax(logits, dim=1)
        idx = probs.argmax(dim=1).item()

    return embedding.cpu().numpy(), probs.cpu().numpy(), idx


# ---------- Combined multimodal feature vector ----------
def build_multimodal_vector(facial_probs, speech_probs, questionnaire_normalized_score):
    """
    Combines both modality class-probability vectors + questionnaire score
    into one flat feature vector for the fusion/scoring stage.
    facial_probs, speech_probs: (6,) arrays. questionnaire_normalized_score: 0-100 float.
    Output: (13,) vector -> [6 facial probs, 6 speech probs, 1 questionnaire score/100]
    """
    import numpy as np
    return np.concatenate([
        facial_probs.flatten(),
        speech_probs.flatten(),
        [questionnaire_normalized_score / 100.0]
    ])