import cv2
import torch
import torch.nn as nn
import numpy as np
import sounddevice as sd
import librosa
from torchvision import models, transforms
from collections import deque
import threading
import time

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
# Both lists must match the class indices used during model training
FACIAL_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "neutral"]
SPEECH_CLASSES = ["angry", "disgust", "fear", "happy", "sad", "neutral"]
AUDIO_SR = 16000
AUDIO_WINDOW_SEC = 2.5
MAX_FRAMES = 200

# ---------- Facial model ----------
def build_facial_model():
    model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(model.fc.in_features, len(FACIAL_CLASSES)))
    model.load_state_dict(torch.load("models/facial_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

facial_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---------- Speech model ----------
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

def build_speech_model():
    model = BiLSTMClassifier()
    model.load_state_dict(torch.load("models/speech_model.pt", map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def extract_speech_features(y, sr=AUDIO_SR):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfcc)
    rms = librosa.feature.rms(y=y)
    pitch, _, _ = librosa.pyin(y, fmin=50, fmax=500)
    pitch = np.nan_to_num(pitch).reshape(1, -1)

    min_t = min(mfcc.shape[1], mfcc_delta.shape[1], rms.shape[1], pitch.shape[1])
    mfcc, mfcc_delta = mfcc[:, :min_t], mfcc_delta[:, :min_t]
    rms, pitch = rms[:, :min_t], pitch[:, :min_t]
    feat = np.vstack([mfcc, mfcc_delta, rms, pitch]).T  # (T, 28)

    if feat.shape[0] < MAX_FRAMES:
        pad = np.zeros((MAX_FRAMES - feat.shape[0], feat.shape[1]))
        feat = np.vstack([feat, pad])
    else:
        feat = feat[:MAX_FRAMES]

    mean = feat.mean(axis=0, keepdims=True)
    std = feat.std(axis=0, keepdims=True) + 1e-6
    return (feat - mean) / std

# ---------- Shared audio buffer ----------
audio_buffer = deque(maxlen=int(AUDIO_SR * AUDIO_WINDOW_SEC))
audio_lock = threading.Lock()

def audio_callback(indata, frames, time_info, status):
    with audio_lock:
        audio_buffer.extend(indata[:, 0])

# ---------- Prediction smoothing ----------
facial_history = deque(maxlen=8)
speech_history = deque(maxlen=5)

def smoothed_label(history, new_label):
    history.append(new_label)
    return max(set(history), key=history.count)

def main():
    facial_model = build_facial_model()
    speech_model = build_speech_model()

    stream = sd.InputStream(samplerate=AUDIO_SR, channels=1, callback=audio_callback)
    stream.start()

    cap = cv2.VideoCapture(0)
    last_speech_time = time.time()
    speech_label, speech_conf = "...", 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        facial_label, facial_conf = "no face", 0.0
        for (x, y, w, h) in faces:
            face_crop = frame[y:y+h, x:x+w]
            face_crop_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
            tensor = facial_transform(face_crop_rgb).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                out = facial_model(tensor)
                probs = torch.softmax(out, dim=1)[0]
                idx = probs.argmax().item()
                facial_label = smoothed_label(facial_history, FACIAL_CLASSES[idx])
                facial_conf = probs[idx].item()
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            break  # first face only

        # run speech inference every ~2 seconds, not every frame
        if time.time() - last_speech_time > 2.0:
            with audio_lock:
                if len(audio_buffer) >= AUDIO_SR:  # at least 1 sec of audio
                    y_audio = np.array(audio_buffer, dtype=np.float32)
                else:
                    y_audio = None
            if y_audio is not None:
                feat = extract_speech_features(y_audio)
                tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = speech_model(tensor)
                    probs = torch.softmax(out, dim=1)[0]
                    idx = probs.argmax().item()
                    speech_label = smoothed_label(speech_history, SPEECH_CLASSES[idx])
                    speech_conf = probs[idx].item()
            last_speech_time = time.time()

        cv2.putText(frame, f"Facial: {facial_label} ({facial_conf:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Speech: {speech_label} ({speech_conf:.2f})", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

        cv2.imshow("Talk2Mind - Live", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    stream.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()