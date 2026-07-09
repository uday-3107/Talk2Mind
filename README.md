# Talk2Mind

A multimodal AI-based mental well-being screening system combining **facial expression recognition**, **speech emotion recognition**, and the clinically validated **PHQ-9 questionnaire** to estimate a user's distress score.

## Project Summary

| Metric | Value |
|---|---|
| **Language** | Python 3.11+ |
| **Total Python files** | 19 |
| **Total source lines** | 2,349 |
| **Directories** | 22 |
| **Trained models** | 2 (ResNet18 + BiLSTM) |
| **API endpoints** | 5 |
| **Dependencies** | 20 Python packages + ffmpeg |
| **Evaluation artifacts** | 10 files (reports + images) |

**How it works:** A user submits a webcam photo, a voice recording, and answers 9 PHQ-9 questions. The facial model (ResNet18, ~65% accuracy on FER-2013) predicts 6 emotion probabilities. The speech model (BiLSTM, ~42-44% on CREMA-D) does the same. The questionnaire is scored on a 0-27 scale (normalized to 0-100). A weighted fusion formula (facial 25%, speech 25%, questionnaire 50%) computes a final distress score with severity bands. A Groq LLM (llama-3.3-70b-versatile) rephrases questions conversationally and generates an empathetic summary. Session history enables trend tracking.

## System Flow

```
┌────────────────────────────────────────────────────────────────┐
│                    TALK2MIND ARCHITECTURE                       │
│                                                                 │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐  │
│  │  Webcam Photo   │   │  Voice Recording  │   │  PHQ-9 Qs   │  │
│  └────────┬────────┘   └────────┬─────────┘   └──────┬──────┘  │
│           │                     │                     │         │
│           ▼                     ▼                     ▼         │
│  ┌─────────────────┐   ┌──────────────────┐   ┌─────────────┐  │
│  │  Facial Model   │   │  Speech Model    │   │  PHQ-9      │  │
│  │  (ResNet18)     │   │  (BiLSTM)        │   │  Scoring    │  │
│  │  ~65% accuracy  │   │  ~42-44% acc     │   │  0-100 norm │  │
│  │  6 emotion probs│   │  6 emotion probs │   │  risk flag  │  │
│  └────────┬────────┘   └────────┬─────────┘   └──────┬──────┘  │
│           │                     │                     │         │
│           └─────────────────────┼─────────────────────┘         │
│                                 │                               │
│                                 ▼                               │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Fusion Layer (weighted scoring)             │   │
│  │   weights: facial=0.25, speech=0.25, questionnaire=0.50 │   │
│  │   distress weights per emotion (angry→0.75, happy→0.0)  │   │
│  │   output: final_score (0-100) + severity_band           │   │
│  └─────────────────────────┬───────────────────────────────┘   │
│                            │                                     │
│                            ▼                                     │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Output Layer                                │   │
│  │   ┌─────────────┐  ┌─────────────┐  ┌──────────────┐   │   │
│  │   │ AI Summary  │  │ Recs by    │  │ Session      │   │   │
│  │   │ (Groq LLM)  │  │ band +     │  │ History +    │   │   │
│  │   │ empathetic  │──│ crisis     │──│ Trend Track  │   │   │
│  │   │ 3-4 sent.  │  │ alert      │  │ improving/   │   │   │
│  │   └─────────────┘  └─────────────┘  │ worsening   │   │   │
│  │                                     └──────────────┘   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Delivery: Streamlit Dashboard  │  FastAPI Backend  │  Docker  │
│            (3 pages)            │  (5 endpoints)    │  compose │
└────────────────────────────────────────────────────────────────┘
```

## Tech Stack

- **Python 3** — core language
- **PyTorch / torchvision** — model training (ResNet18, BiLSTM)
- **librosa** — audio feature extraction (MFCC, delta-MFCC, RMS, pitch)
- **scikit-learn** — evaluation metrics, train/test splits
- **Streamlit** — interactive dashboard (3 pages)
- **FastAPI + uvicorn** — REST API backend
- **Groq API (llama-3.3-70b-versatile)** — conversational question rephrasing, empathetic summary generation
- **OpenCV + MediaPipe** — face detection, landmark localization
- **pytorch-grad-cam** — Grad-CAM heatmap visualization
- **ffmpeg** — audio format conversion (WAV, MP3, M4A, OGG, FLAC → pipeline)
- **python-dotenv** — API key management
- **Docker** — containerized deployment

## Project Structure

```
talk2mind/
├── api/
│   ├── main.py                     # FastAPI backend (5 endpoints)
│   └── schemas.py                  # Pydantic request/response models
├── app/
│   ├── dashboard.py                # Streamlit dashboard (with Grad-CAM)
│   └── live_pipeline.py            # Real-time webcam + mic inference demo
├── src/
│   ├── facial/
│   │   ├── preprocess_facial.py    # FER-2013 preprocessing + transforms
│   │   └── train_facial.py         # ResNet18 fine-tuning
│   ├── speech/
│   │   ├── preprocess_speech.py    # Feature extraction from audio
│   │   └── train_speech.py         # BiLSTM training
│   ├── questionnaire/
│   │   ├── questions.json          # PHQ-9 items
│   │   └── scoring.py              # PHQ-9 scoring + risk flagging
│   ├── fusion/
│   │   ├── feature_extraction.py   # Embedding + probability extraction
│   │   └── fusion_scoring.py       # Weighted multimodal fusion
│   ├── recommendation/
│   │   ├── recommend.py            # Groq API calls + static recommendations
│   │   └── session_history.py      # Per-user history + trend detection
│   └── utils/
│       ├── build_labels.py         # CREMA-D label generation
│       └── split_speech.py         # Actor-disjoint train/val/test split
├── evaluation/
│   ├── metrics.py                  # Classification report + confusion matrix
│   ├── calibration_analysis.py     # Score distribution + band analysis
│   └── gradcam.py                  # Grad-CAM heatmap generator
├── data/                           # Datasets (gitignored)
│   ├── raw/                        # FER-2013 + CREMA-D
│   └── processed/                  # Speech features, train/val/test CSVs
├── models/                         # Trained checkpoints (gitignored)
│   ├── facial_model.pt
│   └── speech_model.pt
├── Dockerfile                      # Container build
├── docker-compose.yml              # Multi-service orchestration
├── .dockerignore
├── .gitignore
├── .env                            # GROQ_API_KEY (gitignored)
├── requirements.txt
└── README.md
```

## Setup

### 1. Prerequisites

- Python 3.11+ (tested on 3.14)
- pip
- ffmpeg (for audio format conversion — `brew install ffmpeg` on macOS)
- (optional) Docker for containerized deployment

### 2. Clone and install

```bash
git clone https://github.com/yourusername/talk2mind.git
cd talk2mind
pip install -r requirements.txt
```

### 4. Set up Groq API key (optional)

```bash
echo "GROQ_API_KEY=your_key_here" > .env
```

The dashboard falls back to static text if this key is not set or the API call fails.

### 5. Download datasets (for training only)

- **Facial**: [FER-2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013) — place in `data/raw/facial/train/` and `data/raw/facial/test/`
- **Speech**: [CREMA-D on Kaggle](https://www.kaggle.com/datasets/ejlok1/cremad) — place `.wav` files in `data/raw/speech/audio/`

### 6. Train models (skip if using pre-trained checkpoints)

```bash
python src/utils/build_labels.py
python src/utils/split_speech.py
python src/speech/preprocess_speech.py
python src/speech/train_speech.py
python src/facial/train_facial.py
```

## Running

### Streamlit Dashboard

```bash
streamlit run app/dashboard.py
```

Flow: enter a session ID → answer 9 PHQ-9 questions (one at a time with conversational phrasing) → take a webcam photo → upload a voice recording (WAV, MP3, M4A, OGG, FLAC) → view score, breakdown, Grad-CAM heatmap, AI summary, recommendations, and trend.

### FastAPI Backend

```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

Browse to `http://localhost:8000/docs` for interactive Swagger UI.

**Endpoints:**

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | API info and available routes |
| `POST` | `/assess` | Full pipeline (image + audio + responses → score + recs + summary) |
| `POST` | `/session/{user_id}/save` | Save a session result with feedback |
| `GET` | `/session/{user_id}/trend` | Get trend direction and delta |
| `GET` | `/session/{user_id}/history` | Get all past sessions as JSON |

### Live Real-Time Pipeline

```bash
python app/live_pipeline.py
```

Runs webcam face detection + continuous mic recording, displaying predicted emotions from both models in real time.

## Evaluation & Analysis

### Model Metrics

```bash
python evaluation/metrics.py
```

Generates classification reports and confusion matrices, saved to `evaluation/results/`.

### Score Calibration Analysis

```bash
python evaluation/calibration_analysis.py
```

Plots score distribution histogram, prints band breakdown, and flags underrepresented bands (<5%) suggesting fusion weight rebalancing.

### Grad-CAM Visualizations

```bash
python evaluation/gradcam.py
```

Generates heatmap overlays for sample test images showing which facial regions drove each emotion prediction. Outputs saved to `evaluation/results/gradcam_examples/`.

Grad-CAM is also displayed live in the Streamlit dashboard on your own uploaded photo.

## Docker

### Build and run

```bash
docker compose build
docker compose up
```

### Individual services

```bash
docker compose up streamlit   # Dashboard on http://localhost:8501
docker compose up api         # API on http://localhost:8000
```

Models are mounted as a bind volume (not baked into the image) to keep image size manageable. The `.env` file is injected at runtime.

## Model Performance

### Facial Emotion Recognition (ResNet18)

- **Dataset**: FER-2013 (6 classes, surprise dropped)
- **Test accuracy**: ~65%
- **Architecture**: ImageNet-pretrained ResNet18, layer4 + FC head unfrozen, dropout 0.5
- **Strengths**: happy, disgust
- **Weaknesses**: angry/fear/neutral confusion (known FER-2013 challenge)

### Speech Emotion Recognition (BiLSTM)

- **Dataset**: CREMA-D (7,442 clips, 6 classes)
- **Test accuracy**: ~42-44%
- **Architecture**: 2-layer bidirectional LSTM (hidden_dim=128), input: 28-dim features × 200 timesteps
- **Training details**: Actor-disjoint split (63/14/14 actors), Focal Loss (gamma=2.0), class-weighted
- **Strengths**: angry, sad
- **Weaknesses**: disgust, fear (modest recall)

## Limitations

- **Speech model accuracy** is modest (42-44%) — built from scratch in one week, not benchmark-competitive. Additional data and architecture tuning would improve it.
- **Fusion weights are hand-designed** — facial 25%, speech 25%, questionnaire 50%. A learned fusion model would require a paired multimodal dataset (same subject providing face, voice, and ground-truth label), which does not currently exist in this project.
- **Personalization is rule-based trend tracking**, not a trained ML model. The dashboard collects user feedback (mood_change, accuracy_rating) to enable future ML-driven personalization once sufficient labeled data is accumulated.
- **Groq API integration** is a network dependency. Both conversational features (question rephrasing, empathetic summary) have fallback logic so the dashboard never breaks when the API is unavailable.

## Future Work

- Real-time dashboard (WebSocket-based instead of Streamlit's request-response model)
- Database backend (PostgreSQL) for persistent user history across sessions
- Training a learned fusion model once paired multimodal data is available
- Containerized deployment (Docker) to Hugging Face Spaces or Cloud Run
- Transformer-based speech model (HuBERT / wav2vec 2.0) for improved SER accuracy
- ML-based personalization using accumulated session feedback data

## Safety Notice

This tool is an experimental screening prototype and **not a clinical diagnostic device**. It does not replace professional mental health care. The PHQ-9 questionnaire item 9 (self-harm ideation) triggers a mandatory crisis alert with contact information for iCall (India) regardless of the overall score. If you or someone you know is in crisis, contact a local emergency number or crisis helpline immediately.