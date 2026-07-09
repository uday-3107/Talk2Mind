import json
import os
import sys
import tempfile
import base64
from contextlib import asynccontextmanager
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from api.schemas import AssessResponse, SaveSessionRequest, TrendResponse
from src.fusion.feature_extraction import (
    DEVICE,
    build_facial_embedder,
    build_speech_embedder,
    facial_transform,
    get_facial_embedding,
    get_speech_embedding,
)
from src.fusion.fusion_scoring import compute_mental_health_score
from src.questionnaire.scoring import score_responses
from src.recommendation.recommend import (
    generate_empathetic_summary,
    get_recommendations,
)
from src.recommendation.session_history import (
    get_history_dataframe,
    get_trend,
    save_session,
)
from src.speech.preprocess_speech import extract_features

facial_model = None
speech_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global facial_model, speech_model
    facial_model = build_facial_embedder()
    speech_model = build_speech_embedder()
    yield


app = FastAPI(title="Talk2Mind API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {
        "app": "Talk2Mind API",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": [
            "POST /assess",
            "POST /session/{user_id}/save",
            "GET /session/{user_id}/trend",
            "GET /session/{user_id}/history",
        ],
    }


def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(400, "Could not decode image — invalid or corrupted file.")
    return frame


def process_facial(image_bytes: bytes) -> np.ndarray:
    frame = decode_image(image_bytes)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    tensor = facial_transform(frame_rgb).unsqueeze(0).to(DEVICE)
    _, probs, _ = get_facial_embedding(facial_model, tensor)
    return probs.flatten()


def process_audio(audio_bytes: bytes) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        try:
            tmp.write(audio_bytes)
            tmp.flush()
            feat = extract_features(tmp_path)
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    _, probs, _ = get_speech_embedding(speech_model, tensor)
    return probs.flatten()


@app.post("/assess", response_model=AssessResponse)
async def assess(
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None),
    image_base64: Optional[str] = Form(None),
    audio_base64: Optional[str] = Form(None),
    responses: str = Form(...),
):
    if image is None and image_base64 is None:
        raise HTTPException(400, "Provide either 'image' (file upload) or 'image_base64'.")
    if audio is None and audio_base64 is None:
        raise HTTPException(400, "Provide either 'audio' (file upload) or 'audio_base64'.")

    try:
        responses_dict = json.loads(responses)
    except json.JSONDecodeError:
        raise HTTPException(400, "'responses' must be a valid JSON object.")
    if not isinstance(responses_dict, dict):
        raise HTTPException(400, "'responses' must be a JSON object.")

    if image is not None:
        image_bytes = await image.read()
    else:
        try:
            image_bytes = base64.b64decode(image_base64)
        except Exception:
            raise HTTPException(400, "image_base64 is not valid base64.")

    if audio is not None:
        audio_bytes = await audio.read()
    else:
        try:
            audio_bytes = base64.b64decode(audio_base64)
        except Exception:
            raise HTTPException(400, "audio_base64 is not valid base64.")

    facial_probs = process_facial(image_bytes)
    speech_probs = process_audio(audio_bytes)
    q_result = score_responses(responses_dict)
    score_result = compute_mental_health_score(
        facial_probs, speech_probs, q_result["normalized_score"]
    )
    recs = get_recommendations(score_result["band"], risk_flag=q_result["risk_flag"])
    summary = generate_empathetic_summary(score_result, recs)

    recommendations = [r for r in recs["recommendations"] if r != recs.get("crisis_alert")]

    return AssessResponse(
        final_score=score_result["final_score"],
        band=score_result["band"],
        breakdown=score_result["breakdown"],
        recommendations=recommendations,
        crisis_alert=recs.get("crisis_alert"),
        empathetic_summary=summary,
    )


@app.post("/session/{user_id}/save")
async def save_session_endpoint(user_id: str, body: SaveSessionRequest):
    try:
        history = save_session(user_id, body.score_result, body.feedback)
    except Exception as e:
        raise HTTPException(500, f"Failed to save session: {e}")
    return {"status": "ok", "sessions_count": len(history)}


@app.get("/session/{user_id}/trend", response_model=TrendResponse)
async def trend_endpoint(user_id: str):
    try:
        trend = get_trend(user_id)
    except Exception as e:
        raise HTTPException(500, f"Failed to get trend: {e}")
    return TrendResponse(**trend)


@app.get("/session/{user_id}/history")
async def history_endpoint(user_id: str):
    try:
        df = get_history_dataframe(user_id)
        if df.empty:
            return []
        records = df.to_dict(orient="records")
        for r in records:
            r["timestamp"] = r["timestamp"].isoformat()
        return records
    except Exception as e:
        raise HTTPException(500, f"Failed to get history: {e}")


if __name__ == "__main__":
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
