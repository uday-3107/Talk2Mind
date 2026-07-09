import numpy as np

# NOTE: Order must match the facial model's class order from preprocess_facial.py
# Speech model uses the same 6 classes but was trained with the same indexing via labels.csv
CLASSES = ["angry", "disgust", "fear", "happy", "sad", "neutral"]


# distress weight per emotion: how much each detected emotion contributes
# to a negative mental well-being signal (0 = no distress, 1 = high distress)
DISTRESS_WEIGHTS = {
    "angry": 0.75,
    "disgust": 0.60,
    "fear": 0.80,
    "happy": 0.0,
    "sad": 0.85,
    "neutral": 0.25,
}

# overall modality weights for the final score
# questionnaire is clinically validated (PHQ-9), given more trust than
# emotion-detection models built this week with limited accuracy
MODALITY_WEIGHTS = {
    "facial": 0.25,
    "speech": 0.25,
    "questionnaire": 0.50,
}


def emotion_distress_score(class_probs: np.ndarray) -> float:
    """
    class_probs: (6,) array of softmax probabilities matching CLASSES order.
    Returns a 0-100 distress score for this modality.
    """
    weights = np.array([DISTRESS_WEIGHTS[c] for c in CLASSES])
    score = np.dot(class_probs, weights)  # weighted average, 0-1
    return score * 100


def compute_mental_health_score(facial_probs, speech_probs, questionnaire_normalized_score):
    """
    facial_probs, speech_probs: (6,) softmax arrays from each model.
    questionnaire_normalized_score: 0-100, already computed by scoring.py.
    Returns dict with final 0-100 score (higher = more distress signal) + breakdown.
    """
    facial_score = emotion_distress_score(np.array(facial_probs).flatten())
    speech_score = emotion_distress_score(np.array(speech_probs).flatten())

    final_score = (
        MODALITY_WEIGHTS["facial"] * facial_score +
        MODALITY_WEIGHTS["speech"] * speech_score +
        MODALITY_WEIGHTS["questionnaire"] * questionnaire_normalized_score
    )

    if final_score < 25:
        band = "low concern"
    elif final_score < 50:
        band = "mild concern"
    elif final_score < 70:
        band = "moderate concern"
    else:
        band = "high concern"

    return {
        "final_score": round(final_score, 2),
        "band": band,
        "breakdown": {
            "facial_distress": round(facial_score, 2),
            "speech_distress": round(speech_score, 2),
            "questionnaire_score": round(questionnaire_normalized_score, 2),
        }
    }


if __name__ == "__main__":
    example_facial = np.array([0.05, 0.05, 0.05, 0.05, 0.1, 0.7])
    example_speech = np.array([0.6, 0.05, 0.05, 0.05, 0.15, 0.1])
    example_q_score = 45.0

    result = compute_mental_health_score(example_facial, example_speech, example_q_score)
    print(result)
