import json
import os
from datetime import datetime

HISTORY_DIR = "data/user_history"
os.makedirs(HISTORY_DIR, exist_ok=True)


def _history_path(user_id: str) -> str:
    return os.path.join(HISTORY_DIR, f"{user_id}.json")


def load_history(user_id: str) -> list:
    path = _history_path(user_id)
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return json.load(f)


def save_session(user_id: str, score_result: dict, feedback: dict = None):
    """
    feedback: optional dict like {"mood_change": "better"/"same"/"worse", "accuracy_rating": 1-5}
    Stored alongside each session so future sessions can build a labeled dataset.
    """
    history = load_history(user_id)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "final_score": score_result["final_score"],
        "band": score_result["band"],
        "breakdown": score_result.get("breakdown", {}),
        "feedback": feedback,
    }
    history.append(entry)
    with open(_history_path(user_id), "w") as f:
        json.dump(history, f, indent=2)
    return history


def get_trend(user_id: str, window: int = 3, current_score: float = None) -> dict:
    """
    current_score: optional — pass the score from the session just run
    (not yet saved) so the trend reflects it immediately, instead of
    only comparing against previously saved sessions.
    """
    history = load_history(user_id)
    scores = [h["final_score"] for h in history]
    if current_score is not None:
        scores.append(current_score)

    if len(scores) < 2:
        return {"direction": "insufficient_data", "delta": None, "sessions_logged": len(scores)}

    latest = scores[-1]
    prior = scores[max(0, len(scores) - 1 - window):-1]
    prior_avg = sum(prior) / len(prior)
    delta = latest - prior_avg

    if abs(delta) < 3:
        direction = "stable"
    elif delta > 0:
        direction = "worsening"
    else:
        direction = "improving"

    return {
        "direction": direction,
        "delta": round(delta, 2),
        "sessions_logged": len(scores),
        "latest_score": latest,
        "prior_avg": round(prior_avg, 2),
    }


def get_history_dataframe(user_id: str, current_score: float = None):
    """
    Returns session history as a pandas DataFrame ready for charting:
    columns = timestamp, final_score. Empty DataFrame if no history.
    current_score: optional — appends the current in-progress session's
    score with 'now' as its timestamp, so the chart matches the trend text.
    """
    import pandas as pd
    from datetime import datetime
    history = load_history(user_id)
    rows = [{"timestamp": h["timestamp"], "final_score": h["final_score"]} for h in history]
    if current_score is not None:
        rows.append({"timestamp": datetime.now().isoformat(), "final_score": current_score})
    if not rows:
        return pd.DataFrame(columns=["timestamp", "final_score"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def trend_message(trend: dict) -> str:
    if trend["direction"] == "insufficient_data":
        return "This is early in your tracking — trends will appear after a few more sessions."
    elif trend["direction"] == "improving":
        return f"Your recent scores show improvement compared to your last {trend['sessions_logged']-1} sessions."
    elif trend["direction"] == "worsening":
        return "Your recent scores suggest increased distress compared to your recent sessions — consider extra support this week."
    else:
        return "Your well-being score has been stable across recent sessions."


if __name__ == "__main__":
    uid = "test_user2"
    for score in [40, 45, 50, 60]:
        save_session(uid, {"final_score": score, "band": "mild concern"})
    trend = get_trend(uid)
    print(trend)
    print(trend_message(trend))