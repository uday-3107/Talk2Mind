import json


def load_questionnaire(path="questions.json"):
    with open(path) as f:
        return json.load(f)


def score_responses(responses: dict) -> dict:
    """
    responses: {"q1": 0-3, "q2": 0-3, ..., "q9": 0-3}
    Returns raw total (0-27) and severity band.
    """
    total = sum(responses.values())

    if total <= 4:
        severity = "minimal"
    elif total <= 9:
        severity = "mild"
    elif total <= 14:
        severity = "moderate"
    elif total <= 19:
        severity = "moderately severe"
    else:
        severity = "severe"

    # flag item 9 (self-harm ideation) regardless of total score
    risk_flag = responses.get("q9", 0) > 0

    # normalize to 0-100 scale for fusion with facial/speech scores
    normalized_score = (total / 27) * 100

    return {
        "raw_total": total,
        "normalized_score": round(normalized_score, 2),
        "severity": severity,
        "risk_flag": risk_flag,
    }


if __name__ == "__main__":
    # example usage
    example_responses = {f"q{i}": 1 for i in range(1, 10)}
    result = score_responses(example_responses)
    print(result)
