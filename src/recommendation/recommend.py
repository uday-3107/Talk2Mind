import json
import os
from groq import Groq


def generate_conversational_questions(qdata: dict) -> list:
    """
    Calls Groq once to rephrase all PHQ-9 items conversationally, with
    friendly transitions. Falls back to original wording if the API fails.
    Returns a list of strings, same length/order as qdata['items'].
    Answer options are NOT touched — scoring stays on the original fixed items.
    """
    original_texts = [item["text"] for item in qdata["items"]]
    fallback = original_texts

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return fallback

    try:
        client = Groq(api_key=api_key)
        prompt = (
            "Rephrase each of these 9 mental health screening questions as a warm, "
            "conversational question a caring assistant might ask one at a time, "
            "with a brief natural transition before each (e.g. 'Thanks for sharing. "
            "Next, I'd like to know...'). Keep each item's clinical meaning EXACTLY "
            "the same — do not soften or change what is being asked, only the tone. "
            "Return ONLY a JSON array of 9 strings, no other text, no markdown.\n\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(original_texts))
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=600,
            timeout=10,
        )
        text = response.choices[0].message.content.strip()
        text = text.strip("`").replace("json", "", 1).strip() if text.startswith("```") else text
        parsed = json.loads(text)
        if isinstance(parsed, list) and len(parsed) == len(original_texts):
            return parsed
        return fallback
    except Exception as e:
        print(f"Conversational question generation failed, using fallback: {e}")
        return fallback


def generate_empathetic_summary(score_result: dict, recs: dict) -> str:
    """
    Calls Groq API to generate a short, natural-language empathetic summary
    of the session. Falls back to a static message if the API call fails
    for any reason (no key, network issue, rate limit, etc.) so the
    dashboard never breaks because of this feature.
    """
    fallback = (
        f"Your check-in shows a well-being score of {score_result['final_score']}/100 "
        f"({score_result['band']}). Please review the recommendations below."
    )

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return fallback

    try:
        client = Groq(api_key=api_key)
        prompt = (
            f"You are a warm, supportive assistant summarizing a mental well-being "
            f"check-in. The user's score is {score_result['final_score']}/100 "
            f"(band: {score_result['band']}). Facial distress: "
            f"{score_result['breakdown']['facial_distress']}, speech distress: "
            f"{score_result['breakdown']['speech_distress']}, questionnaire score: "
            f"{score_result['breakdown']['questionnaire_score']}. "
            f"Write a short (3-4 sentence), warm, non-clinical summary acknowledging "
            f"how they might be feeling, without diagnosing them or repeating exact "
            f"numbers back robotically. Do not give medical advice — recommendations "
            f"are shown separately."
        )
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            timeout=8,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Groq summary generation failed, using fallback: {e}")
        return fallback

RECOMMENDATIONS = {
    "low concern": [
        "Keep up your current routines — regular sleep, movement, and social contact.",
        "Consider a short daily reflection or gratitude note to maintain your baseline.",
    ],
    "mild concern": [
        "Try a short daily breathing or mindfulness exercise (5-10 minutes).",
        "Prioritize consistent sleep timing this week.",
        "Light physical activity (a 20-30 minute walk) can help regulate mood.",
    ],
    "moderate concern": [
        "Consider talking to a trusted friend, family member, or counselor about how you've been feeling.",
        "Reduce workload or commitments where possible this week to lower stress load.",
        "Structured relaxation techniques (progressive muscle relaxation, guided breathing) may help.",
        "If low mood or stress persists beyond two weeks, consider a check-in with a mental health professional.",
    ],
    "high concern": [
        "We strongly recommend speaking with a licensed mental health professional soon.",
        "Reach out to someone you trust today — you don't need to manage this alone.",
        "If you're in India, iCall (9152987821) offers free, confidential counseling support.",
    ],
}

CRISIS_MESSAGE = (
    "Your responses indicate thoughts of self-harm. This is serious, and support is available right now. "
    "If you are in immediate danger, please contact a local emergency number, or a crisis line such as "
    "iCall (9152987821, India) or a helpline in your country. Please consider reaching out to someone "
    "you trust or a mental health professional as soon as possible."
)


def get_recommendations(band: str, risk_flag: bool = False) -> dict:
    """
    band: one of 'low concern', 'mild concern', 'moderate concern', 'high concern'
          (output of fusion_scoring.compute_mental_health_score)
    risk_flag: from questionnaire scoring (PHQ-9 item 9)
    """
    result = {
        "band": band,
        "recommendations": RECOMMENDATIONS.get(band, RECOMMENDATIONS["moderate concern"]),
        "crisis_alert": None,
    }

    if risk_flag:
        result["crisis_alert"] = CRISIS_MESSAGE
        # risk_flag overrides band-based framing — always show crisis message first
        result["recommendations"] = [CRISIS_MESSAGE] + result["recommendations"]

    return result


if __name__ == "__main__":
    print(get_recommendations("moderate concern", risk_flag=False))
    print()
    print(get_recommendations("low concern", risk_flag=True))