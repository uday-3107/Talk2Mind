import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.fusion.fusion_scoring import compute_mental_health_score

HISTORY_DIR = "data/user_history"
RESULTS_DIR = "evaluation/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad"]
RNG = np.random.default_rng(42)


def load_all_sessions():
    sessions = []
    if not os.path.isdir(HISTORY_DIR):
        return sessions
    for fname in os.listdir(HISTORY_DIR):
        if not fname.endswith(".json"):
            continue
        path = os.path.join(HISTORY_DIR, fname)
        with open(path) as f:
            data = json.load(f)
            if isinstance(data, list):
                sessions.extend(data)
    return sessions


def random_emotion_probs():
    alpha = np.ones(6) * RNG.uniform(0.2, 0.8)
    if RNG.random() < 0.5:
        idx = RNG.integers(6)
        alpha[idx] = RNG.uniform(3, 10)
    return RNG.dirichlet(alpha)


def generate_synthetic_sessions(n=30):
    synthetic = []
    for _ in range(n):
        facial_probs = random_emotion_probs()
        speech_probs = random_emotion_probs()
        q_score = RNG.uniform(0, 100)
        result = compute_mental_health_score(facial_probs, speech_probs, q_score)
        synthetic.append(result)
    return synthetic


def plot_histogram(sessions):
    scores = [s["final_score"] for s in sessions]
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = [0, 25, 50, 70, 100]
    ax.hist(scores, bins=bins, color="steelblue", edgecolor="white", alpha=0.8)
    for i in range(len(bins) - 1):
        lower = bins[i]
        upper = bins[i + 1]
        count = sum(1 for s in scores if lower <= s < upper)
        if i == len(bins) - 2:
            count += sum(1 for s in scores if s == 100)
        mid = (bins[i] + bins[i + 1]) / 2
        ax.text(mid, count + 0.5, str(count), ha="center", fontweight="bold")
    ax.set_xlabel("Mental Well-Being Score")
    ax.set_ylabel("Number of Sessions")
    ax.set_title(f"Score Distribution (n={len(scores)})")
    ax.axvline(25, color="green", linestyle="--", alpha=0.5, label="low/mild boundary")
    ax.axvline(50, color="orange", linestyle="--", alpha=0.5, label="mild/moderate boundary")
    ax.axvline(70, color="red", linestyle="--", alpha=0.5, label="moderate/high boundary")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "score_distribution.png"), dpi=150)
    print(f"Histogram saved to {RESULTS_DIR}/score_distribution.png")
    plt.close()


def band_breakdown(sessions):
    bands = {"low concern": 0, "mild concern": 0, "moderate concern": 0, "high concern": 0}
    for s in sessions:
        b = s["band"]
        if b in bands:
            bands[b] += 1
    total = len(sessions)
    print(f"\n{'Band':<25} {'Count':<8} {'%':<8} {'Flag':<10}")
    print("-" * 50)
    any_flag = False
    for band in ["low concern", "mild concern", "moderate concern", "high concern"]:
        count = bands[band]
        pct = (count / total) * 100 if total else 0
        flag = ""
        if count == 0:
            flag = "EMPTY"
            any_flag = True
        elif pct < 5:
            flag = f"<5% ({pct:.1f}%)"
            any_flag = True
        print(f"{band:<25} {count:<8} {pct:<8.1f} {flag:<10}")
    if any_flag:
        print("\n>> One or more bands have <5% representation — fusion weights may need rebalancing.")
    else:
        print("\n>> All bands adequately populated — current fusion distribution looks balanced.")
    return bands


def main():
    real_sessions = load_all_sessions()
    print(f"Found {len(real_sessions)} real session(s) in {HISTORY_DIR}/")

    if len(real_sessions) < 10:
        needed = 30
        print(f"Fewer than 10 real sessions. Generating {needed} synthetic sessions...")
        synthetic = generate_synthetic_sessions(needed)
        all_sessions = real_sessions + synthetic
        print(
            f"Using {len(real_sessions)} real + {len(synthetic)} synthetic = "
            f"{len(all_sessions)} total sessions."
        )
    else:
        all_sessions = real_sessions
        print(f"Using {len(all_sessions)} real sessions (>=10, no synthetic needed).")

    plot_histogram(all_sessions)
    band_breakdown(all_sessions)


if __name__ == "__main__":
    main()
