"""
sus_study.py
------------
SUS (System Usability Scale) human study data collector.

Implements:
1. Interactive CLI to collect SUS responses from one participant.
2. Batch loader to import SUS responses from a CSV file.
3. SUS score calculator (Brooke 1996).
4. Aggregator for n ≥ 30 participants.

SUS is a 10-item Likert questionnaire (1–5 scale) yielding a 0–100 score.
Calculation (Brooke 1996):
  - Odd-numbered items (1,3,5,7,9): score = response - 1
  - Even-numbered items (2,4,6,8,10): score = 5 - response
  - Total raw score × 2.5 = SUS score (0–100)

Scoring benchmark:
  ≥ 85 = Excellent (Grade A)
  70–84 = Good (Grade B/C)
  50–69 = OK (Grade D)
  < 50  = Poor (Grade F)

Citation: Brooke, J. (1996). SUS: A quick and dirty usability scale.
  In P. Jordan et al. (Eds.), Usability Evaluation in Industry (pp. 189-194).
  Taylor & Francis.
"""

from __future__ import annotations
import os
import csv
import json
from datetime import datetime, timezone

_SUS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "sus_data"
)
os.makedirs(_SUS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Standard SUS questions (Brooke 1996)
# ---------------------------------------------------------------------------
SUS_QUESTIONS = [
    "1.  I think that I would like to use this system frequently.",
    "2.  I found the system unnecessarily complex.",
    "3.  I thought the system was easy to use.",
    "4.  I think that I would need the support of a technical person to use this system.",
    "5.  I found the various functions in this system were well integrated.",
    "6.  I thought there was too much inconsistency in this system.",
    "7.  I would imagine that most people would learn to use this system very quickly.",
    "8.  I found the system very cumbersome to use.",
    "9.  I felt very confident using the system.",
    "10. I needed to learn a lot of things before I could get going with this system.",
]

SUS_SCALE = "Rate each statement from 1 (Strongly Disagree) to 5 (Strongly Agree)."

# Odd items (1-indexed): contribution = response - 1
# Even items: contribution = 5 - response
ODD_ITEMS  = {1, 3, 5, 7, 9}
EVEN_ITEMS = {2, 4, 6, 8, 10}


# ---------------------------------------------------------------------------
# Score calculation
# ---------------------------------------------------------------------------
def calculate_sus_score(responses: list[int]) -> float:
    """
    Calculate SUS score from a list of 10 Likert responses (1–5).

    Parameters
    ----------
    responses : list[int]
        10 responses in order (items 1–10), each 1–5.

    Returns
    -------
    float  SUS score 0–100.
    """
    if len(responses) != 10:
        raise ValueError(f"SUS requires exactly 10 responses, got {len(responses)}.")
    for i, r in enumerate(responses, 1):
        if r not in (1, 2, 3, 4, 5):
            raise ValueError(f"Response {i} is {r!r} — must be 1–5.")

    raw = 0
    for i, r in enumerate(responses, 1):
        if i in ODD_ITEMS:
            raw += r - 1
        else:
            raw += 5 - r

    return round(raw * 2.5, 1)


def classify_sus(score: float) -> str:
    if score >= 85:
        return "Excellent (A)"
    elif score >= 70:
        return "Good (B/C)"
    elif score >= 50:
        return "OK (D)"
    else:
        return "Poor (F)"


# ---------------------------------------------------------------------------
# Single-participant CLI collector
# ---------------------------------------------------------------------------
def collect_sus_interactive(
    participant_id: str,
    chatbot_name: str,
) -> dict:
    """
    Interactively collect SUS responses from one participant via CLI.

    Parameters
    ----------
    participant_id : str
        Unique ID for this participant (e.g. "P01").
    chatbot_name : str
        "rasa" or "ollama" — which chatbot they just used.

    Returns
    -------
    dict  Full participant record with SUS score.
    """
    print(f"\n{'='*60}")
    print(f"SUS QUESTIONNAIRE — {chatbot_name.upper()} | Participant: {participant_id}")
    print(f"{'='*60}")
    print(SUS_SCALE)
    print()

    responses = []
    for i, question in enumerate(SUS_QUESTIONS, 1):
        while True:
            try:
                val = int(input(f"  {question}\n  Your rating (1-5): ").strip())
                if 1 <= val <= 5:
                    responses.append(val)
                    break
                else:
                    print("  Please enter a number between 1 and 5.")
            except ValueError:
                print("  Invalid input. Please enter 1, 2, 3, 4, or 5.")

    score = calculate_sus_score(responses)
    record = {
        "participant_id": participant_id,
        "chatbot_name": chatbot_name,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "responses": responses,
        "sus_score": score,
        "classification": classify_sus(score),
    }

    print(f"\n  SUS Score for {chatbot_name}: {score} / 100  ({classify_sus(score)})")
    return record


# ---------------------------------------------------------------------------
# CSV import
# ---------------------------------------------------------------------------
def load_sus_from_csv(filepath: str) -> list[dict]:
    """
    Load SUS responses from a CSV file.

    Expected CSV columns:
        participant_id, chatbot_name, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10

    Returns
    -------
    list[dict]  Participant records with calculated SUS scores.
    """
    records = []
    with open(filepath, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                responses = [int(row[f"q{i}"]) for i in range(1, 11)]
                score = calculate_sus_score(responses)
                records.append({
                    "participant_id": row.get("participant_id", "unknown"),
                    "chatbot_name": row.get("chatbot_name", "unknown"),
                    "timestamp_utc": row.get("timestamp_utc", ""),
                    "responses": responses,
                    "sus_score": score,
                    "classification": classify_sus(score),
                })
            except (KeyError, ValueError) as e:
                print(f"  [SUS] Skipping row {row}: {e}")
    return records


# ---------------------------------------------------------------------------
# Save and aggregate
# ---------------------------------------------------------------------------
def save_sus_records(records: list[dict], chatbot_name: str):
    """Save SUS records to JSON in results/sus_data/."""
    filename = f"sus_{chatbot_name}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}.json"
    filepath = os.path.join(_SUS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
    print(f"  [SUS] Saved {len(records)} records to {filepath}")
    return filepath


def aggregate_sus(records: list[dict]) -> dict:
    """
    Compute aggregate SUS statistics from a list of participant records.

    Returns
    -------
    dict:
        n            : int
        mean_sus     : float
        std_sus      : float
        min_sus      : float
        max_sus      : float
        classification: str  (of mean SUS)
        scores       : list[float]
    """
    import statistics
    scores = [r["sus_score"] for r in records]
    n = len(scores)

    if n == 0:
        return {"n": 0, "mean_sus": None, "error": "No records."}

    mean = round(statistics.mean(scores), 2)
    std = round(statistics.stdev(scores) if n > 1 else 0.0, 2)

    return {
        "n": n,
        "mean_sus": mean,
        "std_sus": std,
        "min_sus": round(min(scores), 1),
        "max_sus": round(max(scores), 1),
        "classification": classify_sus(mean),
        "scores": scores,
    }


def load_all_sus_for_chatbot(chatbot_name: str) -> list[dict]:
    """Load all SUS records for a given chatbot from results/sus_data/."""
    records = []
    for filename in os.listdir(_SUS_DIR):
        if filename.startswith(f"sus_{chatbot_name}_") and filename.endswith(".json"):
            path = os.path.join(_SUS_DIR, filename)
            with open(path, "r", encoding="utf-8") as f:
                records.extend(json.load(f))
    return records


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Simulate 5 participants for each chatbot
    import random
    random.seed(42)

    for chatbot in ["rasa", "ollama"]:
        print(f"\n--- Simulating 5 participants for {chatbot} ---")
        records = []
        for i in range(1, 6):
            # Rasa: slightly lower usability; Ollama: higher
            base = 55 if chatbot == "rasa" else 70
            responses = [max(1, min(5, random.randint(base // 20, 5))) for _ in range(10)]
            score = calculate_sus_score(responses)
            records.append({
                "participant_id": f"P{i:02d}",
                "chatbot_name": chatbot,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "responses": responses,
                "sus_score": score,
                "classification": classify_sus(score),
            })

        agg = aggregate_sus(records)
        print(f"  n={agg['n']}  mean={agg['mean_sus']}  std={agg['std_sus']}  [{agg['classification']}]")
        save_sus_records(records, chatbot)
