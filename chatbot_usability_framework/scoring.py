"""
scoring.py
----------
Composite Usability Score (CUS) engine.

Combines the five normalized metric scores (each 0–100) into a single
Composite Usability Score using two models:

1. Equal-weighted CUS (baseline):
   CUS_equal = mean(context_retention, response_time, helpful_response,
                    lexical_diversity, error_rate) × 1/5 each (20%)

2. Regression-weighted CUS (final):
   Weights are derived from multiple linear regression of the five
   metric scores against human SUS scores (n ≥ 30).
   CUS_regression = β1×m1 + β2×m2 + β3×m3 + β4×m4 + β5×m5
   (coefficients loaded from analysis.py output after the SUS study)

Both CUS variants are on a 0–100 scale matching SUS for direct comparison.

Document references: Section 10.3, Section 4 (System Architecture).
"""

from __future__ import annotations
import os
import json


# ---------------------------------------------------------------------------
# Default equal weights (20% each)
# ---------------------------------------------------------------------------
METRIC_NAMES = [
    "context_retention",
    "response_time",
    "helpful_response",
    "lexical_diversity",
    "error_rate",
]

EQUAL_WEIGHTS = {m: 0.20 for m in METRIC_NAMES}

# Path for regression weights (written by analysis.py after SUS study)
_WEIGHTS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "regression_weights.json"
)


def _load_regression_weights() -> dict | None:
    """Load regression-derived weights if available. Returns None if not yet computed."""
    if not os.path.exists(_WEIGHTS_FILE):
        return None
    with open(_WEIGHTS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("weights")


# ---------------------------------------------------------------------------
# Score normalisation guard
# ---------------------------------------------------------------------------
def _validate_score(score: float | None, metric_name: str) -> float:
    """Clamp score to [0, 100]. Treat None as 0 with a warning."""
    if score is None:
        print(f"  [CUS] Warning: {metric_name} score is None — treating as 0.")
        return 0.0
    return max(0.0, min(100.0, float(score)))


# ---------------------------------------------------------------------------
# CUS computation
# ---------------------------------------------------------------------------
def compute_cus(metric_results: dict) -> dict:
    """
    Compute the Composite Usability Score from five metric result dicts.

    Parameters
    ----------
    metric_results : dict
        {
            "context_retention": {..., "normalized_score": float},
            "response_time":     {..., "normalized_score": float},
            "helpful_response":  {..., "normalized_score": float},
            "lexical_diversity": {..., "normalized_score": float},
            "error_rate":        {..., "normalized_score": float},
        }

    Returns
    -------
    dict:
        scores           : {metric: normalized_score}
        cus_equal        : float  — equal-weighted CUS (0–100)
        cus_regression   : float | None — regression-weighted CUS (if weights available)
        weights_used     : dict   — weights used for regression CUS (or None)
        classification   : str   — Good (≥70) | Moderate (50–70) | Poor (<50)
        regression_available : bool
    """
    scores: dict[str, float] = {}
    for metric in METRIC_NAMES:
        raw = metric_results.get(metric, {})
        if isinstance(raw, dict):
            val = raw.get("normalized_score")
        else:
            val = raw
        scores[metric] = _validate_score(val, metric)

    # Equal-weighted CUS
    cus_equal = round(sum(scores.values()) / len(METRIC_NAMES), 2)

    # Regression-weighted CUS
    reg_weights = _load_regression_weights()
    cus_regression = None
    if reg_weights:
        cus_regression = round(
            sum(reg_weights.get(m, 0.20) * scores[m] for m in METRIC_NAMES), 2
        )
        cus_regression = max(0.0, min(100.0, cus_regression))

    # Classify the equal-weighted CUS (baseline)
    if cus_equal >= 70:
        classification = "Good"
    elif cus_equal >= 50:
        classification = "Moderate"
    else:
        classification = "Poor"

    return {
        "scores": scores,
        "cus_equal": cus_equal,
        "cus_regression": cus_regression,
        "weights_used": reg_weights,
        "classification": classification,
        "regression_available": reg_weights is not None,
    }


def compute_cus_comparison(
    metric_results_rasa: dict,
    metric_results_ollama: dict,
) -> dict:
    """
    Compute and compare CUS for both chatbots side by side.

    Returns
    -------
    dict:
        rasa   : CUS result for Rasa
        ollama : CUS result for Ollama
        winner : str  — "rasa" | "ollama" | "tie"
        delta  : float  — |CUS_rasa - CUS_ollama|
    """
    rasa_cus = compute_cus(metric_results_rasa)
    ollama_cus = compute_cus(metric_results_ollama)

    r_score = rasa_cus["cus_equal"]
    o_score = ollama_cus["cus_equal"]
    delta = round(abs(r_score - o_score), 2)

    if delta < 1.0:
        winner = "tie"
    elif r_score > o_score:
        winner = "rasa"
    else:
        winner = "ollama"

    return {
        "rasa": rasa_cus,
        "ollama": ollama_cus,
        "winner": winner,
        "delta": delta,
    }


def save_regression_weights(weights: dict[str, float], metadata: dict = None):
    """
    Save regression-derived metric weights to disk.
    Called by analysis.py after the SUS regression completes.

    Parameters
    ----------
    weights : dict[str, float]
        {metric_name: weight_coefficient} — must sum to ≈1.0.
    metadata : dict
        Optional: R², p-values, n_participants, etc.
    """
    os.makedirs(os.path.dirname(_WEIGHTS_FILE), exist_ok=True)
    data = {"weights": weights}
    if metadata:
        data["metadata"] = metadata
    with open(_WEIGHTS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"  [CUS] Regression weights saved to: {_WEIGHTS_FILE}")


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_rasa = {
        "context_retention": {"normalized_score": 83.3},
        "response_time": {"normalized_score": 92.0},
        "helpful_response": {"normalized_score": 73.3},
        "lexical_diversity": {"normalized_score": 71.0},
        "error_rate": {"normalized_score": 86.7},
    }
    sample_ollama = {
        "context_retention": {"normalized_score": 66.7},
        "response_time": {"normalized_score": 78.0},
        "helpful_response": {"normalized_score": 86.7},
        "lexical_diversity": {"normalized_score": 82.0},
        "error_rate": {"normalized_score": 93.3},
    }

    comparison = compute_cus_comparison(sample_rasa, sample_ollama)
    print(f"Rasa CUS (equal):   {comparison['rasa']['cus_equal']} — {comparison['rasa']['classification']}")
    print(f"Ollama CUS (equal): {comparison['ollama']['cus_equal']} — {comparison['ollama']['classification']}")
    print(f"Winner: {comparison['winner']}  |  Delta: {comparison['delta']}")
