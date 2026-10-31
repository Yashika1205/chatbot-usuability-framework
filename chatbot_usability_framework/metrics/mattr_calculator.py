"""
metrics/mattr_calculator.py
----------------------------
MATTR-50 (Moving Average Type-Token Ratio) implementation.

Algorithm (Covington & McFall, 2010, Journal of Quantitative Linguistics 17(2)):
  1. Tokenize the text using NLTK word_tokenize.
  2. Lowercase all tokens.
  3. Slide a window of W=50 tokens across the token list (step = 1).
  4. For each window: TTR = unique tokens in window / W
  5. MATTR = mean of all window TTRs.

Responses with fewer than MIN_TOKENS (20) tokens are flagged SHORT and excluded
from averaging — prevents penalizing a correct one-sentence answer.

Thresholds are derived empirically from pilot data (see pilot_calibrate()).
Default placeholders are used until calibration is run.
"""

from __future__ import annotations
import os
import json
import nltk

# Ensure punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
WINDOW_SIZE: int = 50        # MATTR window (W=50 as per Covington & McFall 2010)
MIN_TOKENS: int = 20         # Exclude responses shorter than this

# Default threshold placeholders — MUST be replaced by pilot calibration
# before the main test run. See pilot_calibrate() below.
_THRESHOLDS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "mattr_thresholds.json"
)

DEFAULT_THRESHOLDS = {
    "good_min": 0.75,      # placeholder — set from pilot
    "moderate_min": 0.55,  # placeholder — set from pilot
    # below moderate_min = Poor
    "calibrated": False,
}


def _load_thresholds() -> dict:
    if os.path.exists(_THRESHOLDS_FILE):
        with open(_THRESHOLDS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_THRESHOLDS.copy()


def _save_thresholds(thresholds: dict):
    os.makedirs(os.path.dirname(_THRESHOLDS_FILE), exist_ok=True)
    with open(_THRESHOLDS_FILE, "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)


# ---------------------------------------------------------------------------
# Core MATTR calculation
# ---------------------------------------------------------------------------
def compute_mattr(text: str, window: int = WINDOW_SIZE) -> dict:
    """
    Compute the MATTR-50 score for a single text response.

    Parameters
    ----------
    text : str
        The bot response text.
    window : int
        Sliding window size (default 50).

    Returns
    -------
    dict with keys:
        mattr        : float | None  — score in [0,1], or None if SHORT
        token_count  : int
        status       : str           — "scored" | "short_excluded"
        window_ttrs  : list[float]   — all per-window TTRs (empty if short)
    """
    tokens = nltk.word_tokenize(text.lower())
    # Keep only alphabetic/alphanumeric tokens (filter punctuation)
    tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    n = len(tokens)

    if n < MIN_TOKENS:
        return {
            "mattr": None,
            "token_count": n,
            "status": "short_excluded",
            "window_ttrs": [],
        }

    if n < window:
        # Fewer tokens than window: compute single TTR over all tokens
        ttr = len(set(tokens)) / n
        return {
            "mattr": round(ttr, 4),
            "token_count": n,
            "status": "scored",
            "window_ttrs": [ttr],
        }

    window_ttrs = []
    for i in range(n - window + 1):
        w = tokens[i : i + window]
        ttr = len(set(w)) / window
        window_ttrs.append(ttr)

    mattr = sum(window_ttrs) / len(window_ttrs)
    return {
        "mattr": round(mattr, 4),
        "token_count": n,
        "status": "scored",
        "window_ttrs": [round(x, 4) for x in window_ttrs],
    }


def classify_mattr(score: float, thresholds: dict = None) -> str:
    """
    Classify an MATTR score as Good / Moderate / Poor.

    Uses pilot-calibrated thresholds if available.
    """
    if thresholds is None:
        thresholds = _load_thresholds()
    if score >= thresholds["good_min"]:
        return "Good"
    elif score >= thresholds["moderate_min"]:
        return "Moderate"
    else:
        return "Poor"


# ---------------------------------------------------------------------------
# Script-level MATTR scoring (operates on a conversation log)
# ---------------------------------------------------------------------------
def score_lexical_diversity(log: list[dict]) -> dict:
    """
    Compute MATTR-50 for every turn in a conversation log and return
    the per-turn results and the averaged script score.

    Parameters
    ----------
    log : list[dict]
        Conversation log from ConversationLogger (bot_response field used).

    Returns
    -------
    dict with keys:
        per_turn        : list[dict]  — per-turn MATTR results
        scored_turns    : int         — turns included in average
        excluded_turns  : int         — turns excluded (SHORT)
        average_mattr   : float | None
        classification  : str         — Good | Moderate | Poor | N/A
        normalized_score: float       — 0–100 for CUS
    """
    thresholds = _load_thresholds()
    per_turn = []
    scored_scores = []

    for turn in log:
        response = turn.get("bot_response", "")
        result = compute_mattr(response)
        result["turn_number"] = turn.get("turn_number")
        per_turn.append(result)
        if result["status"] == "scored" and result["mattr"] is not None:
            scored_scores.append(result["mattr"])

    if not scored_scores:
        avg = None
        classification = "N/A"
        normalized = 0.0
    else:
        avg = round(sum(scored_scores) / len(scored_scores), 4)
        classification = classify_mattr(avg, thresholds)
        # Normalize to 0–100 (MATTR is already 0–1 so multiply by 100)
        normalized = round(avg * 100, 2)

    return {
        "per_turn": per_turn,
        "scored_turns": len(scored_scores),
        "excluded_turns": len(per_turn) - len(scored_scores),
        "average_mattr": avg,
        "classification": classification,
        "normalized_score": normalized,
    }


# ---------------------------------------------------------------------------
# Pilot calibration
# ---------------------------------------------------------------------------
def pilot_calibrate(good_responses: list[str], padded_responses: list[str]) -> dict:
    """
    Run MATTR on known-good and known-padded responses and derive thresholds
    at the observed median boundary.

    This must be run before the main test. Results are saved to
    results/mattr_thresholds.json and used automatically by score_lexical_diversity.

    Parameters
    ----------
    good_responses : list[str]
        3+ accurate, specific, varied-vocabulary responses.
    padded_responses : list[str]
        3+ filler-heavy, repetitive responses.

    Returns
    -------
    dict  The calibrated thresholds dict.
    """
    good_scores = [
        compute_mattr(r)["mattr"] for r in good_responses
        if compute_mattr(r)["mattr"] is not None
    ]
    padded_scores = [
        compute_mattr(r)["mattr"] for r in padded_responses
        if compute_mattr(r)["mattr"] is not None
    ]

    if not good_scores or not padded_scores:
        raise ValueError("Need at least one scoreable response in each category.")

    good_min_score = min(good_scores)
    padded_max_score = max(padded_scores)

    # Good threshold = min of good scores (conservative)
    # Moderate boundary = midpoint between worst good and best padded
    good_threshold = good_min_score
    midpoint = (good_min_score + padded_max_score) / 2

    thresholds = {
        "good_min": round(good_threshold, 4),
        "moderate_min": round(midpoint, 4),
        "calibrated": True,
        "good_scores": [round(s, 4) for s in good_scores],
        "padded_scores": [round(s, 4) for s in padded_scores],
    }
    _save_thresholds(thresholds)
    return thresholds


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    good = [
        "Index funds are diversified investment vehicles that track a market index like the S&P 500. "
        "They offer low expense ratios, broad diversification, and consistent long-term returns, "
        "making them an excellent choice for most retail investors who want market exposure without "
        "the complexity of stock picking. Over a 20-year horizon, the majority of actively managed "
        "funds underperform their benchmark index, which further supports passive investing.",
        "Compound interest is the process by which interest is earned not only on your initial principal "
        "but also on the accumulated interest from previous periods. For example, if you invest $1000 at "
        "5% annually, after year one you have $1050. In year two, you earn 5% on $1050, not $1000, "
        "resulting in $1102.50. Over decades, this exponential growth can dramatically increase your wealth.",
        "To create a monthly budget, start by listing all income sources, then categorise your fixed "
        "expenses like rent and utilities. Next, track variable expenses such as groceries and entertainment. "
        "Subtract total expenses from income to find your surplus or deficit. Adjust discretionary spending "
        "to ensure you save at least 20% of your income, following the 50-30-20 rule as a guideline.",
    ]
    padded = [
        "That is a great question! I am happy to help you with that. Sure, I can definitely assist. "
        "Basically, in general terms, what you are asking is a good thing. I would say that yes, it is "
        "definitely something worth considering. You know, many people ask about this very thing and it "
        "is indeed very important. So yes, it is a good question and I hope I have been helpful today.",
        "Well, you know, that is a really interesting thing you have brought up. I think, generally speaking, "
        "when we talk about these kinds of topics, it is always good to think carefully. There are many factors "
        "to consider here. At the end of the day, what matters most is that you think about what is best for you. "
        "I hope that helps! Please feel free to ask if you have any more questions about this topic.",
    ]

    thresholds = pilot_calibrate(good, padded)
    print("Calibrated thresholds:", thresholds)

    test_response = good[0]
    result = compute_mattr(test_response)
    print(f"\nTest MATTR: {result['mattr']} ({classify_mattr(result['mattr'])})")
    print(f"Token count: {result['token_count']}, Windows: {len(result['window_ttrs'])}")
