"""
metrics/metric_engine.py
------------------------
All five metric modules for the Chatbot Usability Testing Framework.

Metrics
-------
1. context_retention  — checkpoint pass rate % (ISO 9241-11 effectiveness)
2. response_time      — average latency in ms (ISO 9241-11 efficiency)
3. helpful_response   — helpful turn ratio % (ISO 9241-11 effectiveness)
4. lexical_diversity  — MATTR-50 score (response naturalness)
5. error_rate         — breakdown count / total turns (robustness)

Each module exposes a compute_<metric>(log) function that returns a
standardised result dict including a normalized_score (0–100) for the CUS.

Thresholds are from the framework document (Section 10.1), grounded in
cited literature (Nielsen 1993, ISO 9241-11, Følstad & Brandtzaeg 2017,
Comm100 2026, Higashinaka et al. 2015, Covington & McFall 2010).
"""

from __future__ import annotations
import re
from metrics.mattr_calculator import score_lexical_diversity


# ===========================================================================
# METRIC 1 — Context Retention Rate
# ===========================================================================
# Thresholds (ISO 9241-11 effectiveness; pilot-defined):
#   Good     ≥ 83%   (5–6 / 6 checkpoints pass)
#   Moderate 50–83%  (3–4 / 6)
#   Poor     < 50%   (0–2 / 6)

CHECKPOINT_TURNS = {5, 10, 15, 20, 25, 30}
_CONTEXT_KEYWORDS = [
    "vegetarian", "no meat", "plant-based", "meat-free",
    "vegan", "vegetable", "dairy", "egg",
]


def _checkpoint_passes(bot_response: str) -> bool:
    """
    Return True if the bot response acknowledges the vegetarian constraint.
    Checks for presence of any context keyword (case-insensitive).
    """
    resp_lower = bot_response.lower()
    return any(kw in resp_lower for kw in _CONTEXT_KEYWORDS)


def compute_context_retention(log: list[dict]) -> dict:
    """
    Score context retention from a conversation log.

    Looks at turns flagged as checkpoints (metadata.checkpoint_turn ∈
    {5,10,15,20,25,30}) and checks whether the bot response contains
    evidence that it remembered the planted fact ("I'm vegetarian").

    Returns
    -------
    dict:
        checkpoint_results  : list[dict]  per-checkpoint pass/fail
        passed              : int
        total_checkpoints   : int
        retention_rate_pct  : float
        classification      : str  Good | Moderate | Poor
        normalized_score    : float  0–100
    """
    checkpoint_results = []

    for turn in log:
        meta = turn.get("metadata", {})
        turn_num = turn.get("turn_number")
        if turn_num in CHECKPOINT_TURNS or meta.get("checkpoint"):
            passed = _checkpoint_passes(turn.get("bot_response", ""))
            checkpoint_results.append({
                "turn_number": turn_num,
                "checkpoint_id": meta.get("checkpoint"),
                "passed": passed,
                "bot_response_preview": turn.get("bot_response", "")[:120],
            })

    total = len(checkpoint_results)
    if total == 0:
        return {
            "checkpoint_results": [],
            "passed": 0,
            "total_checkpoints": 0,
            "retention_rate_pct": 0.0,
            "classification": "N/A",
            "normalized_score": 0.0,
        }

    passed = sum(1 for c in checkpoint_results if c["passed"])
    rate = (passed / total) * 100

    if rate >= 83:
        classification = "Good"
    elif rate >= 50:
        classification = "Moderate"
    else:
        classification = "Poor"

    return {
        "checkpoint_results": checkpoint_results,
        "passed": passed,
        "total_checkpoints": total,
        "retention_rate_pct": round(rate, 2),
        "classification": classification,
        "normalized_score": round(rate, 2),
    }


# ===========================================================================
# METRIC 2 — Average Response Time
# ===========================================================================
# Thresholds (Nielsen 1993; Følstad & Skjuve 2019):
#   Good     < 1000 ms
#   Moderate 1000–10000 ms
#   Poor     > 10000 ms
#
# Inversion for normalization: lower latency → higher score.
# Score formula: clamp(100 - (avg_ms / 10000) * 100, 0, 100)
# At avg=0 ms → 100; at avg=10000 ms → 0; clamped at 0 below 10000+.

RESPONSE_TIME_GOOD_MS = 1_000
RESPONSE_TIME_MODERATE_MS = 10_000


def compute_response_time(log: list[dict]) -> dict:
    """
    Compute average response latency from a conversation log.
    Error turns (latency_ms < 0) are excluded from the average.

    Returns
    -------
    dict:
        per_turn_latencies : list[dict]
        valid_turns        : int
        average_ms         : float | None
        min_ms             : float | None
        max_ms             : float | None
        classification     : str  Good | Moderate | Poor
        normalized_score   : float  0–100 (higher = faster)
    """
    per_turn = []
    valid_latencies = []

    for turn in log:
        lat = turn.get("latency_ms", -1)
        turn_num = turn.get("turn_number")
        turn_type = turn.get("metadata", {}).get("turn_type", "unknown")
        entry = {
            "turn_number": turn_num,
            "latency_ms": lat,
            "turn_type": turn_type,
            "valid": lat >= 0,
        }
        per_turn.append(entry)
        if lat >= 0:
            valid_latencies.append(lat)

    if not valid_latencies:
        return {
            "per_turn_latencies": per_turn,
            "valid_turns": 0,
            "average_ms": None,
            "min_ms": None,
            "max_ms": None,
            "classification": "N/A",
            "normalized_score": 0.0,
        }

    avg = sum(valid_latencies) / len(valid_latencies)
    classification = (
        "Good" if avg < RESPONSE_TIME_GOOD_MS
        else "Moderate" if avg <= RESPONSE_TIME_MODERATE_MS
        else "Poor"
    )
    # Invert: 0 ms → 100, 10000 ms → 0
    normalized = max(0.0, min(100.0, 100.0 - (avg / RESPONSE_TIME_MODERATE_MS) * 100))

    return {
        "per_turn_latencies": per_turn,
        "valid_turns": len(valid_latencies),
        "average_ms": round(avg, 2),
        "min_ms": round(min(valid_latencies), 2),
        "max_ms": round(max(valid_latencies), 2),
        "classification": classification,
        "normalized_score": round(normalized, 2),
    }


# ===========================================================================
# METRIC 3 — Helpful Response Ratio
# ===========================================================================
# Thresholds (Følstad & Brandtzaeg 2017; Comm100 2026 benchmark):
#   Good     > 80%   (≥ 12 / 15 turns helpful)
#   Moderate 60–80%  (9–11 / 15)
#   Poor     < 60%   (< 9 / 15)
#
# Scoring per turn:
#   - Rule-based turns: keyword match against expected keywords in metadata
#   - Prometheus turns: score injected externally via prometheus_judge.py
#   - Human spot-check turns: score injected externally

HELPFUL_GOOD_PCT = 80.0
HELPFUL_MODERATE_PCT = 60.0


def _rule_based_check(bot_response: str, expected_keywords: str) -> bool:
    """
    Return True if the bot response contains at least one expected keyword.
    expected_keywords is a pipe-separated string: "python|api|sql"
    """
    if not expected_keywords:
        return True  # No rule defined — default pass
    resp_lower = bot_response.lower()
    keywords = [kw.strip().lower() for kw in expected_keywords.split("|")]
    return any(kw in resp_lower for kw in keywords)


def compute_helpful_response(
    log: list[dict],
    prometheus_scores: dict[int, bool] = None,
    human_scores: dict[int, bool] = None,
) -> dict:
    """
    Compute Helpful Response Ratio from a conversation log.

    Parameters
    ----------
    log : list[dict]
        Conversation log. Each turn's metadata.scoring field indicates
        "rule" or "prometheus".
    prometheus_scores : dict[int, bool]
        {turn_number: is_helpful} from Prometheus 2 judge.
        If None, rule-based scoring is used for all turns.
    human_scores : dict[int, bool]
        {turn_number: is_helpful} from human spot-check rater.
        Used only for Cohen's Kappa computation (not for the main ratio).

    Returns
    -------
    dict:
        per_turn        : list[dict]
        helpful_turns   : int
        total_turns     : int
        helpful_ratio_pct: float
        classification  : str
        normalized_score: float  0–100
        human_scores_used: dict  (for Kappa calculation downstream)
    """
    if prometheus_scores is None:
        prometheus_scores = {}
    if human_scores is None:
        human_scores = {}

    per_turn = []
    helpful_count = 0
    total = 0

    for turn in log:
        turn_num = turn.get("turn_number")
        bot_response = turn.get("bot_response", "")
        meta = turn.get("metadata", {})
        scoring = meta.get("scoring", "rule")
        expected = meta.get("expected", "")

        # Determine helpfulness
        if scoring == "prometheus" and turn_num in prometheus_scores:
            is_helpful = prometheus_scores[turn_num]
            method = "prometheus"
        elif scoring == "prometheus":
            # Prometheus score not yet available — fall back to rule
            is_helpful = _rule_based_check(bot_response, expected)
            method = "rule_fallback"
        else:
            is_helpful = _rule_based_check(bot_response, expected)
            method = "rule"

        total += 1
        if is_helpful:
            helpful_count += 1

        entry = {
            "turn_number": turn_num,
            "scoring_method": method,
            "is_helpful": is_helpful,
            "expected_keywords": expected,
            "human_spot_check": meta.get("human_spot_check", False),
            "response_preview": bot_response[:100],
        }
        per_turn.append(entry)

    if total == 0:
        return {
            "per_turn": [],
            "helpful_turns": 0,
            "total_turns": 0,
            "helpful_ratio_pct": 0.0,
            "classification": "N/A",
            "normalized_score": 0.0,
            "human_scores_used": human_scores,
        }

    ratio = (helpful_count / total) * 100
    classification = (
        "Good" if ratio > HELPFUL_GOOD_PCT
        else "Moderate" if ratio >= HELPFUL_MODERATE_PCT
        else "Poor"
    )

    return {
        "per_turn": per_turn,
        "helpful_turns": helpful_count,
        "total_turns": total,
        "helpful_ratio_pct": round(ratio, 2),
        "classification": classification,
        "normalized_score": round(ratio, 2),
        "human_scores_used": human_scores,
    }


# ===========================================================================
# METRIC 4 — Lexical Diversity (MATTR-50)
# Delegates entirely to mattr_calculator.py
# ===========================================================================

def compute_lexical_diversity(log: list[dict]) -> dict:
    """
    Compute MATTR-50 lexical diversity for all turns in the log.
    Delegates to metrics/mattr_calculator.py.

    Returns the full result dict from score_lexical_diversity().
    """
    return score_lexical_diversity(log)


# ===========================================================================
# METRIC 5 — Error Rate
# ===========================================================================
# Thresholds (Comm100 2026 benchmark; Higashinaka et al. 2015 taxonomy):
#   Good     < 10%   (0–1 breakdown / 15 turns)
#   Moderate 10–25%  (2–3 / 15)
#   Poor     ≥ 25%   (≥ 4 / 15)
#
# A "breakdown" is detected when the bot fails to handle the turn:
#   - Response contains failure markers (see _ERROR_MARKERS below)
#   - OR response is empty / too short (< 5 chars)
#   - The error type is classified using Higashinaka et al. taxonomy

ERROR_GOOD_PCT = 10.0
ERROR_MODERATE_PCT = 25.0

_ERROR_MARKERS = [
    "i don't understand", "i'm not sure", "i cannot help",
    "i can't help", "sorry, i don't", "sorry, i can't",
    "i don't know", "i'm unable", "i cannot", "i can't",
    "could you clarify", "could you please clarify",
    "i'm not able", "i am not able", "this is outside",
    "out of scope", "i'm sorry, but",
    "[error]", "[no response]",
]


def _detect_breakdown(bot_response: str, meta: dict) -> tuple[bool, str]:
    """
    Return (is_breakdown: bool, reason: str).

    A breakdown is detected if:
    - Response is empty or < 5 characters.
    - Response contains a known failure marker.
    - For out-of-scope turns: the bot answers the out-of-scope question
      instead of redirecting (keyword-based heuristic).
    """
    if not bot_response or len(bot_response.strip()) < 5:
        return True, "empty_response"

    resp_lower = bot_response.lower()

    # Known error/confusion markers
    for marker in _ERROR_MARKERS:
        if marker in resp_lower:
            # Clarification requests are acceptable for ambiguous turns
            error_type = meta.get("error_type", "")
            if error_type == "ambiguous_no_context" and "clarif" in resp_lower:
                return False, "acceptable_clarification"
            return True, f"error_marker: {marker}"

    # For typo-heavy turns: if the expected keyword is completely missing, it's a breakdown
    error_type = meta.get("error_type", "")
    expected = meta.get("expected", "")
    if error_type == "typo_heavy" and expected:
        keywords = [kw.strip().lower() for kw in expected.split("|")]
        if not any(kw in resp_lower for kw in keywords):
            return True, "missed_intent_typo"

    # For out-of-scope turns: if the bot actually answers the OOS question, mark breakdown
    if error_type == "out_of_scope":
        # If the response doesn't contain any scope-redirect language, it's a breakdown
        redirect_markers = ["scope", "help", "support", "assist", "unable", "focus"]
        if not any(m in resp_lower for m in redirect_markers):
            return True, "answered_out_of_scope"

    return False, "ok"


def compute_error_rate(log: list[dict]) -> dict:
    """
    Compute Error Rate from a conversation log.

    Returns
    -------
    dict:
        per_turn         : list[dict]
        breakdown_count  : int
        total_turns      : int
        error_rate_pct   : float
        classification   : str  Good | Moderate | Poor
        normalized_score : float  0–100 (higher = fewer errors)
        taxonomy_summary : dict  {higashinaka_category: count}
    """
    per_turn = []
    breakdown_count = 0
    taxonomy_summary: dict[str, int] = {}

    for turn in log:
        bot_response = turn.get("bot_response", "")
        meta = turn.get("metadata", {})
        turn_num = turn.get("turn_number")
        higashinaka_cat = meta.get("higashinaka", "unknown")

        is_breakdown, reason = _detect_breakdown(bot_response, meta)
        if is_breakdown:
            breakdown_count += 1
            taxonomy_summary[higashinaka_cat] = taxonomy_summary.get(higashinaka_cat, 0) + 1

        per_turn.append({
            "turn_number": turn_num,
            "error_type": meta.get("error_type", "unknown"),
            "higashinaka_category": higashinaka_cat,
            "is_breakdown": is_breakdown,
            "reason": reason,
            "response_preview": bot_response[:100],
        })

    total = len(per_turn)
    if total == 0:
        return {
            "per_turn": [],
            "breakdown_count": 0,
            "total_turns": 0,
            "error_rate_pct": 0.0,
            "classification": "N/A",
            "normalized_score": 100.0,
            "taxonomy_summary": {},
        }

    rate = (breakdown_count / total) * 100
    classification = (
        "Good" if rate < ERROR_GOOD_PCT
        else "Moderate" if rate < ERROR_MODERATE_PCT
        else "Poor"
    )
    # Invert: 0% errors → 100 score, 100% errors → 0 score
    normalized = round(100.0 - rate, 2)

    return {
        "per_turn": per_turn,
        "breakdown_count": breakdown_count,
        "total_turns": total,
        "error_rate_pct": round(rate, 2),
        "classification": classification,
        "normalized_score": max(0.0, normalized),
        "taxonomy_summary": taxonomy_summary,
    }
