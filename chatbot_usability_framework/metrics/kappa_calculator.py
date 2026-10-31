"""
metrics/kappa_calculator.py
---------------------------
Cohen's Kappa (κ) inter-rater reliability calculator.

Used to validate the Prometheus 2 LLM judge against human ratings
on the 5 overlap turns of the helpful_response script
(turns 2, 5, 7, 10, 14 — see framework document Section 10.2).

Target: κ ≥ 0.61 (Substantial agreement per Landis & Koch 1977).
If not achieved, replace Prometheus 2 with full human scoring.

Kappa scale (Landis & Koch 1977, Biometrics 33(1):159-174):
    κ < 0.20     = Slight
    κ 0.21–0.40  = Fair
    κ 0.41–0.60  = Moderate
    κ 0.61–0.80  = Substantial ✓ (target)
    κ > 0.80     = Almost perfect
"""

from __future__ import annotations
import os
import json


# Kappa scale as defined by Landis & Koch (1977)
KAPPA_SCALE = [
    (0.80, "Almost perfect"),
    (0.61, "Substantial ✓ (target met)"),
    (0.41, "Moderate"),
    (0.21, "Fair"),
    (0.00, "Slight"),
]

KAPPA_TARGET = 0.61

# Turns used for Kappa validation (overlap: Prometheus 2 + human)
KAPPA_OVERLAP_TURNS = {2, 5, 7, 10, 14}

# Path for saving Kappa results
_KAPPA_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "results", "kappa_results.json"
)


def compute_kappa(
    rater_a: list[bool],
    rater_b: list[bool],
) -> dict:
    """
    Compute Cohen's Kappa between two binary raters.

    Parameters
    ----------
    rater_a : list[bool]
        Ratings from rater A (Prometheus 2: True = helpful).
    rater_b : list[bool]
        Ratings from rater B (human: True = helpful).

    Both lists must be the same length and correspond to the same items.

    Returns
    -------
    dict:
        kappa              : float
        interpretation     : str
        target_met         : bool  (κ ≥ 0.61)
        agreement_count    : int
        total_items        : int
        observed_agreement : float  (P_o)
        expected_agreement : float  (P_e)
        confusion_matrix   : dict
    """
    if len(rater_a) != len(rater_b):
        raise ValueError(
            f"Rater lists must be the same length. "
            f"Got {len(rater_a)} and {len(rater_b)}."
        )
    n = len(rater_a)
    if n == 0:
        raise ValueError("Cannot compute Kappa on empty lists.")

    # Build confusion matrix: (a_val, b_val) → count
    # Categories: True (helpful), False (not helpful)
    from collections import Counter
    confusion: dict[tuple, int] = Counter()
    for a, b in zip(rater_a, rater_b):
        confusion[(bool(a), bool(b))] += 1

    tp = confusion.get((True, True), 0)
    tn = confusion.get((False, False), 0)
    fp = confusion.get((False, True), 0)   # A says not helpful, B says helpful
    fn = confusion.get((True, False), 0)   # A says helpful, B says not helpful

    # Observed agreement (P_o)
    p_o = (tp + tn) / n

    # Expected agreement (P_e) — product of marginals
    a_pos = (tp + fn) / n   # proportion A rates helpful
    a_neg = (tn + fp) / n
    b_pos = (tp + fp) / n   # proportion B rates helpful
    b_neg = (tn + fn) / n

    p_e = (a_pos * b_pos) + (a_neg * b_neg)

    # Cohen's Kappa
    if p_e == 1.0:
        # Perfect expected agreement — Kappa undefined; treat as 1.0
        kappa = 1.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    kappa = round(kappa, 4)

    # Interpret
    interpretation = "Slight"
    for threshold, label in KAPPA_SCALE:
        if kappa >= threshold:
            interpretation = label
            break

    return {
        "kappa": kappa,
        "interpretation": interpretation,
        "target_met": kappa >= KAPPA_TARGET,
        "agreement_count": tp + tn,
        "total_items": n,
        "observed_agreement": round(p_o, 4),
        "expected_agreement": round(p_e, 4),
        "confusion_matrix": {
            "true_positive": tp,
            "true_negative": tn,
            "false_positive_a_says_no_b_says_yes": fp,
            "false_negative_a_says_yes_b_says_no": fn,
        },
    }


def compute_kappa_from_scores(
    prometheus_scores: dict[int, bool],
    human_scores: dict[int, bool],
    overlap_turns: set[int] = None,
) -> dict:
    """
    Compute Kappa from the score dicts produced by prometheus_judge.py
    and a human rater.

    Parameters
    ----------
    prometheus_scores : dict[int, bool]
        {turn_number: is_helpful} from Prometheus 2
    human_scores : dict[int, bool]
        {turn_number: is_helpful} from human rater
    overlap_turns : set[int]
        Turns scored by both. Default: KAPPA_OVERLAP_TURNS {2,5,7,10,14}.

    Returns
    -------
    dict  The full Kappa result dict + which turns were used.
    """
    if overlap_turns is None:
        overlap_turns = KAPPA_OVERLAP_TURNS

    common = sorted(overlap_turns & set(prometheus_scores.keys()) & set(human_scores.keys()))
    if not common:
        return {
            "kappa": None,
            "error": "No common scored turns found. Check that both raters scored the overlap turns.",
            "expected_turns": sorted(overlap_turns),
            "prometheus_scored": sorted(prometheus_scores.keys()),
            "human_scored": sorted(human_scores.keys()),
        }

    p_ratings = [prometheus_scores[t] for t in common]
    h_ratings = [human_scores[t] for t in common]

    result = compute_kappa(p_ratings, h_ratings)
    result["turns_used"] = common
    result["missing_turns"] = sorted(overlap_turns - set(common))

    # Add action recommendation
    if result["target_met"]:
        result["action"] = (
            f"Kappa={result['kappa']} ≥ 0.61 — Prometheus 2 judge is validated. "
            "Proceed with full Prometheus scoring."
        )
    else:
        result["action"] = (
            f"Kappa={result['kappa']} < 0.61 — Prometheus 2 judge NOT validated. "
            "Replace Prometheus scoring with full human scoring on all 5 judge turns."
        )

    # Save to file
    _save_kappa(result)
    return result


def _save_kappa(result: dict):
    os.makedirs(os.path.dirname(_KAPPA_FILE), exist_ok=True)
    with open(_KAPPA_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)


def collect_human_scores_interactive(overlap_turns: list[int] = None) -> dict[int, bool]:
    """
    CLI helper: prompts the human rater to enter helpful/not-helpful for each
    overlap turn. Returns {turn_number: is_helpful}.

    This is called from sus_study.py when the human rater is present.
    """
    if overlap_turns is None:
        overlap_turns = sorted(KAPPA_OVERLAP_TURNS)

    print("\n" + "=" * 60)
    print("HUMAN RATER — Helpful Response Scoring (Cohen's Kappa)")
    print("=" * 60)
    print("For each turn shown, rate the chatbot response as helpful (y) or not (n).")
    print("=" * 60)

    human_scores: dict[int, bool] = {}
    for turn_num in overlap_turns:
        while True:
            answer = input(f"  Turn {turn_num} — Helpful? (y/n): ").strip().lower()
            if answer in ("y", "yes", "1"):
                human_scores[turn_num] = True
                break
            elif answer in ("n", "no", "0"):
                human_scores[turn_num] = False
                break
            else:
                print("  Please enter y or n.")
    return human_scores


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # Perfect agreement
    result = compute_kappa([True, True, False, False, True], [True, True, False, False, True])
    print(f"Perfect agreement: κ={result['kappa']} — {result['interpretation']}")

    # Substantial agreement
    result = compute_kappa([True, True, False, True, False], [True, False, False, True, False])
    print(f"4/5 agreement:     κ={result['kappa']} — {result['interpretation']}")

    # From score dicts
    prometheus = {2: True, 5: True, 7: False, 10: True, 14: True}
    human = {2: True, 5: False, 7: False, 10: True, 14: True}
    result2 = compute_kappa_from_scores(prometheus, human)
    print(f"\nFrom score dicts:  κ={result2['kappa']} — {result2['interpretation']}")
    print(f"Action: {result2['action']}")
