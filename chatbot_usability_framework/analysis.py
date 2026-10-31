"""
analysis.py
-----------
Correlation and regression analysis for the Chatbot Usability Framework.

Primary result:
    Pearson r between CUS (automated) and SUS (human) scores.
    Also computes Spearman ρ as a non-parametric cross-check.

Secondary result:
    Multiple linear regression of the five metric scores against SUS scores.
    Regression coefficients become the regression-weighted CUS weights.

Statistical requirements (Cohen 1988):
    n ≥ 30 participants required.
    At n=30, critical Pearson r (α=0.05, two-tailed) = 0.361.
    Power to detect r=0.5 (medium effect) exceeds 80%.

Citations:
    Cohen, J. (1988). Statistical Power Analysis for the Behavioral Sciences, 2nd ed.
    Pearson/Spearman: scipy.stats
    OLS regression: numpy.linalg.lstsq
"""

from __future__ import annotations
import json
import os
import math
import numpy as np
from scipy import stats

from scoring import save_regression_weights

_ANALYSIS_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "analysis_results.json"
)


# ---------------------------------------------------------------------------
# Correlation analysis (primary result)
# ---------------------------------------------------------------------------
def compute_correlation(
    cus_scores: list[float],
    sus_scores: list[float],
) -> dict:
    """
    Compute Pearson r and Spearman ρ between CUS and SUS scores.

    Parameters
    ----------
    cus_scores : list[float]
        Automated Composite Usability Score for each participant (0–100).
        All participants used the same CUS (their chatbot interaction is
        scored by the same metric run; variance comes from the SUS).
        In practice: use the CUS of the chatbot the participant tested.
    sus_scores : list[float]
        Human SUS scores for each participant (0–100), same order.

    Returns
    -------
    dict:
        n                 : int
        pearson_r         : float
        pearson_p         : float
        spearman_rho      : float
        spearman_p        : float
        statistically_significant : bool  (p < 0.05)
        interpretation    : str
        critical_r_n30    : float  = 0.361
    """
    n = len(cus_scores)
    if n != len(sus_scores):
        raise ValueError("cus_scores and sus_scores must be the same length.")
    if n < 3:
        raise ValueError(f"Need at least 3 observations, got {n}.")

    arr_cus = np.array(cus_scores, dtype=float)
    arr_sus = np.array(sus_scores, dtype=float)

    pearson_r, pearson_p = stats.pearsonr(arr_cus, arr_sus)
    spearman_rho, spearman_p = stats.spearmanr(arr_cus, arr_sus)

    significant = pearson_p < 0.05

    # Interpretation
    abs_r = abs(pearson_r)
    if abs_r >= 0.7:
        strength = "strong"
    elif abs_r >= 0.5:
        strength = "moderate"
    elif abs_r >= 0.3:
        strength = "weak"
    else:
        strength = "negligible"

    direction = "positive" if pearson_r >= 0 else "negative"
    interpretation = (
        f"{strength.capitalize()} {direction} correlation (r={pearson_r:.3f}, "
        f"p={pearson_p:.4f}). "
        f"{'Statistically significant.' if significant else 'NOT statistically significant (p ≥ 0.05).'}"
    )

    warning = None
    if n < 30:
        warning = (
            f"WARNING: n={n} is below the recommended minimum of 30. "
            "Critical r at n=30 is 0.361; your critical r is higher, "
            "so even a real medium effect may not reach significance."
        )

    result = {
        "n": n,
        "pearson_r": round(float(pearson_r), 4),
        "pearson_p": round(float(pearson_p), 6),
        "spearman_rho": round(float(spearman_rho), 4),
        "spearman_p": round(float(spearman_p), 6),
        "statistically_significant": bool(significant),
        "interpretation": interpretation,
        "critical_r_n30": 0.361,
    }
    if warning:
        result["warning"] = warning
    return result


# ---------------------------------------------------------------------------
# Multiple linear regression (secondary result — metric weights)
# ---------------------------------------------------------------------------
METRIC_ORDER = [
    "context_retention",
    "response_time",
    "helpful_response",
    "lexical_diversity",
    "error_rate",
]


def compute_regression(
    metric_scores_matrix: list[list[float]],
    sus_scores: list[float],
) -> dict:
    """
    Multiple linear regression: SUS ~ β0 + β1×m1 + ... + β5×m5

    Parameters
    ----------
    metric_scores_matrix : list[list[float]]
        Shape (n_participants, 5). Each row is one participant's
        five metric scores for the chatbot they tested.
        Order: [context_retention, response_time, helpful_response,
                lexical_diversity, error_rate]
    sus_scores : list[float]
        SUS score for each participant. Same order as rows.

    Returns
    -------
    dict:
        intercept    : float
        coefficients : {metric_name: float}
        weights      : {metric_name: float}   — normalised to sum to 1.0
        r_squared    : float
        adj_r_squared: float
        metric_importance: list[{metric, coef, weight, rank}]
    """
    X_raw = np.array(metric_scores_matrix, dtype=float)
    y = np.array(sus_scores, dtype=float)
    n, p = X_raw.shape

    # Add intercept column
    X = np.column_stack([np.ones(n), X_raw])

    # OLS: β = (X'X)^-1 X'y  (via lstsq for numerical stability)
    coeffs, residuals, rank, sv = np.linalg.lstsq(X, y, rcond=None)

    intercept = float(coeffs[0])
    coef_dict = {name: float(coeffs[i + 1]) for i, name in enumerate(METRIC_ORDER)}

    # R²
    y_pred = X @ coeffs
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / max(1, n - p - 1)

    # Normalize positive coefficients to weights summing to 1.0
    # Use abs value for ranking, then re-normalize positives only
    # (negative coefficients are set to a small minimum weight of 0.05)
    MIN_WEIGHT = 0.05
    raw_weights = {}
    for name in METRIC_ORDER:
        c = coef_dict[name]
        raw_weights[name] = max(MIN_WEIGHT, c) if c > 0 else MIN_WEIGHT

    total_w = sum(raw_weights.values())
    weights = {name: round(w / total_w, 4) for name, w in raw_weights.items()}

    # Importance ranking
    importance = sorted(
        [{"metric": m, "coefficient": round(coef_dict[m], 4), "weight": weights[m]}
         for m in METRIC_ORDER],
        key=lambda x: x["weight"],
        reverse=True,
    )
    for rank_i, item in enumerate(importance, 1):
        item["rank"] = rank_i

    result = {
        "intercept": round(intercept, 4),
        "coefficients": {m: round(v, 4) for m, v in coef_dict.items()},
        "weights": weights,
        "r_squared": round(r2, 4),
        "adj_r_squared": round(adj_r2, 4),
        "metric_importance": importance,
        "n": n,
    }
    return result


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------
def run_full_analysis(
    cus_scores: list[float],
    sus_scores: list[float],
    metric_scores_matrix: list[list[float]] = None,
    chatbot_name: str = "combined",
) -> dict:
    """
    Run the complete analysis: correlation + optional regression.
    Saves results to results/analysis_results.json.

    Parameters
    ----------
    cus_scores : list[float]
    sus_scores : list[float]
    metric_scores_matrix : list[list[float]], optional
        If provided, runs regression and saves weights for CUS.
    chatbot_name : str

    Returns
    -------
    dict  Full analysis result.
    """
    print(f"\n[Analysis] Running for chatbot={chatbot_name!r}, n={len(sus_scores)}")

    result = {"chatbot_name": chatbot_name}

    # Correlation
    corr = compute_correlation(cus_scores, sus_scores)
    result["correlation"] = corr
    print(f"  Pearson r = {corr['pearson_r']}  p = {corr['pearson_p']}")
    print(f"  Spearman ρ = {corr['spearman_rho']}")
    print(f"  {corr['interpretation']}")
    if "warning" in corr:
        print(f"  ⚠ {corr['warning']}")

    # Regression (secondary result)
    if metric_scores_matrix is not None:
        reg = compute_regression(metric_scores_matrix, sus_scores)
        result["regression"] = reg
        print(f"\n  Regression R² = {reg['r_squared']}")
        print("  Metric weights (regression-derived):")
        for item in reg["metric_importance"]:
            print(f"    #{item['rank']} {item['metric']}: weight={item['weight']} (coef={item['coefficient']})")

        # Save regression weights so scoring.py can use them
        save_regression_weights(reg["weights"], metadata={
            "r_squared": reg["r_squared"],
            "n": reg["n"],
            "chatbot_name": chatbot_name,
        })

    # Save full analysis
    os.makedirs(os.path.dirname(_ANALYSIS_FILE), exist_ok=True)
    with open(_ANALYSIS_FILE, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"\n  [Analysis] Results saved to: {_ANALYSIS_FILE}")

    return result


# ---------------------------------------------------------------------------
# Quick smoke-test with synthetic data
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import random
    random.seed(99)

    n = 30
    # Simulate CUS scores around 65 (Rasa) with some noise
    cus = [max(0, min(100, 65 + random.gauss(0, 8))) for _ in range(n)]
    # SUS moderately correlated with CUS (r ≈ 0.6)
    sus = [max(0, min(100, c * 0.6 + random.gauss(30, 8))) for c in cus]
    # Metric matrix: 5 metrics, some correlated with SUS
    metrics = []
    for i in range(n):
        metrics.append([
            max(0, min(100, sus[i] * 0.9 + random.gauss(5, 6))),   # context_retention (high corr)
            max(0, min(100, 90 + random.gauss(0, 5))),               # response_time (low corr)
            max(0, min(100, sus[i] * 0.7 + random.gauss(15, 8))),   # helpful_response
            max(0, min(100, sus[i] * 0.5 + random.gauss(30, 10))),  # lexical_diversity
            max(0, min(100, sus[i] * 0.8 + random.gauss(10, 7))),   # error_rate
        ])

    result = run_full_analysis(cus, sus, metrics, chatbot_name="rasa")
    print("\nTop metric by regression weight:", result["regression"]["metric_importance"][0])
