"""
report.py
---------
Report generator for the Chatbot Usability Testing Framework.

Generates a human-readable text report from all metric results,
CUS scores, Kappa results, and correlation analysis.

Output: results/reports/<chatbot>_report_<timestamp>.txt
"""

from __future__ import annotations
import os
import json
from datetime import datetime, timezone
from tabulate import tabulate

_REPORTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "results", "reports"
)
os.makedirs(_REPORTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _fmt(val, suffix="", decimals=2):
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}{suffix}"
    return f"{val}{suffix}"


def _section(title: str) -> str:
    line = "=" * 70
    return f"\n{line}\n  {title.upper()}\n{line}\n"


def _subsection(title: str) -> str:
    line = "-" * 50
    return f"\n{line}\n  {title}\n{line}\n"


# ---------------------------------------------------------------------------
# Per-metric report blocks
# ---------------------------------------------------------------------------
def _report_context_retention(result: dict) -> str:
    out = _subsection("Metric 1: Context Retention")
    out += f"  Score:          {_fmt(result.get('retention_rate_pct'), '%')} ({result.get('classification', 'N/A')})\n"
    out += f"  Normalized:     {_fmt(result.get('normalized_score'))} / 100\n"
    out += f"  Checkpoints:    {result.get('passed', 0)} / {result.get('total_checkpoints', 6)} passed\n"
    out += f"  Threshold:      Good ≥83%, Moderate 50–83%, Poor <50%\n"
    out += f"  Grounding:      ISO 9241-11 effectiveness\n"
    out += "\n  Checkpoint detail:\n"
    for cp in result.get("checkpoint_results", []):
        status = "PASS ✓" if cp["passed"] else "FAIL ✗"
        out += f"    Turn {cp['turn_number']:>2} (CP #{cp.get('checkpoint_id', '?')}): {status}\n"
    return out


def _report_response_time(result: dict) -> str:
    out = _subsection("Metric 2: Average Response Time")
    out += f"  Average:        {_fmt(result.get('average_ms'), ' ms')} ({result.get('classification', 'N/A')})\n"
    out += f"  Normalized:     {_fmt(result.get('normalized_score'))} / 100 (higher = faster)\n"
    out += f"  Min / Max:      {_fmt(result.get('min_ms'), ' ms')} / {_fmt(result.get('max_ms'), ' ms')}\n"
    out += f"  Valid turns:    {result.get('valid_turns', 0)}\n"
    out += f"  Threshold:      Good <1000 ms, Moderate 1000–10000 ms, Poor >10000 ms\n"
    out += f"  Grounding:      Nielsen (1993) Usability Engineering; ISO 9241-11 efficiency\n"
    return out


def _report_helpful_response(result: dict) -> str:
    out = _subsection("Metric 3: Helpful Response Ratio")
    out += f"  Score:          {_fmt(result.get('helpful_ratio_pct'), '%')} ({result.get('classification', 'N/A')})\n"
    out += f"  Normalized:     {_fmt(result.get('normalized_score'))} / 100\n"
    out += f"  Helpful turns:  {result.get('helpful_turns', 0)} / {result.get('total_turns', 15)}\n"
    out += f"  Threshold:      Good >80%, Moderate 60–80%, Poor <60%\n"
    out += f"  Grounding:      Følstad & Brandtzaeg (2017); Comm100 2026 benchmark\n"
    method_counts = {}
    for t in result.get("per_turn", []):
        m = t.get("scoring_method", "unknown")
        method_counts[m] = method_counts.get(m, 0) + 1
    if method_counts:
        out += "  Scoring breakdown:\n"
        for method, count in sorted(method_counts.items()):
            out += f"    {method}: {count} turns\n"
    return out


def _report_lexical_diversity(result: dict) -> str:
    out = _subsection("Metric 4: Lexical Diversity (MATTR-50)")
    out += f"  Average MATTR:  {_fmt(result.get('average_mattr'), '', 4)} ({result.get('classification', 'N/A')})\n"
    out += f"  Normalized:     {_fmt(result.get('normalized_score'))} / 100\n"
    out += f"  Scored turns:   {result.get('scored_turns', 0)}\n"
    out += f"  Excluded turns: {result.get('excluded_turns', 0)} (< 20 tokens)\n"
    out += f"  Threshold:      Pilot-calibrated (Covington & McFall 2010)\n"
    out += f"  Grounding:      MATTR = Moving Average Type-Token Ratio (W=50)\n"
    return out


def _report_error_rate(result: dict) -> str:
    out = _subsection("Metric 5: Error Rate")
    out += f"  Error rate:     {_fmt(result.get('error_rate_pct'), '%')} ({result.get('classification', 'N/A')})\n"
    out += f"  Normalized:     {_fmt(result.get('normalized_score'))} / 100 (higher = fewer errors)\n"
    out += f"  Breakdowns:     {result.get('breakdown_count', 0)} / {result.get('total_turns', 15)}\n"
    out += f"  Threshold:      Good <10%, Moderate 10–25%, Poor ≥25%\n"
    out += f"  Grounding:      Higashinaka et al. (2015) COLING; Comm100 2026 benchmark\n"
    taxonomy = result.get("taxonomy_summary", {})
    if taxonomy:
        out += "  Breakdown taxonomy (Higashinaka et al.):\n"
        for cat, count in sorted(taxonomy.items(), key=lambda x: -x[1]):
            out += f"    {cat}: {count}\n"
    return out


# ---------------------------------------------------------------------------
# CUS section
# ---------------------------------------------------------------------------
def _report_cus(cus_result: dict) -> str:
    out = _section("Composite Usability Score (CUS)")
    out += f"  CUS (equal-weighted):      {_fmt(cus_result.get('cus_equal'))} / 100  — {cus_result.get('classification', 'N/A')}\n"
    cus_reg = cus_result.get("cus_regression")
    if cus_reg is not None:
        out += f"  CUS (regression-weighted): {_fmt(cus_reg)} / 100\n"
    else:
        out += f"  CUS (regression-weighted): N/A (SUS study not yet complete)\n"

    scores = cus_result.get("scores", {})
    rows = [[m.replace("_", " ").title(), f"{v:.2f}"] for m, v in scores.items()]
    out += "\n" + tabulate(rows, headers=["Metric", "Normalized Score (0–100)"], tablefmt="simple") + "\n"

    if cus_result.get("regression_available") and cus_result.get("weights_used"):
        out += "\n  Regression weights:\n"
        for m, w in sorted(cus_result["weights_used"].items(), key=lambda x: -x[1]):
            out += f"    {m}: {w:.4f}\n"
    return out


# ---------------------------------------------------------------------------
# Kappa section
# ---------------------------------------------------------------------------
def _report_kappa(kappa_result: dict) -> str:
    if not kappa_result:
        return "\n[Kappa] Not yet computed.\n"
    out = _section("Cohen's Kappa — Prometheus 2 Validation")
    out += f"  κ = {_fmt(kappa_result.get('kappa'), '', 4)}  —  {kappa_result.get('interpretation', 'N/A')}\n"
    out += f"  Target met:  {'YES ✓' if kappa_result.get('target_met') else 'NO ✗ — replace with human scoring'}\n"
    out += f"  Turns used:  {kappa_result.get('turns_used', [])}\n"
    if kappa_result.get("action"):
        out += f"  Action:      {kappa_result['action']}\n"
    out += f"  Grounding:   Landis & Koch (1977) Biometrics 33(1)\n"
    return out


# ---------------------------------------------------------------------------
# Correlation section
# ---------------------------------------------------------------------------
def _report_correlation(analysis_result: dict) -> str:
    if not analysis_result:
        return "\n[Correlation] SUS study not yet complete.\n"
    corr = analysis_result.get("correlation", {})
    out = _section("Correlation: CUS vs SUS (Primary Result)")
    out += f"  n participants:  {corr.get('n', 'N/A')}\n"
    out += f"  Pearson r:       {_fmt(corr.get('pearson_r'))}  (p = {_fmt(corr.get('pearson_p'), '', 4)})\n"
    out += f"  Spearman ρ:      {_fmt(corr.get('spearman_rho'))}\n"
    out += f"  Significant:     {'Yes (p < 0.05)' if corr.get('statistically_significant') else 'No (p ≥ 0.05)'}\n"
    out += f"  Interpretation:  {corr.get('interpretation', 'N/A')}\n"
    if "warning" in corr:
        out += f"  ⚠ {corr['warning']}\n"

    reg = analysis_result.get("regression", {})
    if reg:
        out += _subsection("Multiple Regression — Metric Weights (Secondary Result)")
        out += f"  R² = {_fmt(reg.get('r_squared'), '', 4)}  |  Adj. R² = {_fmt(reg.get('adj_r_squared'), '', 4)}\n"
        rows = [[item["rank"], item["metric"].replace("_", " ").title(),
                 f"{item['coefficient']:.4f}", f"{item['weight']:.4f}"]
                for item in reg.get("metric_importance", [])]
        out += "\n" + tabulate(rows,
            headers=["Rank", "Metric", "Coefficient", "Norm. Weight"], tablefmt="simple") + "\n"
    return out


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def generate_report(
    chatbot_name: str,
    metric_results: dict,
    cus_result: dict = None,
    kappa_result: dict = None,
    analysis_result: dict = None,
    sus_aggregate: dict = None,
) -> str:
    """
    Generate a full usability report as a string and save it to disk.

    Parameters
    ----------
    chatbot_name : str
    metric_results : dict  {metric_name: result_dict}
    cus_result : dict      from scoring.compute_cus()
    kappa_result : dict    from kappa_calculator.compute_kappa_from_scores()
    analysis_result : dict from analysis.run_full_analysis()
    sus_aggregate : dict   from sus_study.aggregate_sus()

    Returns
    -------
    str  The full report text.
    """
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = []
    lines.append("=" * 70)
    lines.append(f"  CHATBOT USABILITY TESTING FRAMEWORK — RESULTS REPORT")
    lines.append(f"  Chatbot:   {chatbot_name.upper()}")
    lines.append(f"  Generated: {ts}")
    lines.append("=" * 70)

    lines.append(_section("Executive Summary"))
    if cus_result:
        lines.append(
            f"  Composite Usability Score (CUS): "
            f"{_fmt(cus_result.get('cus_equal'))} / 100  —  {cus_result.get('classification', 'N/A')}"
        )
    if sus_aggregate and sus_aggregate.get("mean_sus") is not None:
        lines.append(
            f"  Human SUS Score (mean, n={sus_aggregate.get('n')}): "
            f"{_fmt(sus_aggregate.get('mean_sus'))} / 100  —  {sus_aggregate.get('classification', 'N/A')}"
        )
    if analysis_result and analysis_result.get("correlation", {}).get("pearson_r") is not None:
        corr = analysis_result["correlation"]
        lines.append(
            f"  CUS–SUS Correlation: r = {_fmt(corr['pearson_r'])} "
            f"({'significant' if corr.get('statistically_significant') else 'not significant'})"
        )

    # Per-metric sections
    lines.append(_section("Per-Metric Results"))
    if "context_retention" in metric_results:
        lines.append(_report_context_retention(metric_results["context_retention"]))
    if "response_time" in metric_results:
        lines.append(_report_response_time(metric_results["response_time"]))
    if "helpful_response" in metric_results:
        lines.append(_report_helpful_response(metric_results["helpful_response"]))
    if "lexical_diversity" in metric_results:
        lines.append(_report_lexical_diversity(metric_results["lexical_diversity"]))
    if "error_rate" in metric_results:
        lines.append(_report_error_rate(metric_results["error_rate"]))

    if cus_result:
        lines.append(_report_cus(cus_result))
    if kappa_result:
        lines.append(_report_kappa(kappa_result))
    if analysis_result:
        lines.append(_report_correlation(analysis_result))

    # Tuning recommendations
    lines.append(_section("Tuning Recommendations"))
    if metric_results:
        for metric_key, label, suggestion in [
            ("context_retention", "Context Retention",
             "Increase context window size; add RAG step to re-inject earlier user facts."),
            ("response_time", "Response Time",
             "Use smaller/faster model; add response caching; reduce prompt length."),
            ("helpful_response", "Helpful Response Ratio",
             "Improve system prompt / intent classification; add few-shot examples."),
            ("lexical_diversity", "Lexical Diversity",
             "Add conciseness instructions; penalize hedge phrases in generation settings."),
            ("error_rate", "Error Rate",
             "Add input normalization (typo correction); add clarification fallback intent."),
        ]:
            r = metric_results.get(metric_key, {})
            clf = r.get("classification", "N/A")
            if clf in ("Poor", "Moderate"):
                lines.append(f"  ⚠ {label} ({clf}): {suggestion}")

    lines.append("\n" + "=" * 70)
    lines.append("  END OF REPORT")
    lines.append("=" * 70 + "\n")

    report_text = "\n".join(lines)

    # Save to file
    ts_file = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    filename = f"{chatbot_name}_report_{ts_file}.txt"
    filepath = os.path.join(_REPORTS_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"  [Report] Saved to: {filepath}")

    return report_text


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    sample_metrics = {
        "context_retention": {
            "passed": 4, "total_checkpoints": 6,
            "retention_rate_pct": 66.7, "classification": "Moderate",
            "normalized_score": 66.7,
            "checkpoint_results": [
                {"turn_number": t, "checkpoint_id": i+1, "passed": i < 4}
                for i, t in enumerate([5, 10, 15, 20, 25, 30])
            ],
        },
        "response_time": {
            "average_ms": 850.0, "classification": "Good",
            "normalized_score": 91.5, "min_ms": 120.0, "max_ms": 2100.0,
            "valid_turns": 15, "per_turn_latencies": [],
        },
        "helpful_response": {
            "helpful_turns": 11, "total_turns": 15,
            "helpful_ratio_pct": 73.3, "classification": "Moderate",
            "normalized_score": 73.3, "per_turn": [],
        },
        "lexical_diversity": {
            "average_mattr": 0.72, "classification": "Good",
            "normalized_score": 72.0, "scored_turns": 10,
            "excluded_turns": 2, "per_turn": [],
        },
        "error_rate": {
            "breakdown_count": 3, "total_turns": 15,
            "error_rate_pct": 20.0, "classification": "Moderate",
            "normalized_score": 80.0,
            "taxonomy_summary": {"not_understandable": 2, "not_relevant": 1},
            "per_turn": [],
        },
    }

    from scoring import compute_cus
    cus = compute_cus(sample_metrics)
    report = generate_report("rasa", sample_metrics, cus_result=cus)
    print(report[:1500])
