"""
metrics.py — Automated Usability Testing Framework
Scores all 5 metrics from conversation logs.
DESIGN PRINCIPLE: domain-agnostic, not overfitted to any script topic.
Each metric scores WHAT was measured, not WHERE (no hardcoded turn numbers
in the core logic — turn numbers only appear in script-specific config that
is kept separate and overridable).
"""

import json, re
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

LOGS_DIR = Path("logs")

# ══════════════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ══════════════════════════════════════════════════════════════════════

def _text(val):
    """Normalise any cell value to a clean lowercase string."""
    return str(val).strip().lower() if val and str(val) not in ["nan","None",""] else ""


def _word_count(text):
    return len(str(text).split())


def _is_empty_or_error(text):
    t = _text(text)
    return (not t
            or t.startswith("[error")
            or t in ["[no response]", "nan", "none", ""]
            or t == "i am a bot, powered by rasa.")


# ── Universal non-answer detector (domain-agnostic) ───────────────────
# These phrases mean the bot REFUSED or DEFLECTED regardless of topic
NON_ANSWER_SIGNALS = [
    r"i('m| am) (just |only )?an? (ai|language model|llm|chatbot|bot)",
    r"(i |i'm )?(cannot|can'?t|unable to|not able to) (help|assist|provide|answer|give)",
    r"(beyond|outside|not within) (my )?(capabilities|scope|knowledge|training)",
    r"(please |i recommend you )?(consult|see|contact|speak (to|with)) (a |an )?(professional|doctor|lawyer|expert|specialist)",
    r"i (don'?t|do not) have (the ability|access|information|real.?time)",
    r"(this (is|falls) )?(beyond|outside) what i",
    r"i('m| am) not (qualified|trained|able)",
]

def is_non_answer(response):
    """Returns True if bot refused/deflected regardless of topic."""
    r = _text(response)
    return any(re.search(p, r) for p in NON_ANSWER_SIGNALS)


# ── Memory denial detector (CR-specific but phrasing-agnostic) ────────
MEMORY_DENIAL_PATTERNS = [
    r"(this|we('ve| have)?) (is |are )?(just |only )?(the |a )?(beginning|start) of (our|this) (conversation|chat|session)",
    r"(i |i've |i have )?(haven'?t|have not|didn'?t|did not) (discussed|mentioned|talked about|said|told)",
    r"(you |i )(haven'?t|have not|didn'?t|did not) (provided|given|shared|mentioned|told me) (any(thing)?|your)",
    r"(no|without) (prior|previous|earlier) (context|conversation|information|discussion|history)",
    r"(i don'?t|i do not|i have no) (have |)(any )?(prior|previous|earlier)? (information|context|knowledge) about (you|your)",
    r"(this|that'?s) (your|the) first (message|question|time)",
    r"(we|i) (haven'?t|have not) (spoken|talked|chatted) before",
    r"(first|beginning of|start of) (our|this) (conversation|session|chat|exchange)",
    r"haven'?t (had|been (able to) have) (a |any )?(conversation|discussion|chat)",
]

def is_memory_denial(response):
    """Returns True if bot explicitly says it has no memory of prior turns."""
    r = _text(response)
    return any(re.search(p, r) for p in MEMORY_DENIAL_PATTERNS)


# ══════════════════════════════════════════════════════════════════════
# METRIC 1: CONTEXT RETENTION
# Checks if bot remembered a USER FACT planted earlier in conversation.
# Domain-agnostic: fact type and violation type defined per-script via
# the CSV columns 'fact_being_tested' and 'pass_criteria'.
# Scoring uses three independent signals, not just keyword matching.
# ══════════════════════════════════════════════════════════════════════

def score_cr(df):
    """
    Scores context retention checkpoints.
    Requires CSV columns: turn_type (value 'CHECKPOINT'),
    fact_being_tested, pass_criteria.
    Falls back gracefully if columns are missing.
    """
    if "turn_type" not in df.columns:
        return {"metric": "CR", "score": None, "band": "NO DATA",
                "reason": "CSV missing turn_type column"}

    checkpoints = df[df["turn_type"].str.upper() == "CHECKPOINT"].copy()
    if checkpoints.empty:
        return {"metric": "CR", "score": None, "band": "NO DATA",
                "reason": "No CHECKPOINT rows found"}

    results = []
    for _, row in checkpoints.iterrows():
        response  = str(row.get("response", ""))
        turn_num  = int(row.get("turn_number", 0))
        criteria  = _text(row.get("pass_criteria", ""))
        fact      = _text(row.get("fact_being_tested", ""))

        passed, reason = _evaluate_cr_checkpoint(response, criteria, fact)
        results.append({"turn": turn_num, "pass": passed, "reason": reason,
                        "response_snippet": response[:100]})

    passed = sum(1 for r in results if r["pass"])
    total  = len(results)
    rate   = round(passed / total * 100, 1)
    band   = "GOOD" if rate >= 83 else ("MODERATE" if rate >= 50 else "POOR")

    print(f"\n  [CR] {passed}/{total} checkpoints passed = {rate}% → {band}")
    for r in results:
        icon = "✓" if r["pass"] else "✗"
        print(f"    Turn {r['turn']}: {icon} {r['reason']}")
        print(f"           {r['response_snippet'][:80]}")

    return {"metric": "CR", "passed": passed, "total": total,
            "score": rate, "band": band, "details": results}


def _evaluate_cr_checkpoint(response, criteria, fact):
    """
    Three-signal evaluation — order matters, first match wins.
    Signal 1: Empty / error → FAIL
    Signal 2: Memory denial → FAIL
    Signal 3: Pass-criteria check from CSV oracle
    """
    # Signal 1
    if _is_empty_or_error(response):
        return False, "EMPTY_OR_ERROR"

    # Signal 2
    if is_memory_denial(response):
        return False, "MEMORY_DENIAL"

    # Signal 3 — use pass_criteria from CSV if available
    if criteria:
        result, reason = _check_criteria(response, criteria)
        return result, reason

    # No criteria defined — cannot score, mark as UNSCORED
    return None, "NO_CRITERIA_DEFINED"


def _check_criteria(response, criteria):
    """
    Interprets the pass_criteria string from the CSV.
    Supports two modes:
    Mode A — negation check:  criteria contains 'FAIL if' or 'must not'
    Mode B — positive check:  criteria contains 'must', 'should', 'response must'
    Falls back to basic keyword presence if no mode matched.
    """
    r = _text(response)
    c = criteria.lower()

    # Mode A: FAIL conditions defined in criteria
    if "fail if" in c:
        fail_part = c.split("fail if")[-1].strip()
        # extract key terms from the fail condition
        fail_terms = [t.strip().strip(".,") for t in re.split(r'\bor\b|\band\b|,', fail_part) if len(t.strip()) > 3]
        for term in fail_terms:
            # remove common filler from criteria text
            clean = re.sub(r'\b(any|the|a|an|is|are|was|were|it|in|on|at)\b', '', term).strip()
            if clean and clean in r:
                return False, f"FAIL_CONDITION_MET: '{clean}'"
        return True, "PASS — no fail conditions triggered"

    # Mode B: positive requirements
    if any(kw in c for kw in ["must", "should", "response must", "must contain"]):
        req_part = re.sub(r'(must|should|response must contain|must contain)', '', c).strip()
        req_terms = [t.strip().strip(".,") for t in re.split(r'\bor\b|\band\b|,', req_part) if len(t.strip()) > 3]
        # at least half the positive requirements must be met
        met = sum(1 for t in req_terms
                  if re.sub(r'\b(any|the|a|an)\b', '', t).strip() in r)
        threshold = max(1, len(req_terms) // 2)
        if met >= threshold:
            return True, f"PASS — {met}/{len(req_terms)} requirements met"
        return False, f"FAIL — only {met}/{len(req_terms)} requirements met"

    return True, "PASS — no violations detected (no structured criteria)"


# ══════════════════════════════════════════════════════════════════════
# METRIC 2: RESPONSE TIME
# Domain-agnostic: just timestamps, works on any topic.
# Handles outliers, errors, burst turns, cold-start bias.
# ══════════════════════════════════════════════════════════════════════

def score_rt(df):
    """
    Scores response time across ALL turns in the log.
    RT is script-agnostic — computed on every turn automatically.
    Citation: Nielsen (1993); Følstad & Skjuve (2019).
    Thresholds: Good < 1000ms, Moderate 1000–10000ms, Poor > 10000ms.
    """
    all_lat   = df["latency_ms"]
    valid     = all_lat[all_lat > 0]
    errors    = all_lat[all_lat <= 0]

    if valid.empty:
        return {"metric": "RT", "score": None, "band": "NO DATA",
                "error_turns": int(len(errors))}

    avg    = round(float(valid.mean()), 2)
    median = round(float(valid.median()), 2)
    mn     = round(float(valid.min()), 2)
    mx     = round(float(valid.max()), 2)
    sd     = round(float(valid.std()), 2)

    # Outlier detection: turns > mean + 2*SD flagged separately
    outlier_thresh = avg + 2 * sd
    outliers = valid[valid > outlier_thresh]
    avg_no_outliers = round(float(valid[valid <= outlier_thresh].mean()), 2) if len(valid) > len(outliers) else avg

    band = "GOOD" if avg < 1000 else ("MODERATE" if avg <= 10000 else "POOR")

    print(f"\n  [RT] Mean: {avg}ms | Median: {median}ms | SD: {sd}ms → {band}")
    print(f"    Min: {mn}ms | Max: {mx}ms | Errors excluded: {len(errors)}")
    print(f"    Outliers (>{round(outlier_thresh)}ms): {len(outliers)} turns")
    print(f"    Mean without outliers: {avg_no_outliers}ms")

    # Per-script breakdown
    for script, grp in df[df["latency_ms"] > 0].groupby("script"):
        print(f"    {script}: {round(grp['latency_ms'].mean(),1)}ms avg ({len(grp)} turns)")

    # Burst turn analysis if turn_type column exists
    if "turn_type" in df.columns:
        burst = df[df["turn_type"].str.contains("burst", case=False, na=False) & (df["latency_ms"] > 0)]
        normal = df[~df["turn_type"].str.contains("burst", case=False, na=False) & (df["latency_ms"] > 0)]
        if not burst.empty and not normal.empty:
            b_avg = round(float(burst["latency_ms"].mean()), 2)
            n_avg = round(float(normal["latency_ms"].mean()), 2)
            degradation = round((b_avg - n_avg) / n_avg * 100, 1)
            print(f"    Burst latency: {b_avg}ms | Non-burst: {n_avg}ms | Degradation: {degradation}%")

    return {"metric": "RT", "avg_ms": avg, "median_ms": median,
            "min_ms": mn, "max_ms": mx, "sd_ms": sd,
            "avg_no_outliers_ms": avg_no_outliers,
            "outlier_turns": int(len(outliers)),
            "error_turns": int(len(errors)),
            "turns_measured": int(len(valid)),
            "score": avg, "band": band}


# ══════════════════════════════════════════════════════════════════════
# METRIC 3: MATTR — Lexical Diversity
# Domain-agnostic: vocabulary richness is independent of topic.
# Runs on ALL turns across ALL scripts.
# Thresholds set from pilot data (not assumed).
# Citation: Covington & McFall (2010).
# ══════════════════════════════════════════════════════════════════════

# SET THESE FROM YOUR PILOT RUN (Cell 8 in Colab setup)
MATTR_GOOD_THRESHOLD     = 0.69   # ← replace with your pilot output
MATTR_MODERATE_THRESHOLD = 0.59   # ← replace with your pilot output


def score_mattr(df):
    """
    Scores lexical diversity across ALL logged turns.
    Excludes: errors, too-short responses, degenerate repetition.
    Stratifies by response length to avoid length-bias in reporting.
    """
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    def _mattr(text, window=50):
        r = str(text).strip().lower()
        if _is_empty_or_error(r): return None, "ERROR"
        try:    tokens = nltk.word_tokenize(r)
        except: tokens = r.split()
        if len(tokens) < 20:    return None, f"SHORT({len(tokens)}w)"
        if len(set(tokens)) / len(tokens) < 0.1: return None, "DEGENERATE"
        if len(tokens) < window:
            return round(len(set(tokens)) / len(tokens), 4), "OK_NARROW"
        scores = [len(set(tokens[i:i+window]))/window for i in range(len(tokens)-window+1)]
        return round(sum(scores)/len(scores), 4), "OK"

    records = []
    for _, row in df.iterrows():
        score, status = _mattr(row["response"])
        wc = _word_count(row["response"])
        records.append({
            "score":   score,
            "status":  status,
            "script":  row.get("script", "?"),
            "wc":      wc,
        })

    scored   = [r for r in records if r["score"] is not None]
    excluded = [r for r in records if r["score"] is None]

    if not scored:
        return {"metric": "MATTR", "score": None,
                "band": "ALL_EXCLUDED", "turns_excluded": len(excluded)}

    avg  = round(sum(r["score"] for r in scored) / len(scored), 4)
    mn   = round(min(r["score"] for r in scored), 4)
    mx   = round(max(r["score"] for r in scored), 4)
    band = ("GOOD"     if avg >= MATTR_GOOD_THRESHOLD else
            "MODERATE" if avg >= MATTR_MODERATE_THRESHOLD else "POOR")

    excl_reasons = Counter(r["status"] for r in excluded)

    print(f"\n  [MATTR] Scored: {len(scored)} | Excluded: {len(excluded)}")
    print(f"    Avg: {avg} | Min: {mn} | Max: {mx} → {band}")
    print(f"    Thresholds: Good>={MATTR_GOOD_THRESHOLD} | Moderate>={MATTR_MODERATE_THRESHOLD}")
    for reason, count in excl_reasons.items():
        print(f"    Excluded ({reason}): {count}")

    # Per-script breakdown
    by_script = defaultdict(list)
    for r in scored: by_script[r["script"]].append(r["score"])
    for script, sc in by_script.items():
        print(f"    {script}: {round(sum(sc)/len(sc),4)} avg ({len(sc)} turns)")

    return {"metric": "MATTR", "avg": avg, "min": mn, "max": mx,
            "turns_scored": len(scored), "turns_excluded": len(excluded),
            "exclusion_reasons": dict(excl_reasons),
            "score": avg, "band": band}


# ══════════════════════════════════════════════════════════════════════
# METRIC 4: HELPFUL RESPONSE RATIO
# Uses judge_type column from CSV to route each turn to the right scorer.
# Rule-based turns: script supplies pass_criteria → evaluated generically.
# Prometheus turns: scored by LLM judge (see prometheus.py).
# Domain-agnostic: criteria come from CSV, not hardcoded here.
# ══════════════════════════════════════════════════════════════════════

def score_hr(df):
    """
    Routes each HR turn to the right scoring method based on judge_type.
    Rule-based: checks pass_criteria from CSV using generic NLP signals.
    Prometheus: reads pre-computed scores from prometheus_scores.json.
    """
    if "script" in df.columns:
        hr_df = df[df["script"] == "HR"].copy()
    else:
        hr_df = df.copy()

    if hr_df.empty:
        return {"metric": "HR", "score": None, "band": "NO DATA"}

    if "judge_type" not in hr_df.columns:
        return {"metric": "HR", "score": None, "band": "NO DATA",
                "reason": "CSV missing judge_type column"}

    results = []

    # Load Prometheus scores if available
    prometheus_scores = _load_prometheus_scores(hr_df["bot"].iloc[0] if "bot" in hr_df.columns else "")

    for _, row in hr_df.iterrows():
        t           = int(row["turn_number"])
        response    = str(row.get("response", ""))
        judge_type  = _text(row.get("judge_type", "rule"))
        criteria    = _text(row.get("pass_criteria", ""))

        if _is_empty_or_error(response):
            results.append({"turn": t, "pass": False,
                            "reason": "EMPTY_OR_ERROR", "method": "auto"})
            continue

        if is_non_answer(response) and "rule" in judge_type:
            results.append({"turn": t, "pass": False,
                            "reason": "NON_ANSWER_DEFLECTION", "method": "auto"})
            continue

        if "prometheus" in judge_type:
            # Read from pre-computed prometheus scores
            p_result = prometheus_scores.get(str(t))
            if p_result is not None:
                results.append({"turn": t, "pass": bool(p_result["pass"]),
                                "reason": f"PROMETHEUS: {p_result['verdict'][:60]}",
                                "method": "prometheus"})
            else:
                results.append({"turn": t, "pass": None,
                                "reason": "PROMETHEUS_NOT_RUN",
                                "method": "prometheus_pending"})
            continue

        # Rule-based: use pass_criteria from CSV
        if criteria:
            passed, reason = _check_criteria(response, criteria)
            results.append({"turn": t, "pass": passed,
                            "reason": reason, "method": "rule"})
        else:
            # No criteria: use generic helpfulness signals
            passed, reason = _generic_helpfulness(response)
            results.append({"turn": t, "pass": passed,
                            "reason": reason, "method": "generic"})

    # Only score turns with a definite result
    scored   = [r for r in results if r["pass"] is not None]
    pending  = [r for r in results if r["pass"] is None]
    passed_n = sum(1 for r in scored if r["pass"])
    total    = len(scored)
    rate     = round(passed_n / total * 100, 1) if total > 0 else 0
    band     = "GOOD" if rate > 80 else ("MODERATE" if rate >= 60 else "POOR")

    print(f"\n  [HR] {passed_n}/{total} scored turns passed = {rate}% → {band}")
    if pending:
        print(f"    {len(pending)} Prometheus turns pending — run prometheus.py")
    for r in results:
        icon = ("✓" if r["pass"] else ("✗" if r["pass"] is False else "?"))
        print(f"    Turn {r['turn']}: {icon} [{r['method']}] {r['reason'][:70]}")

    return {"metric": "HR", "passed": passed_n, "total": total,
            "pending_prometheus": len(pending),
            "score": rate, "band": band, "details": results}


def _load_prometheus_scores(bot_name):
    """Load pre-computed Prometheus scores for this bot."""
    p_file = LOGS_DIR / "prometheus_scores.json"
    if not p_file.exists():
        return {}
    with open(p_file) as f:
        all_scores = json.load(f)
    return all_scores.get(bot_name, {})


def _generic_helpfulness(response):
    """
    Fallback helpfulness check when no pass_criteria is defined.
    Uses domain-agnostic signals:
    - Response is substantive (>30 words)
    - Not a non-answer
    - Contains at least one concrete signal (number, named entity, action verb)
    """
    r = str(response)
    wc = _word_count(r)

    if wc < 30:
        return False, "TOO_SHORT_FOR_HELPFULNESS"

    # Check for concrete content signals
    has_number    = bool(re.search(r'\b\d+[\d%$.,]*\b', r))
    has_list      = r.count('\n') >= 2 or r.count('•') >= 1 or r.count('-') >= 2
    has_action    = bool(re.search(
        r'\b(try|use|do|check|visit|apply|create|start|open|go|get|make|take|consider|ensure)\b',
        r.lower()))
    has_specific  = wc > 50  # longer responses more likely to be substantive

    signals = sum([has_number, has_list, has_action, has_specific])
    if signals >= 2:
        return True, f"GENERIC_PASS ({signals}/4 helpfulness signals)"
    return False, f"GENERIC_FAIL ({signals}/4 helpfulness signals)"


# ══════════════════════════════════════════════════════════════════════
# METRIC 5: ERROR RATE
# Domain-agnostic: scores how the bot handles malformed / noisy input.
# Uses noise_type column from CSV to apply type-appropriate checks.
# Three severity levels: hard breakdown, soft breakdown, recovered.
# ══════════════════════════════════════════════════════════════════════

# True breakdown signals (universal, topic-independent)
HARD_BREAKDOWN_PATTERNS = [
    r"i (don'?t|do not) understand (you|that|this|what|your)",
    r"(i |i'm )?(completely |totally )?lost",
    r"(that|this) (doesn'?t|does not) (make sense|compute)",
    r"i cannot (process|understand|parse|handle) (that|this|your)",
    r"(please|could you) (rephrase|clarify|restate|say that again)",
    r"(i|i'm) (not sure|unsure) what (you('re| are)|that)",
    r"^(huh|what|eh|sorry)\??$",
    # Rasa-specific defaults
    r"^(bye|great, carry on!?|here is something to cheer you up)$",
    r"^i am a bot, powered by rasa\.$",
]

# LLM hedge signals — bot is uncertain but TRYING to help (not a breakdown)
HEDGE_BUT_HELPING_PATTERNS = [
    r"(i('m| am) not sure (which|what|which one) (you'?re?|you are) referring to)",
    r"(could you (please )?(clarify|specify|provide more|tell me which))",
    r"(i('d| would) (be happy|like) to help)",
    r"(let me (help|try|see|check))",
    r"(i understand (you'?re?|you are|your))",
    r"(i'll do my best)",
]

# Out-of-scope: bot should DECLINE not attempt
OUT_OF_SCOPE_DECLINE_SIGNALS = [
    r"(i('m| am) (an |a )?(ai|language model|chatbot))",
    r"(i (can'?t|cannot|don'?t have the ability to) (book|order|purchase|buy|access|retrieve|check|look up))",
    r"(that'?s? (not|outside) (something |)(i can|within) (help|my|do))",
    r"(i (don'?t|do not) have (real.?time|live|current|access to))",
    r"(not (something|a service|a feature) i (can|offer|provide))",
]

def score_er(df):
    """
    Scores error rate on noisy/adversarial inputs.
    Requires CSV columns: noise_type, pass_criteria.
    Excludes contradiction T1 rows (context-setting turns, not scored).
    Reports breakdown by noise category.
    """
    if "script" in df.columns:
        er_df = df[df["script"] == "ER"].copy()
    else:
        er_df = df.copy()

    if er_df.empty:
        return {"metric": "ER", "score": None, "band": "NO DATA"}

    if "noise_type" not in er_df.columns:
        return {"metric": "ER", "score": None, "band": "NO DATA",
                "reason": "CSV missing noise_type column"}

    # Exclude T1 rows (contradiction context-setters)
    er_df = er_df[~er_df["noise_type"].str.contains("t1|_t1$", case=False, na=False)]

    results = []
    for _, row in er_df.iterrows():
        response   = str(row.get("response", ""))
        noise_type = _text(row.get("noise_type", ""))
        turn_num   = int(row.get("turn_number", 0))
        criteria   = _text(row.get("pass_criteria", ""))

        broken, reason, severity = _evaluate_er_turn(response, noise_type, criteria)
        results.append({"turn": turn_num, "noise": noise_type,
                        "breakdown": broken, "severity": severity,
                        "reason": reason, "response": response[:80]})

    hard    = sum(1 for r in results if r["severity"] == "HARD")
    soft    = sum(1 for r in results if r["severity"] == "SOFT")
    total   = len(results)
    # Weighted: hard breakdown = 1, soft = 0.5, OK = 0
    weighted_breakdowns = hard + soft * 0.5
    rate    = round(weighted_breakdowns / total * 100, 1) if total > 0 else 0
    band    = "GOOD" if rate < 10 else ("MODERATE" if rate <= 25 else "POOR")

    print(f"\n  [ER] Hard: {hard} | Soft: {soft} | OK: {total-hard-soft} / {total} turns")
    print(f"    Weighted breakdown rate: {rate}% → {band}")
    for r in results:
        icon = "✗" if r["breakdown"] else "✓"
        print(f"    Turn {r['turn']} ({r['noise']}): {icon} [{r['severity']}] {r['reason']}")

    # Per-noise-type breakdown
    by_type = defaultdict(lambda: {"total":0,"breakdowns":0})
    for r in results:
        noise_cat = r["noise"].split("_")[0]  # group t2 variants together
        by_type[noise_cat]["total"] += 1
        by_type[noise_cat]["breakdowns"] += 1 if r["breakdown"] else 0
    print(f"\n    Per noise-type:")
    for nt, counts in by_type.items():
        print(f"      {nt}: {counts['breakdowns']}/{counts['total']}")

    return {"metric": "ER", "hard_breakdowns": hard, "soft_breakdowns": soft,
            "total": total, "weighted_rate": rate,
            "score": rate, "band": band,
            "by_noise_type": {k: dict(v) for k,v in by_type.items()}}


def _evaluate_er_turn(response, noise_type, criteria):
    """
    Returns (is_breakdown: bool, reason: str, severity: str)
    severity: HARD / SOFT / OK
    HARD = complete failure (no response, pure default, explicit confusion)
    SOFT = tried but failed (wrong direction, ignored correction)
    OK   = handled correctly
    """
    r = str(response).strip()

    # Empty/error = always HARD breakdown
    if _is_empty_or_error(r):
        return True, "EMPTY_OR_ERROR", "HARD"

    r_lower = r.lower()

    # Check for hard breakdown phrases
    has_hard = any(re.search(p, r_lower) for p in HARD_BREAKDOWN_PATTERNS)
    has_hedge_helping = any(re.search(p, r_lower) for p in HEDGE_BUT_HELPING_PATTERNS)

    # Hard breakdown only if no hedging/helping
    if has_hard and not has_hedge_helping:
        return True, "HARD_BREAKDOWN_PHRASE", "HARD"

    # Out-of-scope noise type needs a specific decline
    if "out_of_scope" in noise_type or "oos" in noise_type:
        has_decline = any(re.search(p, r_lower) for p in OUT_OF_SCOPE_DECLINE_SIGNALS)
        if not has_decline and _word_count(r) > 80:
            # Bot gave a long answer to something out of scope = soft fail
            return True, "OOS_NOT_DECLINED (long response given)", "SOFT"
        return False, "OOS_CORRECTLY_DECLINED", "OK"

    # Contradiction T2: bot should pivot to new intent
    if "contradiction" in noise_type and "t2" in noise_type:
        # If criteria available, use it
        if criteria:
            passed, reason = _check_criteria(response, criteria)
            if not passed:
                return True, f"CORRECTION_IGNORED: {reason}", "SOFT"
            return False, "CORRECTION_ACKNOWLEDGED", "OK"
        # Fallback: check if bot acknowledged the change
        change_ack = bool(re.search(
            r"(alright|sure|of course|i'?ve? (changed|updated|noted)|got it|okay|understood)",
            r_lower))
        if change_ack:
            return False, "CORRECTION_ACKNOWLEDGED", "OK"
        return True, "CORRECTION_POSSIBLY_IGNORED", "SOFT"

    # Ambiguous input: bot should ask for clarification
    if "ambiguous" in noise_type:
        asks_clarification = bool(re.search(
            r"(which|what|could you (specify|clarify|tell me)|can you (tell|provide)|"
            r"more (information|details|context)|"
            r"(order|item|product) (number|are you referring))",
            r_lower))
        if asks_clarification:
            return False, "CLARIFICATION_REQUESTED", "OK"
        # Bot guessed and answered = soft fail
        if _word_count(r) > 50:
            return True, "ASSUMED_INTENT_WITHOUT_CLARIFYING", "SOFT"

    # Incomplete input: bot should ask what was meant
    if "incomplete" in noise_type:
        asks_completion = bool(re.search(
            r"(it looks like|seems like|cut off|incomplete|what were you going to|"
            r"you didn'?t finish|could you complete|please continue)",
            r_lower))
        if asks_completion or has_hedge_helping:
            return False, "INCOMPLETE_HANDLED", "OK"

    # Typo/noisy input: bot should understand despite errors
    if "typo" in noise_type or "noisy" in noise_type:
        if has_hard and not has_hedge_helping:
            return True, "FAILED_TO_DECODE_TYPO", "HARD"
        # If bot gave any substantive response, it decoded the typo
        if _word_count(r) > 20:
            return False, "TYPO_DECODED_SUCCESSFULLY", "OK"

    # Nonsense input: bot should indicate confusion politely
    if "nonsense" in noise_type:
        # If bot gave a very long response to nonsense = it hallucinated
        if _word_count(r) > 100 and not has_hedge_helping:
            return True, "HALLUCINATED_RESPONSE_TO_NONSENSE", "SOFT"

    # Default: not a breakdown
    return False, "OK", "OK"


# ══════════════════════════════════════════════════════════════════════
# COMPOSITE USABILITY SCORE
# ══════════════════════════════════════════════════════════════════════

def compute_cus(scores):
    """Equal-weight baseline CUS. Final weights via regression post-SUS."""
    normalized = {}

    cr = scores.get("CR", {})
    rt = scores.get("RT", {})
    hr = scores.get("HR", {})
    mx = scores.get("MATTR", {})
    er = scores.get("ER", {})

    if cr and cr.get("score") is not None:
        normalized["CR"] = cr["score"]
    if rt and rt.get("score") is not None:
        normalized["RT"] = max(0, round(100 - rt["score"] / 100, 2))
    if hr and hr.get("score") is not None:
        normalized["HR"] = hr["score"]
    if mx and mx.get("score") is not None:
        normalized["MATTR"] = round(mx["score"] * 100, 2)
    if er and er.get("score") is not None:
        normalized["ER"] = max(0, round(100 - er["score"], 2))

    if not normalized:
        return None

    cus = round(sum(normalized.values()) / len(normalized), 2)

    print(f"\n  [CUS] Normalized scores (equal weights):")
    for k, v in normalized.items():
        print(f"    {k}: {v}/100")
    print(f"  CUS = {cus}/100")

    return {"normalized": normalized, "cus_equal_weights": cus,
            "metrics_included": list(normalized.keys())}


# ══════════════════════════════════════════════════════════════════════
# MAIN SCORING FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def score_all(bot_name, logs_dir=None):
    """Score a single bot from the latest FULL_RUN log."""
    global LOGS_DIR
    if logs_dir: LOGS_DIR = Path(logs_dir)

    patterns = ["RUN*_FULL_*.csv", "FULL_RUN_*.csv"]
    run_files = []
    for p in patterns:
        run_files = sorted(LOGS_DIR.glob(p))
        if run_files: break

    if not run_files:
        print("No run files found. Run pipeline.py first.")
        return

    df_all = pd.read_csv(run_files[-1])
    df     = df_all[df_all["bot"] == bot_name].copy()

    if df.empty:
        print(f"No data for {bot_name} in {run_files[-1].name}")
        return

    print(f"\n{'='*60}\n  SCORING: {bot_name} | {len(df)} turns\n{'='*60}")

    scores = {
        "CR":    score_cr(df),
        "RT":    score_rt(df),
        "MATTR": score_mattr(df),
        "HR":    score_hr(df),
        "ER":    score_er(df),
    }
    cus = compute_cus(scores)

    output = {"bot": bot_name, "scores": scores, "cus": cus,
              "timestamp": datetime.now().isoformat()}
    out_path = LOGS_DIR / f"SCORES_{bot_name}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  ✓ Scores saved: {out_path}")
    return output


def score_all_runs(logs_dir=None):
    """Score across all runs → report mean ± SD per metric per bot."""
    global LOGS_DIR
    if logs_dir: LOGS_DIR = Path(logs_dir)

    patterns = ["RUN*_FULL_*.csv", "FULL_RUN_*.csv"]
    run_files = []
    for p in patterns:
        run_files = sorted(LOGS_DIR.glob(p))
        if run_files: break

    if not run_files:
        print("No run files found.")
        return

    print(f"Found {len(run_files)} run file(s).")
    bots = ["Rasa", "Llama3", "Mistral"]
    all_scores = {b: {"CR":[],"RT":[],"HR":[],"MATTR":[],"ER":[],"CUS":[]}
                  for b in bots}

    for f in run_files:
        df_all = pd.read_csv(f)
        for bot in bots:
            df = df_all[df_all["bot"] == bot].copy()
            if df.empty: continue

            cr = score_cr(df);    rt = score_rt(df)
            mx = score_mattr(df); hr = score_hr(df); er = score_er(df)

            def _s(d): return d.get("score") if d else None
            cr_s, rt_s, hr_s, mx_s, er_s = _s(cr),_s(rt),_s(hr),_s(mx),_s(er)

            cus_parts = {}
            if cr_s: cus_parts["CR"]    = cr_s
            if rt_s: cus_parts["RT"]    = max(0, round(100 - rt_s/100, 2))
            if hr_s: cus_parts["HR"]    = hr_s
            if mx_s: cus_parts["MATTR"] = round(mx_s * 100, 2)
            if er_s: cus_parts["ER"]    = max(0, round(100 - er_s, 2))
            cus_val = round(sum(cus_parts.values())/len(cus_parts),2) if cus_parts else None

            for k, v in [("CR",cr_s),("RT",rt_s),("HR",hr_s),
                         ("MATTR",mx_s),("ER",er_s),("CUS",cus_val)]:
                if v is not None:
                    all_scores[bot][k].append(v)

    print(f"\n{'='*60}\n  FINAL RESULTS — Mean ± SD across {len(run_files)} run(s)\n{'='*60}")
    summary = {}
    for bot in bots:
        print(f"\n  {bot}:")
        summary[bot] = {}
        for metric in ["CR","RT","MATTR","HR","ER","CUS"]:
            vals = all_scores[bot][metric]
            if vals:
                mean = round(float(np.mean(vals)), 2)
                sd   = round(float(np.std(vals)),  2)
                print(f"    {metric}: {mean} ± {sd}  (n={len(vals)})")
                summary[bot][metric] = {"mean": mean, "sd": sd, "n": len(vals)}
            else:
                print(f"    {metric}: NO DATA")

    out = LOGS_DIR / f"FINAL_SCORES_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✓ Final scores saved: {out}")
    return summary


if __name__ == "__main__":
    for bot in ["Llama3", "Mistral", "Rasa"]:
        score_all(bot)
