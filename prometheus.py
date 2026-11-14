"""
prometheus.py — LLM-as-Judge scoring for HR Prometheus turns.
Domain-agnostic: rubric comes from CSV pass_criteria column.
Scores 3 consecutive turns per bot to save time.
Run AFTER pipeline.py completes.
"""

import requests, json, pandas as pd
from pathlib import Path
from datetime import datetime

LOGS_DIR      = Path("logs")
OLLAMA_URL    = "http://localhost:11434/api/generate"
PROMETHEUS    = "prometheus-eval/prometheus-7b-v2.0"

# ── Which turns use Prometheus (from hr_script.csv judge_type column) ─
# This is read from the CSV, not hardcoded — but we define a fallback
# rubric template that works for ANY domain
GENERIC_RUBRIC = """Evaluate this chatbot response for helpfulness.
A response is HELPFUL if it:
1. Directly addresses the user's question (does not deflect or refuse without reason)
2. Provides specific, actionable information (not just vague generic advice)
3. Is substantive enough to be useful (more than one sentence for a non-trivial question)

A response is NOT HELPFUL if it:
- Refuses without good reason
- Gives only generic advice with no specifics
- Ignores the actual question asked
- Is too short to be useful

Answer with exactly: PASS or FAIL, then one sentence explaining why."""


def prometheus_judge(question, response, rubric=None):
    """
    Sends a question+response pair to Prometheus for helpfulness evaluation.
    Falls back to generic rubric if none provided.
    Domain-agnostic: rubric determines what 'helpful' means.
    """
    rubric = rubric or GENERIC_RUBRIC
    prompt = f"""You are a neutral evaluator assessing chatbot response quality.

USER QUESTION: {question}

CHATBOT RESPONSE: {response}

EVALUATION RUBRIC:
{rubric}

Your verdict:"""

    try:
        r = requests.post(OLLAMA_URL,
                          json={"model": PROMETHEUS,
                                "prompt": prompt,
                                "stream": False},
                          timeout=300)
        verdict = r.json().get("response", "[ERROR: no response]").strip()
        passed  = verdict.upper().startswith("PASS")
        return {"verdict": verdict, "pass": passed, "error": None}
    except Exception as e:
        return {"verdict": f"[ERROR: {e}]", "pass": None, "error": str(e)}


def run_prometheus(bot_name, turns_to_score=3, logs_dir=None):
    """
    Scores Prometheus turns for one bot.
    turns_to_score: how many Prometheus turns to run (3 recommended for time).
    Reads rubric from pass_criteria column in the HR log CSV.
    """
    global LOGS_DIR
    if logs_dir: LOGS_DIR = Path(logs_dir)

    # Find latest HR log for this bot
    hr_files = sorted(LOGS_DIR.glob(f"*HR_{bot_name}*.csv"))
    if not hr_files:
        hr_files = sorted(LOGS_DIR.glob(f"*_HR_*.csv"))
        hr_files = [f for f in hr_files if bot_name in f.name]

    if not hr_files:
        print(f"No HR log found for {bot_name}")
        return {}

    df = pd.read_csv(hr_files[-1])

    # Filter to Prometheus turns
    if "judge_type" in df.columns:
        p_df = df[df["judge_type"].str.contains("prometheus", case=False, na=False)].copy()
    else:
        print("No judge_type column — cannot identify Prometheus turns")
        return {}

    if p_df.empty:
        print(f"No Prometheus turns found for {bot_name}")
        return {}

    # Limit to turns_to_score
    p_df = p_df.head(turns_to_score)

    print(f"\n=== Prometheus scoring: {bot_name} ({len(p_df)} turns) ===")
    results = {}

    for _, row in p_df.iterrows():
        turn_num = int(row["turn_number"])
        question = str(row["question"])
        response = str(row["response"])
        rubric   = str(row.get("pass_criteria", "")) or GENERIC_RUBRIC

        print(f"  Turn {turn_num}: judging...")
        result = prometheus_judge(question, response, rubric)

        icon = "✓ PASS" if result["pass"] else ("✗ FAIL" if result["pass"] is False else "? ERROR")
        print(f"  Turn {turn_num}: {icon}")
        print(f"           {result['verdict'][:120]}")

        results[str(turn_num)] = result

    return results


def run_all_bots(turns_per_bot=3, logs_dir=None):
    """Run Prometheus for all 3 bots and save results."""
    global LOGS_DIR
    if logs_dir: LOGS_DIR = Path(logs_dir)

    all_results = {}
    for bot in ["Rasa", "Llama3", "Mistral"]:
        bot_results = run_prometheus(bot, turns_to_score=turns_per_bot,
                                     logs_dir=logs_dir)
        all_results[bot] = bot_results

    # Save to logs dir so metrics.py can read it
    out_path = LOGS_DIR / "prometheus_scores.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ All Prometheus scores saved: {out_path}")
    return all_results


if __name__ == "__main__":
    run_all_bots(turns_per_bot=3)
