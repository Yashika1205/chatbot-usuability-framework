"""
metrics/prometheus_judge.py
---------------------------
Prometheus 2 LLM-as-a-Judge integration for Helpful Response Ratio scoring.

Prometheus 2 (Kim et al., EMNLP 2024) is used to judge whether a chatbot
response is helpful for 5 specified turns in the helpful_response script
(turns 2, 3, 7, 10, 12).

Deployment options (in order of preference):
  1. Local via Ollama: ollama pull prometheus  (or kaist-ai/prometheus-2)
  2. Local via Hugging Face transformers: kaist-ai/prometheus-2-7b
  3. Fallback: uses a strong general LLM (e.g. llama3.1:8b) with the
     Prometheus 2 rubric-grounding prompt to approximate the judge.

This module:
  - Constructs the Prometheus 2 absolute scoring prompt for each turn.
  - Calls the judge model via Ollama's /api/chat endpoint.
  - Returns a binary helpful/not-helpful verdict per turn.
  - Records the raw score and rationale for the Cohen's Kappa file.

Citations
---------
Kim et al. (2024) Prometheus 2: An Open Source Language Model Specialized
    in Evaluating Other Language Models. EMNLP 2024.
Zheng et al. (2023) Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena.
    NeurIPS 2023. (Self-preference bias justification for using Prometheus 2)
Landis & Koch (1977) Biometrics 33(1):159-174. (Kappa scale)
"""

from __future__ import annotations
import json
import os
import time
import requests


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
PROMETHEUS_MODEL = os.environ.get("PROMETHEUS_MODEL", "llama3.1:8b")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
JUDGE_TIMEOUT = int(os.environ.get("JUDGE_TIMEOUT", "90"))

# Turns to be judged by Prometheus 2 (from framework document Section 9)
PROMETHEUS_TURNS = {2, 3, 7, 10, 12}

# Kappa overlap turns (Prometheus 2 AND human both score these)
KAPPA_TURNS = {2, 5, 7, 10, 14}

# Helpful threshold: Prometheus 2 scores on 1–5 scale; ≥ 3 = helpful
HELPFUL_THRESHOLD = 3


# ---------------------------------------------------------------------------
# Prometheus 2 Absolute Scoring Prompt
# ---------------------------------------------------------------------------
_PROMETHEUS_SYSTEM = (
    "You are a fair and rigorous evaluator of chatbot helpfulness. "
    "You will be given a user query and a chatbot response. "
    "Score the response on a scale of 1 to 5 for helpfulness based on the rubric below. "
    "Return ONLY a JSON object with two keys: 'score' (integer 1–5) and 'rationale' (one sentence)."
)

_PROMETHEUS_RUBRIC = """
Rubric:
1 - Not helpful: Response ignores the question, is nonsensical, or provides harmful/irrelevant information.
2 - Slightly helpful: Partial answer that misses the core of the question or is too vague to be actionable.
3 - Moderately helpful: Addresses the question but lacks detail, specificity, or actionable guidance.
4 - Helpful: Correctly addresses the question with enough detail to be actionable.
5 - Very helpful: Complete, accurate, specific, and directly actionable response that fully serves the user's need.
""".strip()

_PROMETHEUS_USER_TEMPLATE = (
    "User query: {user_message}\n\n"
    "Chatbot response: {bot_response}\n\n"
    "{rubric}\n\n"
    "Provide your evaluation as JSON: {{\"score\": <1-5>, \"rationale\": \"<one sentence>\"}}"
)


def _build_prompt(user_message: str, bot_response: str) -> str:
    return _PROMETHEUS_USER_TEMPLATE.format(
        user_message=user_message,
        bot_response=bot_response,
        rubric=_PROMETHEUS_RUBRIC,
    )


# ---------------------------------------------------------------------------
# Judge call
# ---------------------------------------------------------------------------
def _call_judge(user_message: str, bot_response: str, use_mock: bool = False) -> dict:
    """
    Call the Prometheus 2 judge model and return the parsed result.

    Returns
    -------
    dict:
        score      : int    1–5
        rationale  : str
        is_helpful : bool   score >= HELPFUL_THRESHOLD
        raw_text   : str    full model output (for debugging)
    """
    if use_mock:
        return _mock_judge(user_message, bot_response)

    prompt = _build_prompt(user_message, bot_response)
    payload = {
        "model": PROMETHEUS_MODEL,
        "messages": [
            {"role": "system", "content": _PROMETHEUS_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    try:
        resp = requests.post(
            f"{OLLAMA_URL}/api/chat",
            json=payload,
            timeout=JUDGE_TIMEOUT,
        )
        resp.raise_for_status()
    except requests.exceptions.ConnectionError:
        raise ConnectionError(
            f"Prometheus judge model not reachable at {OLLAMA_URL}. "
            f"Start Ollama and pull model: ollama pull {PROMETHEUS_MODEL}"
        )

    raw_text = resp.json().get("message", {}).get("content", "")
    return _parse_judge_output(raw_text)


def _parse_judge_output(raw_text: str) -> dict:
    """
    Parse the JSON output from the judge model.
    Falls back gracefully if parsing fails.
    """
    # Try to extract JSON from the response
    text = raw_text.strip()
    # Find first { ... } block
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            parsed = json.loads(text[start:end])
            score = int(parsed.get("score", 1))
            score = max(1, min(5, score))  # clamp to 1–5
            rationale = parsed.get("rationale", "No rationale provided.")
            return {
                "score": score,
                "rationale": rationale,
                "is_helpful": score >= HELPFUL_THRESHOLD,
                "raw_text": raw_text,
            }
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: try to extract a number
    for word in text.split():
        try:
            score = int(word)
            if 1 <= score <= 5:
                return {
                    "score": score,
                    "rationale": "Parsed from unstructured output.",
                    "is_helpful": score >= HELPFUL_THRESHOLD,
                    "raw_text": raw_text,
                }
        except ValueError:
            continue

    # Cannot parse — default to not helpful
    return {
        "score": 1,
        "rationale": f"Parse failure. Raw output: {raw_text[:200]}",
        "is_helpful": False,
        "raw_text": raw_text,
    }


def _mock_judge(user_message: str, bot_response: str) -> dict:
    """Mock judge for offline testing — scores > MIN_TOKENS as helpful."""
    time.sleep(0.1)
    score = 4 if len(bot_response) > 50 else 2
    return {
        "score": score,
        "rationale": f"[MOCK] Response length={len(bot_response)}.",
        "is_helpful": score >= HELPFUL_THRESHOLD,
        "raw_text": f"{{\"score\": {score}, \"rationale\": \"mock\"}}",
    }


# ---------------------------------------------------------------------------
# Script-level judging
# ---------------------------------------------------------------------------
def judge_helpful_response_turns(
    log: list[dict],
    turns_to_judge: set[int] = None,
    use_mock: bool = False,
) -> dict:
    """
    Run Prometheus 2 judging on the specified turns of a conversation log.

    Parameters
    ----------
    log : list[dict]
        Conversation log from the helpful_response script.
    turns_to_judge : set[int]
        Turn numbers to judge. Defaults to PROMETHEUS_TURNS {2,3,7,10,12}.
    use_mock : bool
        Use mock judge (for offline testing).

    Returns
    -------
    dict:
        prometheus_scores    : {turn_number: bool}  for metric_engine
        per_turn_details     : list[dict]
        judge_model          : str
    """
    if turns_to_judge is None:
        turns_to_judge = PROMETHEUS_TURNS

    env_mock = os.environ.get("PROMETHEUS_MOCK", "0").strip() == "1"
    use_mock = use_mock or env_mock

    prometheus_scores: dict[int, bool] = {}
    per_turn_details = []

    for turn in log:
        turn_num = turn.get("turn_number")
        if turn_num not in turns_to_judge:
            continue

        user_msg = turn.get("user_message", "")
        bot_resp = turn.get("bot_response", "")

        result = _call_judge(user_msg, bot_resp, use_mock=use_mock)
        prometheus_scores[turn_num] = result["is_helpful"]

        per_turn_details.append({
            "turn_number": turn_num,
            "user_message": user_msg[:120],
            "bot_response_preview": bot_resp[:120],
            "prometheus_score": result["score"],
            "prometheus_rationale": result["rationale"],
            "is_helpful": result["is_helpful"],
        })

        print(
            f"  [Prometheus] Turn {turn_num}: score={result['score']}/5 "
            f"({'helpful' if result['is_helpful'] else 'NOT helpful'}) — {result['rationale'][:80]}"
        )

    return {
        "prometheus_scores": prometheus_scores,
        "per_turn_details": per_turn_details,
        "judge_model": PROMETHEUS_MODEL,
    }


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    sample_log = [
        {"turn_number": 2, "user_message": "I have 2 years of Python experience. What roles should I target?",
         "bot_response": "With 2 years of Python experience you should target mid-level backend developer roles. Look for positions like Software Engineer II, Python Developer, and Backend Engineer at startups or mid-size companies. Your startup background is valuable — highlight the breadth of your experience."},
        {"turn_number": 3, "user_message": "Can you write a resume summary for a mid-level Python developer?",
         "bot_response": "Sure! Here is one: results-driven Python developer with 2 years of startup experience building scalable backend services. Proficient in REST APIs, SQL, and cloud deployment. Passionate about clean code and agile workflows."},
        {"turn_number": 7, "user_message": "What questions should I ask the interviewer at the end?",
         "bot_response": "I think you should probably just say thank you and leave."},
    ]

    result = judge_helpful_response_turns(sample_log, use_mock=True)
    print("\nPrometheus scores:", result["prometheus_scores"])
    print("Per-turn details:")
    for d in result["per_turn_details"]:
        print(f"  Turn {d['turn_number']}: {d['prometheus_score']}/5 — {d['prometheus_rationale']}")
