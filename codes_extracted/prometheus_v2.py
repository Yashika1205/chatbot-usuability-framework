"""
prometheus_v2.py - LLM-as-judge for the 6 'prometheus' turns of the HR script.

Fixes vs prometheus.py:
  * --run N [N ...]  (the old file had no such flag; scores for every run were written to ONE
    prometheus_scores.json, so runs 1-3 silently shared the same judgments).
  * Scores ALL prometheus turns (old code did head(3) -> 3 of 6 turns stayed 'pending' and HR
    was computed on a shrunken denominator).
  * Uses Prometheus-2's trained ABSOLUTE-GRADING format (1-5 score + "[RESULT] n") instead of a
    free-form PASS/FAIL prompt. Kim et al. (EMNLP 2024), r = 0.897, is only valid in this format.
  * PASS = score >= PASS_MIN (default 4); both >=3 and >=4 are stored so you can report either.
  * Old parser used verdict.startswith("PASS"); a reply of "**PASS**" was recorded as FAIL.
  * REFUSES TO RUN if the judge model is not installed (old run wrote "[ERROR: no response]"
    and metrics.py counted those as FAIL) and runs a sanity gate (good answer >=4, junk <=2).
  * Errors are stored as pass=None and never counted as FAIL.

Setup (after pipeline runs finished, to free VRAM):
  ollama stop llama3.1; ollama stop mistral
  ollama pull hf.co/<user>/<repo>:<quant>      # a GGUF of prometheus-eval/prometheus-7b-v2.0
  export PROMETHEUS_MODEL="hf.co/<user>/<repo>:<quant>"   # exact name shown by `ollama list`
Usage:
  python prometheus_v2.py --run 1 2 3
"""

import argparse, json, os, re, time
from pathlib import Path

import pandas as pd
import requests

OLLAMA_BASE = os.getenv("OLLAMA_BASE", "http://localhost:11434")
LOGS_DIR    = Path(os.getenv("LOGS_DIR", "logs"))
JUDGE       = os.getenv("PROMETHEUS_MODEL", "prometheus")
PASS_MIN    = int(os.getenv("PASS_MIN", 4))

SYSTEM = ("You are a fair judge assistant tasked with providing clear, objective feedback based on "
          "specific criteria, ensuring each assessment reflects the absolute standards set for performance.")

TEMPLATE = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, a reference answer that gets a score of 5, and a score rubric representing a evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is an integer between 1 and 5. You should refer to the score rubric.
3. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (an integer number between 1 and 5)"
4. Please do not generate any other opening, closing, and explanations.

###The instruction to evaluate:
{instruction}

###Response to evaluate:
{response}

###Reference Answer (Score 5):
{reference}

###Score Rubrics:
{rubric}

###Feedback: """


def rubric(question, s1, s2, s3, s4, s5):
    return (f"[{question}]\nScore 1: {s1}\nScore 2: {s2}\nScore 3: {s3}\nScore 4: {s4}\nScore 5: {s5}")


# One spec per prometheus turn of hr_script.csv (turn numbers 2,3,7,10,12,13).
# Reference = what a score-5 answer contains (derived from workbook "Expected Outcome").
SPEC = {
    2: dict(
        reference="A list of 6+ realistic first-round questions for a marketing coordinator: campaign experience, "
                  "analytics tools, social media, content calendar, teamwork/deadlines, why this company.",
        rubric=rubric("Are the interview questions specific to a first-round marketing coordinator interview?",
                      "No interview questions, or refuses/irrelevant.",
                      "A few questions, nearly all generic ('tell me about yourself') with nothing marketing-specific.",
                      "Some marketing-specific questions mixed with mostly generic filler.",
                      "Mostly marketing-coordinator-specific, plausible questions; minor generic filler.",
                      "A well-organised list of role-specific, plausible first-round questions (campaigns, analytics, content, collaboration).")),
    3: dict(
        reference="A framework: pick a real, non-critical weakness, show self-awareness, describe concrete steps taken to "
                  "improve, plus a short worked example answer.",
        rubric=rubric("Does the response give an actionable framework or a worked example for answering 'What's your biggest weakness?'",
                      "No usable advice, or off-topic.",
                      "Only generic advice such as 'be honest' with no structure or example.",
                      "Some structure but no concrete example answer, or an example without the improvement step.",
                      "Clear framework and at least one example, minor gaps.",
                      "Clear framework (real weakness + improvement steps) and a worked example answer.")),
    7: dict(
        reference="Wait ~1 week after the stated date, then send a short polite email; gives timing, tone, and a sample "
                  "email wording.",
        rubric=rubric("Does the response give actionable steps for following up after 2 weeks of silence (timing + tone + example wording)?",
                      "No follow-up advice, or off-topic.",
                      "Vague advice ('follow up politely') without timing or wording.",
                      "Covers one or two of timing, tone, example wording.",
                      "Covers timing and tone and gives some wording.",
                      "Gives concrete timing, tone guidance and a usable example message.")),
    10: dict(
        reference="Frame the gap positively and briefly (upskilling, caregiving, relocation, freelance), keep it short, pivot "
                  "to skills and goals; includes example wording.",
        rubric=rubric("Does the response give a specific strategy (framing and example wording) for explaining a 6-month employment gap?",
                      "No usable advice, or off-topic.",
                      "Only 'be honest' style generic advice.",
                      "Some strategy but no example wording.",
                      "Specific framing strategy plus partial example wording.",
                      "Specific framing strategy, pivot to skills/goals, and example wording the user can say.")),
    12: dict(
        reference="Research market norms, tie the ask to value, anchor with a number, stay polite; includes a sample script such "
                  "as 'Given my experience ..., would a signing bonus of $X be possible?'",
        rubric=rubric("Does the response include specific negotiation language or a sample script the user can actually say?",
                      "No negotiation help, or off-topic.",
                      "Generic encouragement only.",
                      "Talking points but no wording the user can say.",
                      "Talking points and some example phrasing.",
                      "Clear talking points plus a sample script or exact phrasing.")),
    13: dict(
        reference="Takes a position (usually: mention it carefully, as leverage, never as a bluff) or lays out clear pros/cons "
                  "and conditions under which to share it.",
        rubric=rubric("Does the response take a position or lay out clear trade-offs about mentioning a counteroffer, rather than dodging?",
                      "Refuses or gives a non-answer.",
                      "Dodges: 'it depends' with no substance.",
                      "Mentions considerations but no position or clear trade-offs.",
                      "Clear trade-offs or a leaning position with minor gaps.",
                      "Takes a clear position and/or lays out explicit pros and cons with conditions.")),
}

SANITY = [  # (turn, response, expected 'ge' or 'le', bound)
    (3, "Pick a genuine weakness that is not essential to the job, say what you are doing about it, and end on progress. "
        "Example: 'I used to over-commit to projects. I now block time weekly to prioritise and delegate, and my last two "
        "campaigns shipped on time.'", "ge", 4),
    (3, "Great, carry on!", "le", 2),
]


def call_judge(q, response, spec, retries=2):
    prompt = TEMPLATE.format(instruction=q, response=response, reference=spec["reference"], rubric=spec["rubric"])
    last = ""
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{OLLAMA_BASE}/api/chat", timeout=600, json={
                "model": JUDGE, "stream": False, "keep_alive": "30m",
                "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
                "options": {"temperature": 0, "num_ctx": 8192, "seed": 7}})
            r.raise_for_status()
            last = r.json()["message"]["content"]
            m = re.search(r"\[RESULT\]\s*\(?\s*([1-5])", last)
            if m:
                return int(m.group(1)), last
        except Exception as e:
            last = f"[ERROR: {e}]"
        time.sleep(3)
    return None, last


def check_model():
    tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=15).json().get("models", [])
    names = [t["name"] for t in tags]
    if not any(n == JUDGE or n.split(":")[0] == JUDGE.split(":")[0] or JUDGE in n for n in names):
        raise SystemExit(f"Judge model '{JUDGE}' not installed. Installed: {names}\n"
                         f"Set PROMETHEUS_MODEL to the exact name from `ollama list`.")
    print(f"Judge model: {JUDGE}")


def sanity_gate():
    print("Sanity gate ...")
    for turn, resp, op, bound in SANITY:
        s, raw = call_judge("sanity", resp, SPEC[turn])
        ok = s is not None and ((s >= bound) if op == "ge" else (s <= bound))
        print(f"  score={s} expected {op} {bound} -> {'OK' if ok else 'FAIL'}")
        if not ok:
            raise SystemExit("Judge failed the sanity gate (wrong model, bad chat template, or bad parsing).\n"
                             f"Raw output:\n{raw[:500]}")


def judge_run(run):
    f = LOGS_DIR / f"RUN{run}_FULL.csv"
    if not f.exists():
        print(f"  {f} not found - skipped"); return
    df = pd.read_csv(f)
    hr = df[(df["script"] == "HR") & df["judge_type"].astype(str).str.contains("prometheus", case=False)]
    out_path = LOGS_DIR / f"prometheus_run{run}.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for _, row in hr.iterrows():
        bot, t = str(row["bot"]), int(row["turn_number"])
        if t not in SPEC:
            print(f"  ! no judge spec for HR turn {t}"); continue
        if str(t) in out.get(bot, {}) and out[bot][str(t)].get("score") is not None:
            continue                                            # resume
        resp = str(row["response"])
        if resp.startswith("[ERROR") or not resp.strip():
            out.setdefault(bot, {})[str(t)] = {"score": 1, "pass": False, "pass_ge3": False,
                                               "feedback": "bot returned error/empty (auto score 1)", "auto": True}
            continue
        s, raw = call_judge(str(row["question"]), resp, SPEC[t])
        out.setdefault(bot, {})[str(t)] = {
            "score": s, "pass": (None if s is None else s >= PASS_MIN),
            "pass_ge3": (None if s is None else s >= 3), "feedback": raw[:1500], "auto": False}
        print(f"  run{run} {bot:8s} turn {t:>2}: score={s}")
        out_path.write_text(json.dumps(out, indent=2))
    out["_meta"] = {"judge": JUDGE, "pass_min": PASS_MIN, "template": "prometheus-eval absolute grading"}
    out_path.write_text(json.dumps(out, indent=2))
    print(f"  saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=int, nargs="+", required=True)
    ap.add_argument("--skip-sanity", action="store_true")
    a = ap.parse_args()
    check_model()
    if not a.skip_sanity:
        sanity_gate()
    for r in a.run:
        print(f"\n=== Judging run {r} ===")
        judge_run(r)


if __name__ == "__main__":
    main()
