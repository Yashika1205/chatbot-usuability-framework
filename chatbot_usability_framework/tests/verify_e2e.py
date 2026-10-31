"""
tests/verify_e2e.py
-------------------
End-to-end verification test for the Chatbot Usability Testing Framework.
Run with: py tests/verify_e2e.py  (from the chatbot_usability_framework directory)
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force all mocks
os.environ["RASA_MOCK"] = "1"
os.environ["OLLAMA_MOCK"] = "1"
os.environ["PROMETHEUS_MOCK"] = "1"

PASS = "[PASS]"
FAIL = "[FAIL]"
errors = []


def check(label, condition, detail=""):
    if condition:
        print(f"  {PASS} {label}" + (f": {detail}" if detail else ""))
    else:
        print(f"  {FAIL} {label}" + (f": {detail}" if detail else ""))
        errors.append(label)


print("\n" + "=" * 60)
print("  END-TO-END VERIFICATION")
print("=" * 60)

# ---------------------------------------------------------------------------
print("\n1. ADAPTERS")
from adapters.rasa_adapter import RasaAdapter
from adapters.ollama_adapter import OllamaAdapter

rasa = RasaAdapter(use_mock=True)
ollama = OllamaAdapter(use_mock=True)

r1 = rasa.send("Hello, I am vegetarian.")
r2 = ollama.send("Hello, I am vegetarian.")
check("Rasa adapter returns response", bool(r1["response"]), r1["response"][:40])
check("Rasa adapter returns latency", r1["latency_ms"] > 0, f"{r1['latency_ms']} ms")
check("Ollama adapter returns response", bool(r2["response"]), r2["response"][:40])
check("Ollama multi-turn history", len(ollama._messages) >= 2, f"{len(ollama._messages)} messages")
ollama.reset()
check("Ollama reset clears history", len(ollama._messages) == 0)

# ---------------------------------------------------------------------------
print("\n2. LOGGER")
from logger import ConversationLogger, load_log

lg = ConversationLogger("rasa", "test_verify")
lg.log_turn(1, "Hi", "Hello!", 100.0)
lg.log_turn(2, "Bye", "Goodbye!", 90.0)
path = lg.save()
loaded = load_log(path)
check("Logger saves turns", len(loaded) == 2, f"2 turns at {os.path.basename(path)}")
check("Logger preserves latency", loaded[0]["latency_ms"] == 100.0)
check("Logger preserves turn_number", loaded[1]["turn_number"] == 2)

# ---------------------------------------------------------------------------
print("\n3. TASK BANK")
from task_bank.task_bank import get_script, list_scripts, SCRIPTS

scripts = list_scripts()
check("5 scripts registered", len(scripts) == 5, str(scripts))
for name in scripts:
    turns = get_script(name)
    expected = SCRIPTS[name]["total_turns"]
    check(f"Script {name}: correct turn count", len(turns) == expected, f"{len(turns)} turns")

# Check all turns have required fields
for name in scripts:
    for turn in get_script(name):
        assert "turn_number" in turn and "user_message" in turn, f"Missing field in {name} turn {turn}"
check("All turns have required fields", True)

# ---------------------------------------------------------------------------
print("\n4. ORCHESTRATOR")
from orchestrator import Orchestrator

orch_rasa = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0, verbose=False)

# Pilot
pilot_log = orch_rasa.run_pilot("context_retention", num_turns=5)
check("Pilot runs 5 turns", len(pilot_log) == 5)
check("Pilot log has bot_response", all("bot_response" in t for t in pilot_log))

# Full context retention script
full_log = orch_rasa.run_script("context_retention")
check("Full context_retention script: 30 turns", len(full_log) == 30)

# ---------------------------------------------------------------------------
print("\n5. METRIC ENGINE")
from metrics.metric_engine import (
    compute_context_retention, compute_response_time,
    compute_helpful_response, compute_lexical_diversity, compute_error_rate,
)
from metrics.prometheus_judge import judge_helpful_response_turns

# Context Retention
cr = compute_context_retention(full_log)
check("Context Retention computes", "retention_rate_pct" in cr)
check("Context Retention normalized 0-100", 0 <= cr["normalized_score"] <= 100,
      f"{cr['normalized_score']}")
check("Context Retention has classification", cr["classification"] in ("Good", "Moderate", "Poor"))

# Response Time
rt = compute_response_time(full_log)
check("Response Time computes", "average_ms" in rt)
check("Response Time avg > 0", rt["average_ms"] > 0, f"{rt['average_ms']} ms")
check("Response Time normalized 0-100", 0 <= rt["normalized_score"] <= 100)

# Helpful Response
orch_hr = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0, verbose=False)
hr_log = orch_hr.run_script("helpful_response")
judge_result = judge_helpful_response_turns(hr_log, use_mock=True)
check("Prometheus judge returns scores", len(judge_result["prometheus_scores"]) > 0,
      str(judge_result["prometheus_scores"]))
hr = compute_helpful_response(hr_log, prometheus_scores=judge_result["prometheus_scores"])
check("Helpful Response computes", "helpful_ratio_pct" in hr)
check("Helpful Response normalized 0-100", 0 <= hr["normalized_score"] <= 100)

# Lexical Diversity
orch_ld = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0, verbose=False)
ld_log = orch_ld.run_script("lexical_diversity")
ld = compute_lexical_diversity(ld_log)
check("Lexical Diversity computes", "average_mattr" in ld or ld["scored_turns"] == 0)

# Error Rate
orch_er = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0, verbose=False)
er_log = orch_er.run_script("error_rate")
er = compute_error_rate(er_log)
check("Error Rate computes", "error_rate_pct" in er)
check("Error Rate normalized 0-100", 0 <= er["normalized_score"] <= 100)
check("Error Rate has taxonomy", isinstance(er["taxonomy_summary"], dict))

# ---------------------------------------------------------------------------
print("\n6. MATTR CALCULATOR")
from metrics.mattr_calculator import compute_mattr, pilot_calibrate

long_text = (
    "Index funds are diversified investment vehicles that track a market index. "
    "They offer low expense ratios, broad diversification, and consistent long-term "
    "returns. Over a twenty year horizon, the majority of actively managed funds "
    "underperform their benchmark index, which further supports passive investing "
    "strategies for retail investors who want market exposure without complexity."
)
short_text = "Yes."

mattr_long = compute_mattr(long_text)
mattr_short = compute_mattr(short_text)
check("MATTR: long text scored", mattr_long["status"] == "scored", f"score={mattr_long['mattr']}")
check("MATTR: short text excluded", mattr_short["status"] == "short_excluded")
check("MATTR score in 0-1", 0 <= mattr_long["mattr"] <= 1)

good = [long_text, long_text + " Diversification reduces risk substantially over time."]
padded = ["That is a great question. I am happy to help. Sure, I can definitely assist. " * 3]
thresholds = pilot_calibrate(good, padded)
check("MATTR pilot calibration works", thresholds["calibrated"] is True)
check("MATTR thresholds are floats", isinstance(thresholds["good_min"], float))

# ---------------------------------------------------------------------------
print("\n7. COHEN'S KAPPA")
from metrics.kappa_calculator import compute_kappa, compute_kappa_from_scores

# Perfect agreement
k1 = compute_kappa([True, True, False, False, True], [True, True, False, False, True])
check("Kappa: perfect agreement = 1.0", k1["kappa"] == 1.0, str(k1["kappa"]))

# Partial agreement
k2 = compute_kappa([True, True, False, True, False], [True, False, False, True, False])
check("Kappa: 4/5 agreement > 0", k2["kappa"] > 0, str(k2["kappa"]))
check("Kappa: has interpretation", bool(k2["interpretation"]))

# From score dicts
prom = {2: True, 5: True, 7: False, 10: True, 14: True}
human = {2: True, 5: False, 7: False, 10: True, 14: True}
k3 = compute_kappa_from_scores(prom, human)
check("Kappa from score dicts works", "kappa" in k3 and k3["kappa"] is not None)
check("Kappa has action recommendation", "action" in k3)

# ---------------------------------------------------------------------------
print("\n8. COMPOSITE USABILITY SCORE")
from scoring import compute_cus, compute_cus_comparison

metric_results = {
    "context_retention": cr,
    "response_time": rt,
    "helpful_response": hr,
    "lexical_diversity": ld,
    "error_rate": er,
}
cus = compute_cus(metric_results)
check("CUS computes", "cus_equal" in cus)
check("CUS in 0-100", 0 <= cus["cus_equal"] <= 100, f"CUS={cus['cus_equal']}")
check("CUS has classification", cus["classification"] in ("Good", "Moderate", "Poor"))
check("CUS has 5 metric scores", len(cus["scores"]) == 5)

# ---------------------------------------------------------------------------
print("\n9. REPORT GENERATOR")
from report import generate_report

report = generate_report("rasa", metric_results, cus_result=cus)
check("Report generated", len(report) > 500)
check("Report contains CUS section", "COMPOSITE USABILITY SCORE" in report)
check("Report contains per-metric sections", "Metric 1" in report)
check("Report contains chatbot name", "RASA" in report)

# ---------------------------------------------------------------------------
print("\n10. LOG FILES CREATED")
log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "logs")
log_files = [f for f in os.listdir(log_dir) if f.endswith(".jsonl")]
check("Log files created", len(log_files) >= 5, f"{len(log_files)} files")

report_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "reports")
report_files = [f for f in os.listdir(report_dir) if f.endswith(".txt")]
check("Report files created", len(report_files) >= 1, f"{len(report_files)} files")

# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
if errors:
    print(f"  RESULT: {len(errors)} FAILED")
    for e in errors:
        print(f"    - {e}")
else:
    print("  RESULT: ALL CHECKS PASSED")
    print()
    print(f"  CUS Summary (Rasa, mock mode):")
    for metric, score in cus["scores"].items():
        print(f"    {metric:25s}: {score:.1f} / 100")
    print(f"    {'CUS (equal-weighted)':25s}: {cus['cus_equal']} / 100  ({cus['classification']})")
print("=" * 60 + "\n")
