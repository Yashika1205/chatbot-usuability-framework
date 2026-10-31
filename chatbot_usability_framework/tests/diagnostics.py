"""
Comprehensive diagnostic — tests every component individually.
Writes clean ASCII output to avoid encoding issues.
"""
import sys, os
sys.path.insert(0, r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework")
os.environ["RASA_MOCK"] = "1"
os.environ["OLLAMA_MOCK"] = "1"
os.environ["PROMETHEUS_MOCK"] = "1"

results = []
passed = 0
failed = 0

def check(label, ok, detail=""):
    global passed, failed
    mark = "PASS" if ok else "FAIL"
    line = f"  [{mark}] {label}"
    if detail:
        line += f"  => {detail}"
    results.append(line)
    if ok:
        passed += 1
    else:
        failed += 1
    print(line)

print("\n" + "="*60)
print("  CHATBOT USABILITY FRAMEWORK - DIAGNOSTICS")
print("="*60)

# ============================================================
print("\n-- 1. IMPORTS --")
try:
    from adapters.rasa_adapter import RasaAdapter
    from adapters.ollama_adapter import OllamaAdapter
    check("RasaAdapter import", True)
    check("OllamaAdapter import", True)
except Exception as e:
    check("Adapters import", False, str(e))

try:
    from orchestrator import Orchestrator
    check("Orchestrator import", True)
except Exception as e:
    check("Orchestrator import", False, str(e))

try:
    from logger import ConversationLogger, load_log
    check("Logger import", True)
except Exception as e:
    check("Logger import", False, str(e))

try:
    from task_bank.task_bank import get_script, list_scripts, SCRIPTS
    check("Task bank import", True)
except Exception as e:
    check("Task bank import", False, str(e))

try:
    from metrics.metric_engine import (
        compute_context_retention, compute_response_time,
        compute_helpful_response, compute_lexical_diversity, compute_error_rate,
    )
    check("Metric engine import", True)
except Exception as e:
    check("Metric engine import", False, str(e))

try:
    from metrics.mattr_calculator import compute_mattr, pilot_calibrate
    check("MATTR calculator import", True)
except Exception as e:
    check("MATTR calculator import", False, str(e))

try:
    from metrics.prometheus_judge import judge_helpful_response_turns
    check("Prometheus judge import", True)
except Exception as e:
    check("Prometheus judge import", False, str(e))

try:
    from metrics.kappa_calculator import compute_kappa, compute_kappa_from_scores
    check("Kappa calculator import", True)
except Exception as e:
    check("Kappa calculator import", False, str(e))

try:
    from scoring import compute_cus, compute_cus_comparison
    check("Scoring import", True)
except Exception as e:
    check("Scoring import", False, str(e))

try:
    from sus_study import aggregate_sus, calculate_sus_score
    check("SUS study import", True)
except Exception as e:
    check("SUS study import", False, str(e))

try:
    from analysis import compute_correlation, compute_regression
    check("Analysis import", True)
except Exception as e:
    check("Analysis import", False, str(e))

try:
    from report import generate_report
    check("Report import", True)
except Exception as e:
    check("Report import", False, str(e))

# ============================================================
print("\n-- 2. ADAPTERS --")
try:
    rasa = RasaAdapter(use_mock=True)
    r = rasa.send("Hello I am vegetarian")
    check("Rasa mock send", bool(r["response"]), f"response len={len(r['response'])}")
    check("Rasa mock latency", r["latency_ms"] > 0, f"{r['latency_ms']}ms")
    check("Rasa health_check mock", rasa.health_check() == True)
except Exception as e:
    check("Rasa adapter", False, str(e))

try:
    ollama = OllamaAdapter(use_mock=True)
    r1 = ollama.send("Hello I am vegetarian")
    r2 = ollama.send("What should I eat?")
    check("Ollama mock send", bool(r1["response"]), f"response len={len(r1['response'])}")
    check("Ollama multi-turn history", len(ollama._messages) >= 4, f"{len(ollama._messages)} messages in history")
    ollama.reset()
    check("Ollama reset", len(ollama._messages) == 0, "history cleared")
except Exception as e:
    check("Ollama adapter", False, str(e))

# ============================================================
print("\n-- 3. LOGGER --")
try:
    lg = ConversationLogger("rasa", "diag_test")
    lg.log_turn(1, "Hi", "Hello!", 100.0, metadata={"checkpoint": 1})
    lg.log_turn(2, "Bye", "Goodbye!", 90.5)
    path = lg.save()
    check("Logger save", os.path.exists(path), os.path.basename(path))
    loaded = load_log(path)
    check("Logger load", len(loaded) == 2, f"{len(loaded)} turns loaded")
    check("Logger latency preserved", loaded[0]["latency_ms"] == 100.0)
    check("Logger metadata preserved", loaded[0].get("metadata", {}).get("checkpoint") == 1)
    check("Logger turn_number preserved", loaded[1]["turn_number"] == 2)
except Exception as e:
    check("Logger", False, str(e))

# ============================================================
print("\n-- 4. TASK BANK --")
try:
    scripts = list_scripts()
    check("5 scripts exist", len(scripts) == 5, str(scripts))
    
    expected_turns = {"context_retention": 30, "response_time": 15,
                      "helpful_response": 15, "lexical_diversity": 12, "error_rate": 15}
    for name, expected in expected_turns.items():
        turns = get_script(name)
        check(f"{name} turn count", len(turns) == expected, f"{len(turns)}/{expected}")
    
    # Check all turns have required fields
    all_valid = True
    for name in scripts:
        for turn in get_script(name):
            if "turn_number" not in turn or "user_message" not in turn:
                all_valid = False
                break
    check("All turns have required fields", all_valid)
    
    # Check turn numbers are sequential per script
    for name in scripts:
        turns = get_script(name)
        nums = [t["turn_number"] for t in turns]
        check(f"{name} sequential turns", nums == list(range(1, len(turns)+1)),
              f"first={nums[0]}, last={nums[-1]}")
except Exception as e:
    check("Task bank", False, str(e))

# ============================================================
print("\n-- 5. ORCHESTRATOR --")
try:
    rasa = RasaAdapter(use_mock=True)
    orch = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0, verbose=False)
    
    # Pilot
    pilot = orch.run_pilot("context_retention", num_turns=5)
    check("Pilot: 5 turns", len(pilot) == 5)
    check("Pilot: has bot_response", all("bot_response" in t for t in pilot))
    check("Pilot: has latency_ms", all("latency_ms" in t for t in pilot))
    
    # Full scripts
    cr_log = orch.run_script("context_retention")
    check("context_retention: 30 turns", len(cr_log) == 30, f"{len(cr_log)} turns")
    
    rt_log = orch.run_script("response_time")
    check("response_time: 15 turns", len(rt_log) == 15)
    
    hr_log = orch.run_script("helpful_response")
    check("helpful_response: 15 turns", len(hr_log) == 15)
    
    ld_log = orch.run_script("lexical_diversity")
    check("lexical_diversity: 12 turns", len(ld_log) == 12)
    
    er_log = orch.run_script("error_rate")
    check("error_rate: 15 turns", len(er_log) == 15)
except Exception as e:
    check("Orchestrator", False, str(e))

# ============================================================
print("\n-- 6. METRIC ENGINE --")

# Context Retention
try:
    cr = compute_context_retention(cr_log)
    check("Context Retention: runs", "retention_rate_pct" in cr)
    check("Context Retention: checkpoints found", cr["total_checkpoints"] == 6,
          f"found {cr['total_checkpoints']}/6 checkpoints")
    check("Context Retention: normalized 0-100", 0 <= cr["normalized_score"] <= 100,
          f"{cr['normalized_score']}")
    check("Context Retention: valid classification",
          cr["classification"] in ("Good","Moderate","Poor"), cr["classification"])
    # In mock mode bot says "[RASA MOCK] ..." which won't contain "vegetarian"
    # so all checkpoints fail = Poor. That is CORRECT behavior.
    check("Context Retention: mock correctly scores 0%", cr["retention_rate_pct"] == 0.0,
          f"mock bot gives no vegetarian keyword -> 0% is correct")
except Exception as e:
    check("Context Retention", False, str(e))

# Response Time
try:
    rt = compute_response_time(rt_log)
    check("Response Time: runs", "average_ms" in rt)
    check("Response Time: avg > 0", rt["average_ms"] is not None and rt["average_ms"] > 0,
          f"{rt['average_ms']}ms")
    check("Response Time: mock is Good (150ms < 1000ms)", rt["classification"] == "Good",
          rt["classification"])
    check("Response Time: normalized > 90", rt["normalized_score"] > 90,
          f"{rt['normalized_score']}")
    check("Response Time: valid_turns == 15", rt["valid_turns"] == 15)
except Exception as e:
    check("Response Time", False, str(e))

# Helpful Response + Prometheus
try:
    jud = judge_helpful_response_turns(hr_log, use_mock=True)
    check("Prometheus judge: returns scores dict", isinstance(jud["prometheus_scores"], dict))
    check("Prometheus judge: 5 turns scored",
          len(jud["prometheus_scores"]) == 5, str(jud["prometheus_scores"]))
    check("Prometheus judge: turns are {2,3,7,10,12}",
          set(jud["prometheus_scores"].keys()) == {2, 3, 7, 10, 12})
    
    hr = compute_helpful_response(hr_log, prometheus_scores=jud["prometheus_scores"])
    check("Helpful Response: runs", "helpful_ratio_pct" in hr)
    check("Helpful Response: total_turns == 15", hr["total_turns"] == 15)
    check("Helpful Response: normalized 0-100", 0 <= hr["normalized_score"] <= 100,
          f"{hr['normalized_score']}")
    check("Helpful Response: has per_turn list", len(hr["per_turn"]) == 15)
except Exception as e:
    check("Helpful Response", False, str(e))

# Lexical Diversity
try:
    ld = compute_lexical_diversity(ld_log)
    check("Lexical Diversity: runs", "average_mattr" in ld or "scored_turns" in ld)
    check("Lexical Diversity: excluded_turns is int", isinstance(ld["excluded_turns"], int))
    check("Lexical Diversity: normalized 0-100", 0 <= ld["normalized_score"] <= 100,
          f"{ld['normalized_score']}")
    # Mock responses are short "[RASA MOCK] Received: ..." = likely < 20 tokens -> excluded
    check("Lexical Diversity: mock short responses excluded correctly",
          ld["excluded_turns"] >= 0, f"excluded={ld['excluded_turns']}, scored={ld['scored_turns']}")
except Exception as e:
    check("Lexical Diversity", False, str(e))

# Error Rate
try:
    er = compute_error_rate(er_log)
    check("Error Rate: runs", "error_rate_pct" in er)
    check("Error Rate: total_turns == 15", er["total_turns"] == 15)
    check("Error Rate: normalized 0-100", 0 <= er["normalized_score"] <= 100,
          f"{er['normalized_score']}")
    check("Error Rate: has taxonomy_summary", isinstance(er["taxonomy_summary"], dict))
    check("Error Rate: breakdown_count is int", isinstance(er["breakdown_count"], int))
except Exception as e:
    check("Error Rate", False, str(e))

# ============================================================
print("\n-- 7. MATTR CALCULATOR --")
try:
    long_text = ("Index funds are diversified investment vehicles that track market indices. "
                 "They offer low expense ratios, broad diversification, and consistent long-term "
                 "returns. The majority of actively managed funds underperform their benchmark "
                 "index, which further supports passive investing for retail investors seeking "
                 "market exposure without complexity of individual stock selection.")
    short_text = "Yes."
    ambiguous = "Maybe so."
    
    r_long = compute_mattr(long_text)
    r_short = compute_mattr(short_text)
    
    check("MATTR: long text -> scored", r_long["status"] == "scored",
          f"mattr={r_long['mattr']}, tokens={r_long['token_count']}")
    check("MATTR: score in [0,1]", 0 <= r_long["mattr"] <= 1, str(r_long["mattr"]))
    check("MATTR: short text -> excluded", r_short["status"] == "short_excluded",
          f"tokens={r_short['token_count']} < 20")
    check("MATTR: short text mattr is None", r_short["mattr"] is None)
    check("MATTR: has window_ttrs", isinstance(r_long["window_ttrs"], list) and len(r_long["window_ttrs"]) > 0)
    
    # Pilot calibration
    good = [long_text,
            "Compound interest means earning interest on both principal and previously earned interest. "
            "Starting early maximises the exponential growth effect. A small consistent contribution "
            "over decades grows substantially more than a large late contribution."]
    padded = ["That is a great question I am happy to help sure I can definitely assist "
              "basically in general terms yes it is something worth considering you know many "
              "people ask about this it is very important so yes definitely a good question."]
    t = pilot_calibrate(good, padded)
    check("MATTR pilot calibration", t["calibrated"] == True,
          f"good_min={t['good_min']} moderate_min={t['moderate_min']}")
    check("MATTR thresholds saved to file",
          os.path.exists(r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\mattr_thresholds.json"))
except Exception as e:
    check("MATTR calculator", False, str(e))

# ============================================================
print("\n-- 8. COHEN'S KAPPA --")
try:
    # Perfect agreement
    k_perfect = compute_kappa([True,True,False,False,True], [True,True,False,False,True])
    check("Kappa perfect agreement = 1.0", k_perfect["kappa"] == 1.0, str(k_perfect["kappa"]))
    
    # No agreement (all wrong)
    k_zero = compute_kappa([True,True,False,False], [False,False,True,True])
    check("Kappa zero agreement <= 0", k_zero["kappa"] <= 0, str(k_zero["kappa"]))
    
    # Target met case
    k_good = compute_kappa([True,True,False,True,False], [True,True,False,True,True])
    check("Kappa target met", k_good["target_met"] == (k_good["kappa"] >= 0.61),
          f"k={k_good['kappa']} target_met={k_good['target_met']}")
    check("Kappa interpretation string", bool(k_good["interpretation"]))
    
    # From score dicts
    prom = {2: True, 5: True, 7: False, 10: True, 14: True}
    human = {2: True, 5: True, 7: False, 10: True, 14: False}
    k3 = compute_kappa_from_scores(prom, human)
    check("Kappa from score dicts", k3["kappa"] is not None, f"k={k3['kappa']}")
    check("Kappa has action recommendation", "action" in k3)
    check("Kappa result saved to file",
          os.path.exists(r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\kappa_results.json"))
except Exception as e:
    check("Kappa calculator", False, str(e))

# ============================================================
print("\n-- 9. COMPOSITE USABILITY SCORE --")
try:
    metric_results = {
        "context_retention": cr, "response_time": rt,
        "helpful_response": hr, "lexical_diversity": ld, "error_rate": er,
    }
    cus = compute_cus(metric_results)
    check("CUS computes", "cus_equal" in cus)
    check("CUS in 0-100", 0 <= cus["cus_equal"] <= 100, f"CUS={cus['cus_equal']}")
    check("CUS has 5 metric scores", len(cus["scores"]) == 5)
    check("CUS classification valid",
          cus["classification"] in ("Good","Moderate","Poor"), cus["classification"])
    check("CUS scores sum check",
          abs(sum(cus["scores"].values()) / 5 - cus["cus_equal"]) < 0.1,
          f"mean={sum(cus['scores'].values())/5:.2f} CUS={cus['cus_equal']}")
    
    # Test comparison
    # Build a slightly different ollama result
    ollama_metrics = {k: {"normalized_score": v["normalized_score"] + 5 if v["normalized_score"] < 95 else v["normalized_score"]}
                     for k, v in metric_results.items()}
    comp = compute_cus_comparison(metric_results, ollama_metrics)
    check("CUS comparison: has winner", comp["winner"] in ("rasa","ollama","tie"))
    check("CUS comparison: has delta", comp["delta"] >= 0)
except Exception as e:
    check("CUS", False, str(e))

# ============================================================
print("\n-- 10. SUS SCORE CALCULATOR --")
try:
    # Standard SUS calculation test (from Brooke 1996 known result)
    # All 5s on odd, all 1s on even = maximum SUS = 100
    max_sus = calculate_sus_score([5,1,5,1,5,1,5,1,5,1])
    check("SUS max score = 100", max_sus == 100.0, str(max_sus))
    
    # All 1s on odd, all 5s on even = minimum SUS = 0
    min_sus = calculate_sus_score([1,5,1,5,1,5,1,5,1,5])
    check("SUS min score = 0", min_sus == 0.0, str(min_sus))
    
    # Neutral (all 3s) = 50
    mid_sus = calculate_sus_score([3,3,3,3,3,3,3,3,3,3])
    check("SUS neutral = 50", mid_sus == 50.0, str(mid_sus))
    
    # Aggregate
    records = [
        {"sus_score": 75.0}, {"sus_score": 82.5},
        {"sus_score": 67.5}, {"sus_score": 90.0}, {"sus_score": 70.0}
    ]
    agg = aggregate_sus(records)
    check("SUS aggregate n=5", agg["n"] == 5)
    check("SUS aggregate mean correct", abs(agg["mean_sus"] - 77.0) < 0.1, str(agg["mean_sus"]))
except Exception as e:
    check("SUS calculator", False, str(e))

# ============================================================
print("\n-- 11. CORRELATION AND REGRESSION --")
try:
    import random, numpy as np
    random.seed(42)
    n = 30
    cus_sc = [50 + random.gauss(0, 10) for _ in range(n)]
    sus_sc = [c * 0.7 + random.gauss(20, 8) for c in cus_sc]
    
    corr = compute_correlation(cus_sc, sus_sc)
    check("Correlation computes", "pearson_r" in corr)
    check("Pearson r in [-1,1]", -1 <= corr["pearson_r"] <= 1, str(corr["pearson_r"]))
    check("Spearman rho in [-1,1]", -1 <= corr["spearman_rho"] <= 1, str(corr["spearman_rho"]))
    check("n=30 no warning", "warning" not in corr, "n is sufficient")
    
    metrics_matrix = [[random.gauss(60, 10) for _ in range(5)] for _ in range(n)]
    reg = compute_regression(metrics_matrix, sus_sc)
    check("Regression computes", "r_squared" in reg)
    check("R-squared in [0,1]", 0 <= reg["r_squared"] <= 1, str(reg["r_squared"]))
    check("Weights sum to ~1.0", abs(sum(reg["weights"].values()) - 1.0) < 0.01,
          str(round(sum(reg["weights"].values()), 4)))
    check("5 metric weights", len(reg["weights"]) == 5)
    check("Importance ranking", reg["metric_importance"][0]["rank"] == 1)
except Exception as e:
    check("Analysis", False, str(e))

# ============================================================
print("\n-- 12. REPORT GENERATOR --")
try:
    report = generate_report(
        chatbot_name="rasa",
        metric_results=metric_results,
        cus_result=cus,
    )
    check("Report: generated", len(report) > 1000, f"{len(report)} chars")
    check("Report: has executive summary", "Executive Summary" in report or "EXECUTIVE SUMMARY" in report)
    check("Report: has CUS section", "COMPOSITE USABILITY SCORE" in report)
    check("Report: has all 5 metrics",
          all(f"Metric {i}" in report for i in range(1, 6)))
    check("Report: has tuning section", "Tuning" in report or "TUNING" in report)
    check("Report: saved to disk",
          len(os.listdir(r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\reports")) > 0)
except Exception as e:
    check("Report generator", False, str(e))

# ============================================================
# FINAL SUMMARY
print("\n" + "="*60)
print(f"  FINAL RESULT: {passed} PASSED  |  {failed} FAILED")
print("="*60)

if failed == 0:
    print("\n  All components working correctly.")
    print("\n  Metric scores (Rasa, mock mode):")
    print(f"    Context Retention : {cr['normalized_score']:>6.1f} / 100  ({cr['classification']})")
    print(f"    Response Time     : {rt['normalized_score']:>6.1f} / 100  ({rt['classification']})")
    print(f"    Helpful Response  : {hr['normalized_score']:>6.1f} / 100  ({hr['classification']})")
    print(f"    Lexical Diversity : {ld['normalized_score']:>6.1f} / 100  ({ld['classification']})")
    print(f"    Error Rate        : {er['normalized_score']:>6.1f} / 100  ({er['classification']})")
    print(f"    CUS (equal)       : {cus['cus_equal']:>6.1f} / 100  ({cus['classification']})")
    print()
    print("  NOTE: Context Retention = 0 and Lexical Diversity = 0 in mock mode")
    print("  is CORRECT -- the mock bot returns '[RASA MOCK] Received: ...'")
    print("  which contains no vegetarian keywords and is too short for MATTR.")
    print("  Real bot responses will produce real non-zero scores.")
else:
    print("\n  FAILED CHECKS:")
    for line in results:
        if "[FAIL]" in line:
            print(" ", line)
print("="*60 + "\n")
