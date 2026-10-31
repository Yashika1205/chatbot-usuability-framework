"""
Simple focused verification — runs all key components and writes results to file.
"""
import sys, os
sys.path.insert(0, r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework")
os.environ["RASA_MOCK"] = "1"
os.environ["OLLAMA_MOCK"] = "1"
os.environ["PROMETHEUS_MOCK"] = "1"

out_path = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\verify_output.txt"
lines = []

def log(msg):
    lines.append(msg)

# 1. Imports
from adapters.rasa_adapter import RasaAdapter
from adapters.ollama_adapter import OllamaAdapter
from orchestrator import Orchestrator
from metrics.metric_engine import (compute_context_retention, compute_response_time,
    compute_helpful_response, compute_lexical_diversity, compute_error_rate)
from metrics.prometheus_judge import judge_helpful_response_turns
from metrics.kappa_calculator import compute_kappa, compute_kappa_from_scores
from metrics.mattr_calculator import compute_mattr, pilot_calibrate
from scoring import compute_cus, compute_cus_comparison
from report import generate_report
log("All imports OK")

# 2. Quick adapter test
rasa = RasaAdapter(use_mock=True)
r = rasa.send("Hello")
log(f"Rasa mock: {r['response'][:40]}  latency={r['latency_ms']}ms")

# 3. Full orchestrator run
orch = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0, verbose=False)
cr_log = orch.run_script("context_retention")
rt_log = orch.run_script("response_time")
hr_log = orch.run_script("helpful_response")
ld_log = orch.run_script("lexical_diversity")
er_log = orch.run_script("error_rate")
log(f"Scripts run: CR={len(cr_log)}, RT={len(rt_log)}, HR={len(hr_log)}, LD={len(ld_log)}, ER={len(er_log)} turns")

# 4. Metrics
cr = compute_context_retention(cr_log)
rt = compute_response_time(rt_log)
jud = judge_helpful_response_turns(hr_log, use_mock=True)
hr = compute_helpful_response(hr_log, prometheus_scores=jud["prometheus_scores"])
ld = compute_lexical_diversity(ld_log)
er = compute_error_rate(er_log)

log(f"Context Retention:  {cr['retention_rate_pct']}% ({cr['classification']})  score={cr['normalized_score']}")
log(f"Response Time:      avg={rt['average_ms']}ms ({rt['classification']})  score={rt['normalized_score']}")
log(f"Helpful Response:   {hr['helpful_ratio_pct']}% ({hr['classification']})  score={hr['normalized_score']}")
log(f"Lexical Diversity:  MATTR={ld['average_mattr']} ({ld['classification']})  score={ld['normalized_score']}")
log(f"Error Rate:         {er['error_rate_pct']}% ({er['classification']})  score={er['normalized_score']}")
log(f"Prometheus scores:  {jud['prometheus_scores']}")
log(f"Error taxonomy:     {er['taxonomy_summary']}")

# 5. Kappa
k = compute_kappa([True, True, False, True, False], [True, True, False, False, False])
log(f"Kappa test:         k={k['kappa']} ({k['interpretation']}) target_met={k['target_met']}")

# 6. MATTR calibration
good = ["Index funds are diversified investment vehicles that track a market index. They offer low expense ratios, broad diversification, and consistent long-term returns making them suitable for retail investors who want market exposure without the complexity of individual stock selection or active portfolio management strategies.", "Compound interest is the process by which interest accrues on both the principal amount and the previously earned interest. This creates an exponential growth effect over time that significantly increases wealth when investments are held for long periods."]
padded = ["That is a great question. I am happy to help you with that. Sure I can definitely assist. Basically in general terms what you are asking is a good thing. I would say that yes it is definitely something worth considering you know."]
thresholds = pilot_calibrate(good, padded)
log(f"MATTR thresholds:   good_min={thresholds['good_min']} moderate_min={thresholds['moderate_min']}")

# 7. CUS
metric_results = {
    "context_retention": cr, "response_time": rt,
    "helpful_response": hr, "lexical_diversity": ld, "error_rate": er,
}
cus = compute_cus(metric_results)
log(f"CUS (equal):        {cus['cus_equal']} / 100 ({cus['classification']})")
log(f"CUS scores:         {cus['scores']}")

# 8. Report
report = generate_report("rasa", metric_results, cus_result=cus)
log(f"Report size:        {len(report)} chars")
log(f"Report has CUS:     {'COMPOSITE USABILITY SCORE' in report}")
log(f"Report has metrics: {'Metric 1' in report}")

# 9. Comparison (run ollama too)
ollama = OllamaAdapter(use_mock=True)
orch2 = Orchestrator(adapter=ollama, chatbot_name="ollama", inter_turn_delay=0, verbose=False)
cr2 = compute_context_retention(orch2.run_script("context_retention"))
rt2 = compute_response_time(orch2.run_script("response_time"))
jud2 = judge_helpful_response_turns(orch2.run_script("helpful_response"), use_mock=True)
hr2 = compute_helpful_response(orch2.run_script("helpful_response"), prometheus_scores=jud2["prometheus_scores"])
ld2 = compute_lexical_diversity(orch2.run_script("lexical_diversity"))
er2 = compute_error_rate(orch2.run_script("error_rate"))
m2 = {"context_retention": cr2, "response_time": rt2, "helpful_response": hr2, "lexical_diversity": ld2, "error_rate": er2}
comp = compute_cus_comparison(metric_results, m2)
log(f"Comparison: Rasa={comp['rasa']['cus_equal']} Ollama={comp['ollama']['cus_equal']} Winner={comp['winner']} Delta={comp['delta']}")

# Write output
log("\n=== ALL CHECKS COMPLETE ===")
output = "\n".join(lines)
with open(out_path, "w", encoding="utf-8") as f:
    f.write(output)
print(output)
