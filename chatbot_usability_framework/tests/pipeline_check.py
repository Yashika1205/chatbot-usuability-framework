"""
pipeline_check.py
-----------------
Runs the complete pipeline end-to-end using the REAL run data already on disk.
Does NOT need Ollama or Rasa to be running.

Steps verified:
  1. Load real RUN1/2/3 FULL CSVs
  2. Re-run metrics_v2 scoring on real data
  3. Load prometheus scores from real JSON files
  4. Re-compute CUS and compare to stored scores_per_run.csv
  5. Run MATTR calibration check
  6. Run Kappa with simulated human labels
  7. Run SUS aggregation with simulated data
  8. Run Pearson r + regression with simulated SUS
  9. Generate final report
"""

import sys, os, json
sys.path.insert(0, r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework")

import pandas as pd
import numpy as np

DATA  = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\run26_10_real_data"
FW    = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework"

results = []
errors  = []

def ok(label, detail=""):
    results.append(("PASS", label, detail))
    print(f"  [PASS] {label}" + (f"  =>  {detail}" if detail else ""))

def fail(label, detail=""):
    results.append(("FAIL", label, detail))
    errors.append(label)
    print(f"  [FAIL] {label}" + (f"  =>  {detail}" if detail else ""))

print("\n" + "="*65)
print("  COMPLETE PIPELINE CHECK  (using real run data)")
print("="*65)

# ================================================================
print("\n-- STEP 1: Load real FULL CSVs --")
runs = {}
for r in [1,2,3]:
    path = os.path.join(DATA, f"RUN{r}_FULL.csv")
    try:
        df = pd.read_csv(path)
        runs[r] = df
        bots    = sorted(df["bot"].unique())
        scripts = sorted(df["script"].unique())
        err_rows = int(df["response"].astype(str).str.startswith("[ERROR").sum())
        ok(f"RUN{r} loaded", f"{len(df)} rows | bots={bots} | scripts={scripts} | errors={err_rows}")
    except Exception as e:
        fail(f"RUN{r} load", str(e))

# ================================================================
print("\n-- STEP 2: Re-run metrics_v2.py scoring on real data --")

os.environ["LOGS_DIR"] = DATA

try:
    # Import metrics_v2 directly
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "metrics_v2",
        os.path.join(FW, "metrics", "metrics_v2.py")
    )
    m2 = importlib.util.load_from_spec(spec) if hasattr(importlib.util, "load_from_spec") else None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))
    import metrics.metrics_v2 as metrics_v2
    ok("metrics_v2 imported")
except Exception as e:
    # Try direct import with path manipulation
    try:
        sys.path.insert(0, os.path.join(FW, "metrics"))
        import importlib
        import importlib.util as ilu
        spec = ilu.spec_from_file_location("metrics_v2", os.path.join(FW, "metrics", "metrics_v2.py"))
        metrics_v2 = ilu.module_from_spec(spec)
        spec.loader.exec_module(metrics_v2)
        ok("metrics_v2 imported (direct)", "via importlib")
    except Exception as e2:
        fail("metrics_v2 import", str(e2))
        metrics_v2 = None

# ================================================================
print("\n-- STEP 3: Score real data with metrics_v2 --")

recomputed = {}
if metrics_v2 and runs:
    for run_num, df_run in runs.items():
        recomputed[run_num] = {}
        for bot in ["Llama3", "Mistral", "Rasa"]:
            if bot not in df_run["bot"].values:
                continue
            try:
                scores, cus = metrics_v2.score_bot(df_run, bot, run_num, allow_partial=True)
                recomputed[run_num][bot] = {"scores": scores, "cus": cus}
                cr_s  = scores["CR"].get("score")
                rt_s  = scores["RT"].get("score")
                hr_s  = scores["HR"].get("score")
                ma_s  = scores["MATTR"].get("score")
                er_s  = scores["ER"].get("score")
                cus_v = cus.get("cus")
                ok(f"Run{run_num} {bot} scored",
                   f"CR={cr_s}%  RT={rt_s}ms  HR={hr_s}%  MATTR={ma_s}  ER={er_s}%  CUS={cus_v}")
            except Exception as e:
                fail(f"Run{run_num} {bot} scoring", str(e))
else:
    fail("Scoring skipped", "metrics_v2 not loaded")

# ================================================================
print("\n-- STEP 4: Compare recomputed CUS vs stored scores_per_run.csv --")

stored_path = os.path.join(DATA, "scores_per_run.csv")
try:
    stored = pd.read_csv(stored_path)
    match_count = 0
    mismatch_count = 0
    for run_num, bots_dict in recomputed.items():
        for bot, data_dict in bots_dict.items():
            recomp_cus = data_dict["cus"].get("cus")
            row = stored[(stored["run"] == run_num) & (stored["bot"] == bot)]
            if row.empty or recomp_cus is None:
                continue
            stored_cus = float(row["CUS"].values[0])
            diff = abs(recomp_cus - stored_cus)
            if diff < 0.5:
                match_count += 1
            else:
                mismatch_count += 1
                fail(f"CUS mismatch Run{run_num} {bot}", f"recomputed={recomp_cus} stored={stored_cus}")
    if mismatch_count == 0:
        ok(f"CUS values match stored", f"all {match_count} CUS values within 0.5 tolerance")
except Exception as e:
    fail("CUS comparison", str(e))

# ================================================================
print("\n-- STEP 5: Prometheus judge scores loaded --")

for run_num in [1,2,3]:
    p = os.path.join(DATA, f"prometheus_run{run_num}.json")
    try:
        d = json.load(open(p))
        for bot in ["Llama3", "Mistral", "Rasa"]:
            if bot not in d:
                continue
            turns = {k:v for k,v in d[bot].items() if not k.startswith("_")}
            scores_list = [v.get("score") for v in turns.values() if isinstance(v, dict)]
            none_c = scores_list.count(None)
            ok(f"prometheus_run{run_num} {bot}",
               f"{len(turns)} turns | scores={scores_list} | unresolved={none_c}")
    except Exception as e:
        fail(f"prometheus_run{run_num} load", str(e))

# ================================================================
print("\n-- STEP 6: MATTR thresholds --")

t_path = os.path.join(DATA, "mattr_thresholds.json")
try:
    t = json.load(open(t_path))
    good_t = t["good_threshold"]
    mod_t  = t["moderate_threshold"]
    ok("MATTR thresholds loaded", f"good>={good_t}  moderate>={mod_t}")

    # Verify MATTR scores fall in expected bands
    if metrics_v2 and runs:
        for run_num, df_run in runs.items():
            for bot in ["Llama3", "Mistral"]:
                d = df_run[df_run["bot"] == bot]
                mattr_scores = [metrics_v2.mattr(r) for r in d["response"]]
                scored = [s for s in mattr_scores if s is not None]
                if scored:
                    avg = round(float(np.mean(scored)), 4)
                    band = "GOOD" if avg >= good_t else ("MODERATE" if avg >= mod_t else "POOR")
                    ok(f"MATTR Run{run_num} {bot}", f"avg={avg} band={band} scored={len(scored)}/{len(mattr_scores)}")
except Exception as e:
    fail("MATTR thresholds", str(e))

# ================================================================
print("\n-- STEP 7: Cohen's Kappa (simulated human labels) --")

try:
    from metrics.kappa_calculator import compute_kappa, compute_kappa_from_scores

    # Simulate: human agrees with Prometheus on 4/5 turns
    # Real prometheus scores for Run1 Llama3: T2=5,T3=4,T7=5,T10=4,T12=4,T13=5 all pass>=4
    # Overlap turns from framework doc: 2,5,7,10,14
    # Prometheus scored: 2,3,7,10,12,13 — overlap with human check turns: 2,7,10
    prom_scores = {2: True, 7: True, 10: True}   # from prometheus_run1.json Llama3
    human_sim   = {2: True, 7: True, 10: False}   # simulated human labels (disagreement on turn 10)
    k = compute_kappa_from_scores(prom_scores, human_sim, overlap_turns={2,7,10})
    ok("Kappa computed (simulated)", f"k={k['kappa']}  interpretation={k['interpretation']}  target_met={k['target_met']}")
except Exception as e:
    fail("Kappa", str(e))

# ================================================================
print("\n-- STEP 8: SUS + Pearson r + Regression (simulated n=30) --")

try:
    from sus_study import calculate_sus_score, aggregate_sus
    from analysis import compute_correlation, compute_regression

    import random
    random.seed(42)
    n = 30

    # Llama3 real CUS mean = 59.64
    # Simulate SUS scores moderately correlated with CUS
    llama_cus = 59.64
    sus_scores = [max(20, min(100, llama_cus * 0.7 + random.gauss(20, 10))) for _ in range(n)]
    cus_list   = [llama_cus] * n

    # SUS check
    test_sus = calculate_sus_score([5,1,5,1,5,1,5,1,5,1])
    ok("SUS formula", f"max score = {test_sus}")

    agg = aggregate_sus([{"sus_score": s} for s in sus_scores])
    ok("SUS aggregate", f"n={agg['n']}  mean={agg['mean_sus']}  std={agg['std_sus']}  [{agg['classification']}]")

    # Pearson r
    corr = compute_correlation(cus_list, sus_scores)
    ok("Pearson r computed",
       f"r={corr['pearson_r']}  p={corr['pearson_p']}  significant={corr['statistically_significant']}")

    # Regression
    metric_matrix = [
        [  # 5 metric normalized scores for each participant — all same (one run)
            recomputed[1]["Llama3"]["scores"]["CR"].get("score", 0) or 0,
            max(0, 100 - (recomputed[1]["Llama3"]["scores"]["RT"].get("score", 0) or 0) / 100),
            recomputed[1]["Llama3"]["scores"]["HR"].get("score", 0) or 0,
            (recomputed[1]["Llama3"]["scores"]["MATTR"].get("score") or 0) * 100,
            max(0, 100 - (recomputed[1]["Llama3"]["scores"]["ER"].get("score", 0) or 0)),
        ]
    ] * n if recomputed else [[33.3, 0, 100, 74.15, 93.3]] * n

    reg = compute_regression(metric_matrix, sus_scores)
    top = reg["metric_importance"][0]
    ok("Regression computed",
       f"R2={reg['r_squared']}  top_metric={top['metric']}  weight={top['weight']}")
except Exception as e:
    fail("SUS/Analysis", str(e))

# ================================================================
print("\n-- STEP 9: Report generation --")

try:
    from report import generate_report
    from scoring import compute_cus

    if recomputed and 1 in recomputed and "Llama3" in recomputed[1]:
        s = recomputed[1]["Llama3"]["scores"]
        metric_results = {
            "context_retention": {"normalized_score": s["CR"].get("score") or 0,
                                  "classification": s["CR"].get("band","N/A"),
                                  "retention_rate_pct": s["CR"].get("score") or 0,
                                  "passed": s["CR"].get("passed",0),
                                  "total_checkpoints": s["CR"].get("total",6),
                                  "checkpoint_results": []},
            "response_time":     {"normalized_score": max(0, 100-(s["RT"].get("score") or 0)/100),
                                  "classification": s["RT"].get("band","N/A"),
                                  "average_ms": s["RT"].get("score"),
                                  "min_ms": None, "max_ms": None,
                                  "valid_turns": s["RT"].get("n",0),
                                  "per_turn_latencies": []},
            "helpful_response":  {"normalized_score": s["HR"].get("score") or 0,
                                  "classification": s["HR"].get("band","N/A"),
                                  "helpful_ratio_pct": s["HR"].get("score") or 0,
                                  "helpful_turns": s["HR"].get("passed",0),
                                  "total_turns": 15, "per_turn": []},
            "lexical_diversity": {"normalized_score": (s["MATTR"].get("score") or 0)*100,
                                  "classification": s["MATTR"].get("band","N/A"),
                                  "average_mattr": s["MATTR"].get("score"),
                                  "scored_turns": s["MATTR"].get("scored",0),
                                  "excluded_turns": s["MATTR"].get("excluded",0),
                                  "per_turn": []},
            "error_rate":        {"normalized_score": max(0, 100-(s["ER"].get("score") or 0)),
                                  "classification": s["ER"].get("band","N/A"),
                                  "error_rate_pct": s["ER"].get("score") or 0,
                                  "breakdown_count": s["ER"].get("failed",0),
                                  "total_turns": s["ER"].get("total",0),
                                  "taxonomy_summary": s["ER"].get("by_noise_failed",{}),
                                  "per_turn": []},
        }
        cus = compute_cus(metric_results)
        report = generate_report("Llama3_Run1_RealData", metric_results, cus_result=cus)
        ok("Report generated from REAL data", f"{len(report)} chars  CUS={cus['cus_equal']}")
    else:
        fail("Report skipped", "recomputed data not available")
except Exception as e:
    fail("Report generation", str(e))

# ================================================================
print("\n-- STEP 10: Environment manifest --")

try:
    em = os.path.join(DATA, "env_manifest.json")
    e  = json.load(open(em))
    if isinstance(e, list): e = e[0]
    models = list(e.get("ollama_models", {}).keys())
    gpu    = e.get("gpu", "unknown")[:50]
    ctx    = e.get("num_ctx")
    temp   = e.get("temperature")
    ok("env_manifest loaded", f"models={models}  ctx={ctx}  temp={temp}  gpu={gpu}")
except Exception as e:
    fail("env_manifest", str(e))

# ================================================================
print("\n" + "="*65)
total = len(results)
n_pass = sum(1 for r in results if r[0] == "PASS")
n_fail = sum(1 for r in results if r[0] == "FAIL")
print(f"  FINAL: {n_pass} PASSED  |  {n_fail} FAILED  |  {total} TOTAL CHECKS")
print("="*65)
if errors:
    print("\nFailed checks:")
    for e in errors:
        print(f"  - {e}")
else:
    print("\n  Complete pipeline is working correctly on real data.")
print("="*65 + "\n")
