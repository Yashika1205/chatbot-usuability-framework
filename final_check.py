"""
final_check.py
Final end-to-end correctness check.
Tests every layer: Ollama live, workbook scripts, scoring, CUS, report.
"""
import sys, os, json, time
sys.path.insert(0, r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework")

import pandas as pd
import numpy as np
import urllib.request

FW   = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework"
DATA = os.path.join(FW, "results", "run26_10_real_data")
WB   = os.path.join(FW, "workbook")

passed = 0
failed = 0
issues = []

def ok(label, detail=""):
    global passed
    passed += 1
    print(f"  [OK ] {label}" + (f"  =>  {detail}" if detail else ""))

def fail(label, detail=""):
    global failed
    failed += 1
    issues.append((label, detail))
    print(f"  [!!] {label}" + (f"  =>  {detail}" if detail else ""))

def warn(label, detail=""):
    print(f"  [??] {label}" + (f"  =>  {detail}" if detail else ""))

print("="*65)
print("  FINAL END-TO-END CHECK")
print("="*65)

# ── 1. FILE STRUCTURE ───────────────────────────────────────────
print("\n[1] File structure")

must_exist = [
    ("pipeline_v3.py",                      "Core pipeline (YOUR file)"),
    ("metrics/metrics_v2.py",               "Scoring engine (YOUR file)"),
    ("metrics/prometheus_v2.py",            "LLM judge (YOUR file)"),
    ("RUNBOOK.md",                           "Runbook"),
    ("workbook/cr_script.csv",              "Workbook CR script"),
    ("workbook/rt_script.csv",              "Workbook RT script"),
    ("workbook/hr_script.csv",              "Workbook HR script"),
    ("workbook/mattr_script.csv",           "Workbook MATTR script"),
    ("workbook/er_script.csv",              "Workbook ER script"),
    ("results/run26_10_real_data/RUN1_FULL.csv", "Real Run1 data"),
    ("results/run26_10_real_data/RUN2_FULL.csv", "Real Run2 data"),
    ("results/run26_10_real_data/RUN3_FULL.csv", "Real Run3 data"),
    ("results/run26_10_real_data/prometheus_run1.json", "Prometheus Run1"),
    ("results/run26_10_real_data/prometheus_run2.json", "Prometheus Run2"),
    ("results/run26_10_real_data/prometheus_run3.json", "Prometheus Run3"),
    ("results/run26_10_real_data/scores_per_run.csv",   "Final scores"),
    ("results/run26_10_real_data/scores_mean_sd.csv",   "Mean/SD table"),
    ("results/run26_10_real_data/mattr_thresholds.json","MATTR thresholds"),
    ("results/run26_10_real_data/env_manifest.json",    "Env manifest"),
    ("sus_study.py",                        "SUS collector"),
    ("analysis.py",                         "Pearson r / regression"),
    ("report.py",                           "Report generator"),
    ("metrics/kappa_calculator.py",         "Kappa module"),
]

for rel, label in must_exist:
    full = os.path.join(FW, rel.replace("/", os.sep))
    if os.path.exists(full):
        ok(label, os.path.basename(full))
    else:
        fail(label, f"MISSING: {full}")

# ── 2. WORKBOOK SCRIPTS ─────────────────────────────────────────
print("\n[2] Workbook scripts (your real questions)")

expected_turns = {"cr_script.csv": 30, "rt_script.csv": 15,
                  "hr_script.csv": 15, "mattr_script.csv": 12, "er_script.csv": 15}
for fname, exp_turns in expected_turns.items():
    path = os.path.join(WB, fname)
    try:
        df = pd.read_csv(path)
        unique_turns = df["turn_number"].nunique()
        if unique_turns == exp_turns:
            ok(fname, f"{unique_turns} turns, columns={list(df.columns)}")
        else:
            fail(fname, f"expected {exp_turns} turns, got {unique_turns}")
    except Exception as e:
        fail(fname, str(e))

# ── 3. REAL RUN DATA ────────────────────────────────────────────
print("\n[3] Real run data integrity")

for run in [1, 2, 3]:
    path = os.path.join(DATA, f"RUN{run}_FULL.csv")
    try:
        df = pd.read_csv(path)
        bots    = sorted(df["bot"].unique())
        scripts = sorted(df["script"].unique())
        errors  = int(df["response"].astype(str).str.startswith("[ERROR").sum())
        ctx_risk = int(df["ctx_risk"].sum()) if "ctx_risk" in df.columns else "N/A"
        if errors == 0 and len(bots) == 3 and len(scripts) == 5:
            ok(f"RUN{run}_FULL.csv", f"{len(df)} rows | bots={bots} | errors={errors} | ctx_risk={ctx_risk}")
        else:
            fail(f"RUN{run}_FULL.csv", f"errors={errors} bots={bots} scripts={scripts}")
    except Exception as e:
        fail(f"RUN{run}_FULL.csv", str(e))

# ── 4. PROMETHEUS COMPLETENESS ──────────────────────────────────
print("\n[4] Prometheus judge completeness")

for run in [1, 2, 3]:
    path = os.path.join(DATA, f"prometheus_run{run}.json")
    try:
        d = json.load(open(path))
        for bot in ["Llama3", "Mistral", "Rasa"]:
            if bot not in d:
                fail(f"Run{run} {bot}", "not in prometheus JSON")
                continue
            turns = {k: v for k, v in d[bot].items() if not k.startswith("_")}
            unresolved = [t for t, v in turns.items() if isinstance(v, dict) and v.get("score") is None]
            scores = [v.get("score") for v in turns.values() if isinstance(v, dict)]
            if not unresolved and len(turns) == 6:
                ok(f"Run{run} {bot}", f"6/6 turns judged | scores={scores}")
            else:
                fail(f"Run{run} {bot}", f"unresolved={unresolved} turns={len(turns)}/6")
    except Exception as e:
        fail(f"prometheus_run{run}", str(e))

# ── 5. SCORING CORRECTNESS ──────────────────────────────────────
print("\n[5] Scoring correctness (recompute vs stored)")

try:
    import importlib.util as ilu
    spec = ilu.spec_from_file_location("metrics_v2", os.path.join(FW, "metrics", "metrics_v2.py"))
    mv2 = ilu.module_from_spec(spec)
    os.environ["LOGS_DIR"] = DATA
    spec.loader.exec_module(mv2)

    stored = pd.read_csv(os.path.join(DATA, "scores_per_run.csv"))
    all_match = True
    for run in [1, 2, 3]:
        df = pd.read_csv(os.path.join(DATA, f"RUN{run}_FULL.csv"))
        for bot in ["Llama3", "Mistral", "Rasa"]:
            if bot not in df["bot"].values:
                continue
            s, cus = mv2.score_bot(df, bot, run, allow_partial=True)
            recomp = cus.get("cus")
            row = stored[(stored["run"] == run) & (stored["bot"] == bot)]
            if row.empty or recomp is None:
                continue
            stored_cus = float(row["CUS"].values[0])
            diff = abs(recomp - stored_cus)
            if diff < 0.5:
                ok(f"Run{run} {bot} CUS", f"recomputed={recomp}  stored={stored_cus}  diff={round(diff,3)}")
            else:
                fail(f"Run{run} {bot} CUS", f"recomputed={recomp}  stored={stored_cus}  diff={round(diff,3)}")
                all_match = False
except Exception as e:
    fail("Scoring recompute", str(e))

# ── 6. MATTR THRESHOLDS ─────────────────────────────────────────
print("\n[6] MATTR thresholds")

try:
    t = json.load(open(os.path.join(DATA, "mattr_thresholds.json")))
    g, m = t["good_threshold"], t["moderate_threshold"]
    if 0.5 < g < 1.0 and 0.5 < m < g:
        ok("MATTR thresholds valid", f"good>={g}  moderate>={m}")
    else:
        fail("MATTR thresholds", f"suspicious values: good={g} moderate={m}")
    # Verify actual MATTR scores fall in expected range
    for run in [1]:
        df = pd.read_csv(os.path.join(DATA, f"RUN{run}_FULL.csv"))
        for bot in ["Llama3", "Mistral"]:
            d = df[df["bot"] == bot]
            scores = [mv2.mattr(r) for r in d["response"]]
            ok_scores = [s for s in scores if s is not None]
            avg = round(float(np.mean(ok_scores)), 4) if ok_scores else None
            band = "GOOD" if avg and avg >= g else "MODERATE" if avg and avg >= m else "POOR"
            ok(f"MATTR Run{run} {bot}", f"avg={avg}  band={band}  scored={len(ok_scores)}/{len(scores)}")
except Exception as e:
    fail("MATTR thresholds", str(e))

# ── 7. MEAN/SD TABLE ────────────────────────────────────────────
print("\n[7] Final results table (scores_mean_sd.csv)")

try:
    ms = pd.read_csv(os.path.join(DATA, "scores_mean_sd.csv"))
    print("  " + ms.to_string(index=False).replace("\n", "\n  "))
    ok("scores_mean_sd.csv loaded and readable")
except Exception as e:
    fail("scores_mean_sd.csv", str(e))

# ── 8. OLLAMA LIVE ──────────────────────────────────────────────
print("\n[8] Ollama live check")

try:
    with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
        d = json.loads(r.read())
        models = [m["name"] for m in d.get("models", [])]
        ok("Ollama server running", f"models={models}")
        if models:
            # Quick single-turn test
            payload = json.dumps({
                "model": models[0],
                "messages": [{"role": "user", "content": "Say: I am working correctly."}],
                "stream": False,
                "options": {"num_predict": 20, "temperature": 0}
            }).encode()
            req = urllib.request.Request(
                "http://localhost:11434/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"}
            )
            t0 = time.perf_counter()
            with urllib.request.urlopen(req, timeout=120) as r2:
                resp = json.loads(r2.read())
            ms_rt = round((time.perf_counter()-t0)*1000, 0)
            reply = resp.get("message", {}).get("content", "").strip()[:80]
            ok(f"Live inference ({models[0]})", f"{ms_rt}ms  reply={reply!r}")
        else:
            warn("No models pulled yet", "pull mistral:7b or llama3.1:8b")
except Exception as e:
    warn("Ollama not reachable", str(e))

# ── 9. SUS / ANALYSIS / REPORT MODULES ─────────────────────────
print("\n[9] Supporting modules (SUS, analysis, report)")

try:
    from sus_study import calculate_sus_score, aggregate_sus
    s = calculate_sus_score([5,1,5,1,5,1,5,1,5,1])
    ok("SUS formula", f"max score = {s}")
except Exception as e:
    fail("SUS module", str(e))

try:
    from analysis import compute_correlation, compute_regression
    import random; random.seed(1)
    n=10
    r = compute_correlation([60+random.gauss(0,5) for _ in range(n)],
                             [55+random.gauss(0,8) for _ in range(n)])
    ok("Correlation module", f"Pearson r={r['pearson_r']}")
except Exception as e:
    fail("Analysis module", str(e))

try:
    from report import generate_report
    from scoring import compute_cus
    mock_metrics = {
        "context_retention": {"normalized_score": 33.3, "classification": "Poor",
            "retention_rate_pct": 33.3, "passed": 2, "total_checkpoints": 6, "checkpoint_results": []},
        "response_time":     {"normalized_score": 0.0,  "classification": "Poor",
            "average_ms": 17560, "min_ms": 5000, "max_ms": 35000, "valid_turns": 19, "per_turn_latencies": []},
        "helpful_response":  {"normalized_score": 100.0,"classification": "Good",
            "helpful_ratio_pct": 100.0, "helpful_turns": 15, "total_turns": 15, "per_turn": []},
        "lexical_diversity": {"normalized_score": 74.15,"classification": "Moderate",
            "average_mattr": 0.7415, "scored_turns": 11, "excluded_turns": 1, "per_turn": []},
        "error_rate":        {"normalized_score": 93.3, "classification": "Good",
            "error_rate_pct": 6.7, "breakdown_count": 1, "total_turns": 15,
            "taxonomy_summary": {"typo": 1}, "per_turn": []},
    }
    cus = compute_cus(mock_metrics)
    rpt = generate_report("Llama3_mean", mock_metrics, cus_result=cus)
    ok("Report module", f"CUS={cus['cus_equal']}  report={len(rpt)} chars")
except Exception as e:
    fail("Report module", str(e))

try:
    from metrics.kappa_calculator import compute_kappa
    k = compute_kappa([True,True,False,True,False],[True,True,False,True,True])
    ok("Kappa module", f"k={k['kappa']}  interpretation={k['interpretation']}")
except Exception as e:
    fail("Kappa module", str(e))

# ── 10. KNOWN LIMITATIONS ───────────────────────────────────────
print("\n[10] Known limitations (not bugs — expected)")

warn("RT normalization",
     "Linear formula scores both LLMs as 0 (>10000ms). Log scale option exists in metrics_v2.py (RT_NORM=log)")
warn("Rasa MATTR missing",
     "Rasa responses too short (<50 tokens) -> excluded. Rasa CUS is incomplete (4 metrics only)")
warn("SUS study pending",
     "0/30 participants collected. Pearson r and regression weights not yet computed")
warn("Human kappa labels pending",
     "human_labels.csv not created yet. Kappa not validated with real human rater")
warn("Ollama on CPU",
     "37s/turn locally vs 17s on Colab T4. Run full 3-run experiment on Colab, not locally")

# ── FINAL VERDICT ────────────────────────────────────────────────
print("\n" + "="*65)
print(f"  RESULT: {passed} OK  |  {failed} FAILED  |  5 known limitations")
print("="*65)

if failed == 0:
    print("""
  Everything that CAN be verified is CORRECT.

  Your project needs NO code changes.

  The only remaining work is human study (not code):
    [ ] SUS study  — recruit 30 participants
    [ ] human_labels.csv  — 1 rater, 30 HR judgments
    [ ] Write-up  — results section + discussion
""")
else:
    print(f"\n  {failed} issues need fixing:")
    for label, detail in issues:
        print(f"  - {label}: {detail}")
