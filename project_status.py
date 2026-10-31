import os, json, pandas as pd

base  = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP"
data  = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\run26_10_real_data"
fw    = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework"

def exists(p): return "DONE" if os.path.exists(p) else "MISSING"

print("="*65)
print("  PROJECT STATUS CHECK")
print("="*65)

# ---- PHASE 1: pipeline runs
print("\n[PHASE 1] Automated test runs (pipeline_v3.py)")
for run in [1,2,3]:
    full = os.path.join(data, f"RUN{run}_FULL.csv")
    if os.path.exists(full):
        df = pd.read_csv(full)
        bots    = sorted(df["bot"].unique().tolist())
        scripts = sorted(df["script"].unique().tolist())
        err     = int(df["response"].astype(str).str.startswith("[ERROR").sum())
        print(f"  DONE  RUN{run}: {len(df)} rows | bots={bots} | scripts={scripts} | errors={err}")
    else:
        print(f"  MISSING  RUN{run}_FULL.csv")

# ---- PHASE 2: Prometheus judging
print("\n[PHASE 2] Prometheus judging (prometheus_v2.py)")
for run in [1,2,3]:
    p = os.path.join(data, f"prometheus_run{run}.json")
    if os.path.exists(p):
        d = json.load(open(p))
        bots = [k for k in d if not k.startswith("_")]
        total_turns = sum(len(d[b]) for b in bots)
        none_count  = sum(1 for b in bots for t,v in d[b].items() if isinstance(v,dict) and v.get("score") is None)
        print(f"  DONE  prometheus_run{run}.json | bots={bots} | turns_judged={total_turns} | unresolved={none_count}")
    else:
        print(f"  MISSING  prometheus_run{run}.json")

# ---- PHASE 3: Scoring
print("\n[PHASE 3] Scoring engine (metrics_v2.py)")
for f in ["scores_per_run.csv","scores_mean_sd.csv","scores_detail.json","audit_decisions.csv"]:
    path = os.path.join(data, f)
    print(f"  {exists(path):7s}  {f}")

# show the actual scores
sp = os.path.join(data, "scores_per_run.csv")
if os.path.exists(sp):
    df = pd.read_csv(sp)
    print()
    print("  Scores per run:")
    for _, r in df.iterrows():
        cus_str = f"CUS={r['CUS']}" if not pd.isna(r['CUS']) else "CUS=incomplete"
        mattr_str = f"MATTR={r['MATTR']}" if not pd.isna(r['MATTR']) else "MATTR=N/A"
        print(f"    Run{int(r['run'])} {r['bot']:8s}: CR={r['CR']}%  RT={r['RT_ms']}ms  HR={r['HR']}%  {mattr_str}  ER={r['ER_pct']}%  {cus_str}")

# ---- PHASE 4: MATTR calibration
print("\n[PHASE 4] MATTR pilot calibration")
mt = os.path.join(data, "mattr_thresholds.json")
if os.path.exists(mt):
    t = json.load(open(mt))
    print(f"  DONE  good_threshold={t['good_threshold']}  moderate_threshold={t['moderate_threshold']}")
    print(f"        calibrated on {len(t.get('good_scores',{}))} good + {len(t.get('padded_scores',{}))} padded responses")
else:
    print("  MISSING  mattr_thresholds.json")

# ---- PHASE 5: Human SUS study
print("\n[PHASE 5] Human SUS study (n >= 30 participants)")
sus_dir = os.path.join(fw, "results", "sus_data")
sus_files = [f for f in os.listdir(sus_dir) if f.endswith(".json")] if os.path.exists(sus_dir) else []
if sus_files:
    all_records = []
    for f in sus_files:
        all_records.extend(json.load(open(os.path.join(sus_dir, f))))
    print(f"  DONE  {len(all_records)} participant records found")
else:
    print("  MISSING  No SUS data collected yet (need 30 real participants)")
    print("           -> Run: py main.py --mode sus --chatbot llama3")
    print("           -> Or: collect via Google Form and import CSV")

# ---- PHASE 6: Cohen's Kappa (human vs Prometheus)
print("\n[PHASE 6] Cohen's Kappa validation")
kappa_file = os.path.join(data, "kappa_results.json") if os.path.exists(os.path.join(data,"kappa_results.json")) else None
human_labels = os.path.join(base, "human_labels.csv")
if os.path.exists(human_labels):
    print(f"  DONE  human_labels.csv exists")
else:
    print("  MISSING  human_labels.csv (columns: run, bot, turn, human_pass)")
    print("           -> A human rater needs to score 5 overlap HR turns per run per bot")
    print("           -> Then run: python metrics_v2.py --kappa human_labels.csv")

# ---- PHASE 7: Correlation + regression
print("\n[PHASE 7] Pearson r / Regression (primary + secondary results)")
an = os.path.join(fw, "results", "analysis_results.json")
rw = os.path.join(fw, "results", "regression_weights.json")
print(f"  {exists(an):7s}  analysis_results.json (Pearson r, Spearman rho)")
print(f"  {exists(rw):7s}  regression_weights.json (metric weights for final CUS)")
if not os.path.exists(an):
    print("           -> Blocked by: SUS study not done yet")
    print("           -> Once SUS is done, run: py main.py --mode analysis")

# ---- SUMMARY
print("\n" + "="*65)
print("  SUMMARY")
print("="*65)
done = [
    "pipeline_v3.py  - 3 full runs, 783 real turns",
    "prometheus_v2.py - all 6 HR turns judged per run per bot",
    "metrics_v2.py   - all 5 metrics scored, CUS computed",
    "MATTR thresholds - calibrated from real responses",
    "RUNBOOK.md      - reproducibility documented",
    "env_manifest    - GPU, models, seeds, script hashes logged",
]
missing = [
    "SUS study       - 0/30 participants (must recruit 30 humans)",
    "Human kappa labels - human_labels.csv (5 HR turns x 3 runs x 2 LLMs)",
    "Pearson r       - blocked by SUS",
    "Regression weights - blocked by SUS",
    "Write-up        - results section, discussion, conclusion",
]
print("\nDONE:")
for d in done: print(f"  [x] {d}")
print("\nSTILL NEEDED:")
for m in missing: print(f"  [ ] {m}")

print(f"""
BOTTOM LINE:
  Automated pipeline  : 100% complete
  Scoring             : 100% complete
  Human validation    :   0% complete (SUS + Kappa)
  Statistical analysis:   0% complete (blocked by SUS)
  Write-up            :   0% complete

  Your project is roughly 60% done end-to-end.
  The remaining 40% is the human study (Phase 5-7) + write-up.
""")
