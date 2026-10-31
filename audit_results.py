"""
audit_results.py - checks the actual run data against the scores claimed in scores_per_run.csv
"""
import json, os, re
import pandas as pd
import numpy as np

BASE = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\run2710_extracted\run26.10"

print("="*70)
print("  AUDIT: run27.10.zip RESULTS")
print("="*70)

# ============================================================
# 1. Load summary tables
# ============================================================
per_run = pd.read_csv(os.path.join(BASE, "scores_per_run.csv"))
mean_sd = pd.read_csv(os.path.join(BASE, "scores_mean_sd.csv"))
audit_df = pd.read_csv(os.path.join(BASE, "audit_decisions.csv"))

print("\n-- scores_per_run.csv --")
print(per_run.to_string(index=False))

print("\n-- scores_mean_sd.csv --")
print(mean_sd.to_string(index=False))

# ============================================================
# 2. Cross-check: recount CR checkpoints from raw CSVs
# ============================================================
print("\n" + "="*70)
print("CROSS-CHECK 1: Context Retention (CR) - recount checkpoints from raw CSVs")
print("="*70)

CHECKPOINT_TURNS = {5, 10, 15, 20, 25, 30}
VEG_KEYWORDS = ["vegetarian","vegan","veggie","meatless","meat-free","plant-based",
                "no meat","without meat","mushroom tofu","tofu","tempeh"]
DENIAL_PHRASES = ["haven't discussed","have not discussed","start of our conversation",
                  "beginning of our conversation","first message"]

def check_veg_in_response(resp):
    if not resp or str(resp).startswith("[ERROR") or str(resp).lower() in ("nan","none",""):
        return False, "EMPTY"
    r = str(resp).lower()
    for d in DENIAL_PHRASES:
        if d in r:
            return False, "MEMORY_DENIAL"
    for kw in VEG_KEYWORDS:
        if kw in r:
            return True, f"found: '{kw}'"
    # Also check meat words not negated (simple check)
    meat_words = ["chicken","beef","pork","fish","salmon","steak","bacon","ham","prosciutto"]
    found_meat = [m for m in meat_words if m in r]
    if found_meat:
        return False, f"meat suggested: {found_meat[:3]}"
    # No veg keyword but also no meat -> ambiguous, count as fail (conservative)
    return False, "no_veg_keyword"

for run in [1, 2, 3]:
    for bot in ["Rasa", "Llama3", "Mistral"]:
        fname = os.path.join(BASE, f"run{run}", f"RUN{run}_CR_{bot}.csv")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}")
            continue
        df = pd.read_csv(fname)
        cp_rows = df[df["turn_number"].isin(CHECKPOINT_TURNS)]
        if cp_rows.empty:
            print(f"  Run{run} {bot}: no checkpoint rows found")
            continue
        passed = 0
        details = []
        for _, row in cp_rows.iterrows():
            p, reason = check_veg_in_response(row["response"])
            if p:
                passed += 1
            details.append(f"T{int(row['turn_number'])}:{'P' if p else 'F'}")
        score = round(100 * passed / len(cp_rows), 1)
        # Compare to reported
        reported_row = per_run[(per_run["run"] == run) & (per_run["bot"] == bot)]
        reported = float(reported_row["CR"].values[0]) if len(reported_row) else None
        match = "OK" if reported is not None and abs(score - reported) < 5 else f"MISMATCH(got {score}, reported {reported})"
        print(f"  Run{run} {bot}: {passed}/{len(cp_rows)} checkpoints = {score}%  [{' '.join(details)}]  -> {match}")

# ============================================================
# 3. Cross-check: Response Time
# ============================================================
print("\n" + "="*70)
print("CROSS-CHECK 2: Response Time - recompute average latency from raw CSVs")
print("="*70)

for run in [1, 2, 3]:
    for bot in ["Rasa", "Llama3", "Mistral"]:
        fname = os.path.join(BASE, f"run{run}", f"RUN{run}_RT_{bot}.csv")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}")
            continue
        df = pd.read_csv(fname)
        valid = df[df["latency_ms"] > 0]["latency_ms"].astype(float)
        if valid.empty:
            print(f"  Run{run} {bot}: no valid latency rows")
            continue
        avg = round(valid.mean(), 1)
        reported_row = per_run[(per_run["run"] == run) & (per_run["bot"] == bot)]
        reported = float(reported_row["RT_ms"].values[0]) if len(reported_row) else None
        match = "OK" if reported is not None and abs(avg - reported) < 50 else f"MISMATCH(got {avg}, reported {reported})"
        # Check for burst turns
        if "turn_type" in df.columns:
            burst = df[df["turn_type"].astype(str).str.contains("burst", case=False) & (df["latency_ms"] > 0)]
            normal = df[~df["turn_type"].astype(str).str.contains("burst", case=False) & (df["latency_ms"] > 0)]
            binfo = f"  burst_avg={round(burst['latency_ms'].mean(),0) if len(burst) else 'N/A'} normal_avg={round(normal['latency_ms'].mean(),0) if len(normal) else 'N/A'}"
        else:
            binfo = ""
        print(f"  Run{run} {bot}: avg={avg}ms  n={len(valid)}{binfo}  -> {match}")

# ============================================================
# 4. Cross-check: Error Rate
# ============================================================
print("\n" + "="*70)
print("CROSS-CHECK 3: Error Rate - count error turns from raw CSVs")
print("="*70)

ERROR_MARKERS = ["i don't understand","i'm not sure","i cannot","i can't","sorry, i don't",
                 "could you clarify","[error","[no response]","i'm unable","i am unable"]

for run in [1, 2, 3]:
    for bot in ["Rasa", "Llama3", "Mistral"]:
        fname = os.path.join(BASE, f"run{run}", f"RUN{run}_ER_{bot}.csv")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}")
            continue
        df = pd.read_csv(fname)
        # Only count non-T1 rows (T1 are setup turns for contradiction pairs)
        if "noise_type" in df.columns:
            scored = df[~df["noise_type"].astype(str).str.endswith("_t1")]
        else:
            scored = df
        n = len(scored)
        breakdown = 0
        for _, row in scored.iterrows():
            resp = str(row["response"]).lower()
            if not resp.strip() or resp.startswith("[error"):
                breakdown += 1
                continue
            for marker in ERROR_MARKERS:
                if marker in resp:
                    breakdown += 1
                    break
        rate = round(100 * breakdown / n, 1) if n else None
        reported_row = per_run[(per_run["run"] == run) & (per_run["bot"] == bot)]
        reported = float(reported_row["ER_pct"].values[0]) if len(reported_row) else None
        match = "OK (within 15%)" if reported is not None and abs(rate - reported) < 15 else f"DIFFERS(got {rate}%, reported {reported}%)"
        print(f"  Run{run} {bot}: {breakdown}/{n} = {rate}%  -> {match}")

# ============================================================
# 5. Cross-check: MATTR scores
# ============================================================
print("\n" + "="*70)
print("CROSS-CHECK 4: MATTR - recompute from raw CSVs")
print("="*70)

WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")

def compute_mattr_local(text, window=50, min_tokens=50):
    if not text or str(text).startswith("[ERROR") or str(text).lower() in ("nan","none",""):
        return None
    tok = WORD.findall(str(text).lower())
    if len(tok) < max(min_tokens, window):
        return None
    sc = [len(set(tok[i:i+window]))/window for i in range(len(tok)-window+1)]
    return float(np.mean(sc))

thresholds_path = os.path.join(BASE, "mattr_thresholds.json")
with open(thresholds_path) as f:
    thr = json.load(f)
GOOD_T = thr["good_threshold"]
MOD_T  = thr["moderate_threshold"]
print(f"  Using thresholds: Good>={GOOD_T}, Moderate>={MOD_T}")

for run in [1, 2, 3]:
    for bot in ["Rasa", "Llama3", "Mistral"]:
        fname = os.path.join(BASE, f"run{run}", f"RUN{run}_MATTR_{bot}.csv")
        if not os.path.exists(fname):
            print(f"  MISSING: {fname}")
            continue
        df = pd.read_csv(fname)
        scores = [compute_mattr_local(r) for r in df["response"]]
        ok = [s for s in scores if s is not None]
        if not ok:
            print(f"  Run{run} {bot}: all responses too short for MATTR (excluded)")
            continue
        avg = round(float(np.mean(ok)), 4)
        band = "GOOD" if avg >= GOOD_T else ("MODERATE" if avg >= MOD_T else "POOR")
        reported_row = per_run[(per_run["run"] == run) & (per_run["bot"] == bot)]
        reported = per_run[(per_run["run"] == run) & (per_run["bot"] == bot)]["MATTR"]
        reported_val = float(reported.values[0]) if len(reported) and not pd.isna(reported.values[0]) else None
        match = "OK" if reported_val is not None and abs(avg - reported_val) < 0.02 else f"MISMATCH(got {avg}, reported {reported_val})"
        print(f"  Run{run} {bot}: MATTR={avg} ({band})  scored={len(ok)}/{len(scores)}  -> {match}")

# ============================================================
# 6. Prometheus judge scores check
# ============================================================
print("\n" + "="*70)
print("CROSS-CHECK 5: Prometheus judge scores (prometheus_run*.json)")
print("="*70)

for run in [1, 2, 3]:
    ppath = os.path.join(BASE, f"prometheus_run{run}.json")
    if not os.path.exists(ppath):
        print(f"  MISSING: prometheus_run{run}.json")
        continue
    with open(ppath) as f:
        pdata = json.load(f)
    for bot in ["Rasa", "Llama3", "Mistral"]:
        if bot not in pdata:
            print(f"  Run{run} {bot}: not in prometheus_run{run}.json")
            continue
        bot_scores = pdata[bot]
        meta = pdata.get("_meta", {})
        pass_min = meta.get("pass_min", 4)
        turns = {t: v for t, v in bot_scores.items() if t != "_meta"}
        scores_list = [(t, v.get("score"), v.get("pass")) for t, v in turns.items()]
        n_scored = sum(1 for _, s, _ in scores_list if s is not None)
        n_pass = sum(1 for _, _, p in scores_list if p is True)
        n_none = sum(1 for _, s, _ in scores_list if s is None)
        detail = " ".join(f"T{t}:{s if s else 'None'}" for t, s, _ in sorted(scores_list, key=lambda x: int(x[0])))
        print(f"  Run{run} {bot}: {n_scored} turns judged  {n_pass}/{n_scored} pass (>={pass_min})  scores=[{detail}]  errors={n_none}")

# ============================================================
# 7. CUS recomputation check
# ============================================================
print("\n" + "="*70)
print("CROSS-CHECK 6: CUS recomputation from reported metric scores")
print("="*70)

for run in [1, 2, 3]:
    for bot in ["Llama3", "Mistral"]:  # Skip Rasa (MATTR missing)
        row = per_run[(per_run["run"] == run) & (per_run["bot"] == bot)]
        if row.empty:
            continue
        r = row.iloc[0]
        cr   = float(r["CR"]) if not pd.isna(r["CR"]) else None
        rt_ms= float(r["RT_ms"]) if not pd.isna(r["RT_ms"]) else None
        hr   = float(r["HR"]) if not pd.isna(r["HR"]) else None
        mattr_v = float(r["MATTR"]) if not pd.isna(r["MATTR"]) else None
        er   = float(r["ER_pct"]) if not pd.isna(r["ER_pct"]) else None

        # Normalize same way as metrics_v2.py
        n = {}
        if cr is not None:    n["CR"]    = cr
        if rt_ms is not None: n["RT"]    = max(0, 100 - rt_ms/100)
        if hr is not None:    n["HR"]    = hr
        if mattr_v is not None: n["MATTR"] = mattr_v * 100
        if er is not None:    n["ER"]    = max(0, 100 - er)

        computed_cus = round(sum(n.values()) / len(n), 2) if n else None
        reported_cus = float(r["CUS"]) if not pd.isna(r["CUS"]) else None
        match = "OK" if reported_cus is not None and computed_cus is not None and abs(computed_cus - reported_cus) < 0.5 else f"MISMATCH"

        print(f"  Run{run} {bot}: CR={cr} RT_norm={round(max(0,100-rt_ms/100),1) if rt_ms else None} "
              f"HR={hr} MATTR_norm={round(mattr_v*100,1) if mattr_v else None} ER_norm={round(max(0,100-er),1) if er else None}")
        print(f"           computed_CUS={computed_cus}  reported_CUS={reported_cus}  -> {match}")

# ============================================================
# 8. Overall verdict
# ============================================================
print("\n" + "="*70)
print("OVERALL FINDINGS")
print("="*70)

print("""
Bots tested    : Rasa, Llama3.1, Mistral (3 runs each on Google Colab T4)
Scripts run    : CR (30 turns), RT (15), HR (15), MATTR (12), ER (15)
Total turns    : 87 turns/bot/run x 3 bots x 3 runs = 783 logged turn rows

Mean scores (from scores_mean_sd.csv):
  Llama3  CUS = 59.64  CR=33.3%  RT=17560ms  HR=100%  MATTR=0.74  ER=8.9%
  Mistral CUS = 51.54  CR=11.1%  RT=10369ms  HR=84.4%  MATTR=0.76  ER=20.0%
  Rasa    CUS = 24.x   CR=0%     RT=<300ms   HR=0%    MATTR=N/A   ER=100%
""")
