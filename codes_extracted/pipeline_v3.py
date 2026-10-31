"""
pipeline_v3.py - Automated Usability Testing Framework (orchestrator)

What changed vs pipeline_v2.py (and why):
  1. BOT-MAJOR LOOP. Old loop was run > script > bot, so Llama and Mistral alternated on
     every script. With OLLAMA_MAX_LOADED_MODELS=1 that forces a model reload at the first
     turn of every script and pollutes RT. Now: run > bot > script. The model is loaded once
     per bot per run, outside the timed region, and the previous model is explicitly unloaded.
  2. ER = one fresh conversation per noise case (T1+T2 contradiction pairs stay together).
     Otherwise turn 8 "when will it arrive" is answerable from the order number given in
     turn 1 and a correct answer would be scored as a failure.
  3. Context-window safety. Ollama silently drops the OLDEST messages when the prompt
     exceeds num_ctx, and the oldest message is exactly the "I'm vegetarian" anchor.
     NUM_CTX raised to 32768 and every turn logs est_prompt_tokens + ctx_risk.
  4. A failed turn creates a history gap, which silently corrupts any later CR checkpoint.
     Gaps are counted per row (history_gap) so the scorer can mark those checkpoints INVALID.
  5. Reproducibility: sampling options + seed recorded, model digests, Ollama version, GPU,
     script hashes written to logs/env_manifest.json.
  6. Resume: finished script x bot x run files are skipped, so a Colab disconnect costs
     one script, not the whole run. Write LOGS_DIR to Google Drive.
  7. Extra per-turn Ollama timings (load / prompt-eval / generation, token counts) so the
     paper can separate "slow because long answer" from "slow model".
  8. Rapid-burst turns are sent back-to-back (no 1 s sleep) as the script requires.

Usage (Colab cell):
  !python pipeline_v3.py --smoke                      # 5-turn CR check on Llama3
  !python pipeline_v3.py --runs 1 2 3                 # full experiment
  !python pipeline_v3.py --runs 1 --bots Llama3 --scripts CR     # partial
"""

import argparse, hashlib, json, os, platform, subprocess, sys, time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

# --------------------------------------------------------------------------- CONFIG
RASA_URL        = os.getenv("RASA_URL", "http://localhost:5005/webhooks/rest/webhook")
OLLAMA_BASE     = os.getenv("OLLAMA_BASE", "http://localhost:11434")
OLLAMA_CHAT_URL = f"{OLLAMA_BASE}/api/chat"
LLAMA_MODEL     = os.getenv("LLAMA_MODEL", "llama3.1")
MISTRAL_MODEL   = os.getenv("MISTRAL_MODEL", "mistral")

WORKBOOK_DIR = Path(os.getenv("WORKBOOK_DIR", "workbook"))
LOGS_DIR     = Path(os.getenv("LOGS_DIR", "logs"))        # point this at Google Drive on Colab

NUM_CTX      = int(os.getenv("NUM_CTX", 32768))   # CR = 30 turns x ~500 tok can pass 16k
NUM_PREDICT  = int(os.getenv("NUM_PREDICT", 0)) or None   # e.g. 700 to cap answer length
TEMPERATURE  = float(os.getenv("TEMPERATURE", 0.7))
KEEP_ALIVE   = "60m"
CTX_RISK_FRACTION = 0.85            # flag a turn if estimated prompt > 85% of NUM_CTX
CHARS_PER_TOKEN   = 3.0             # conservative (real English is ~4)

SCRIPTS = {                         # key -> (csv, history policy)
    "CR":    ("cr_script.csv",    "full"),
    "RT":    ("rt_script.csv",    "full"),
    "HR":    ("hr_script.csv",    "full"),
    "MATTR": ("mattr_script.csv", "full"),
    "ER":    ("er_script.csv",    "per_case"),   # reset context for every turn_number
}
BOT_ORDER = ["Rasa", "Llama3", "Mistral"]
REQUIRED_COLS = {"CR": {"turn_number", "question", "turn_type", "pass_criteria"},
                 "RT": {"turn_number", "question", "turn_type"},
                 "HR": {"turn_number", "question", "judge_type", "pass_criteria"},
                 "MATTR": {"turn_number", "question"},
                 "ER": {"turn_number", "question", "noise_type", "pass_criteria"}}

OLLAMA_MODELS = {"Llama3": LLAMA_MODEL, "Mistral": MISTRAL_MODEL}


def ollama_options(run, idx):
    o = {"num_ctx": NUM_CTX, "temperature": TEMPERATURE,
         "seed": run * 1000 + idx}             # different per run, reproducible
    if NUM_PREDICT:
        o["num_predict"] = NUM_PREDICT
    return o


# --------------------------------------------------------------------------- BOTS
def send_rasa(text, history, sender, run, idx):
    t0 = time.perf_counter()
    try:
        r = requests.post(RASA_URL, json={"sender": sender, "message": text}, timeout=60)
        r.raise_for_status()
        ms = round((time.perf_counter() - t0) * 1000, 2)
        replies = [x["text"] for x in r.json() if "text" in x]
        return {"response": " ".join(replies) or "[no response]", "latency_ms": ms, "meta": {}}
    except Exception as e:
        return {"response": f"[ERROR: {e}]", "latency_ms": -1, "meta": {}}


def send_ollama(model, text, history, run, idx, retries=2):
    messages = list(history) + [{"role": "user", "content": text}]
    payload = {"model": model, "messages": messages, "stream": False,
               "keep_alive": KEEP_ALIVE, "options": ollama_options(run, idx)}
    for attempt in range(retries + 1):
        t0 = time.perf_counter()
        try:
            r = requests.post(OLLAMA_CHAT_URL, json=payload, timeout=900)
            r.raise_for_status()
            ms = round((time.perf_counter() - t0) * 1000, 2)
            j = r.json()
            content = (j.get("message") or {}).get("content", "")
            if not content.strip():
                raise RuntimeError("empty completion")
            meta = {k: j.get(k) for k in ("total_duration", "load_duration", "prompt_eval_count",
                                          "prompt_eval_duration", "eval_count", "eval_duration",
                                          "done_reason")}
            return {"response": content, "latency_ms": ms, "meta": meta}
        except Exception as e:
            if attempt < retries:
                print(f"    ! {type(e).__name__}: {e} - retry {attempt + 1}/{retries}")
                time.sleep(5)
            else:
                return {"response": f"[ERROR: {e}]", "latency_ms": -1, "meta": {}}


def get_sender(bot):
    if bot == "Rasa":
        return send_rasa
    model = OLLAMA_MODELS[bot]
    return lambda text, history, sender, run, idx: send_ollama(model, text, history, run, idx)


# --------------------------------------------------------------------------- MODEL MGMT
def unload(model):
    try:
        requests.post(f"{OLLAMA_BASE}/api/chat",
                      json={"model": model, "messages": [], "keep_alive": 0}, timeout=60)
    except Exception:
        pass


def prepare_bot(bot):
    """Load the bot OUTSIDE the timed region; unload the other LLM (T4 has 15 GB)."""
    if bot == "Rasa":
        for m in OLLAMA_MODELS.values():
            unload(m)
        try:
            r = requests.post(RASA_URL, json={"sender": "warmup", "message": "hi"}, timeout=60)
            print(f"  Rasa ready: {r.json()}")
        except Exception as e:
            raise SystemExit(f"Rasa not reachable at {RASA_URL}: {e}")
        return
    for other, m in OLLAMA_MODELS.items():
        if other != bot:
            unload(m)
    model = OLLAMA_MODELS[bot]
    print(f"  Loading {model} (num_ctx={NUM_CTX}) ...")
    t0 = time.time()
    r = requests.post(OLLAMA_CHAT_URL,
                      json={"model": model, "stream": False, "keep_alive": KEEP_ALIVE,
                            "messages": [{"role": "user", "content": "hi"}],
                            "options": {"num_ctx": NUM_CTX, "num_predict": 8}}, timeout=900)
    r.raise_for_status()
    print(f"  {model} ready in {time.time() - t0:.0f}s")


# --------------------------------------------------------------------------- MANIFEST
def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def write_manifest(args):
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    m = {"created": datetime.now().isoformat(), "python": sys.version.split()[0],
         "platform": platform.platform(), "pandas": pd.__version__,
         "num_ctx": NUM_CTX, "num_predict": NUM_PREDICT, "temperature": TEMPERATURE,
         "seed_rule": "run*1000 + row_index", "history_policy": {k: v[1] for k, v in SCRIPTS.items()},
         "scripts_sha256_16": {k: sha(WORKBOOK_DIR / v[0]) for k, v in SCRIPTS.items()
                               if (WORKBOOK_DIR / v[0]).exists()}}
    try:
        m["ollama_version"] = requests.get(f"{OLLAMA_BASE}/api/version", timeout=10).json()
        tags = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=10).json().get("models", [])
        m["ollama_models"] = {t["name"]: t.get("digest", "")[:16] for t in tags}
    except Exception as e:
        m["ollama_error"] = str(e)
    try:
        m["gpu"] = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                                  capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        m["gpu"] = "unknown"
    path = LOGS_DIR / "env_manifest.json"
    if path.exists():                                     # keep the first, append later ones
        old = json.loads(path.read_text())
        old = old if isinstance(old, list) else [old]
        old.append(m); m = old
    path.write_text(json.dumps(m, indent=2))


# --------------------------------------------------------------------------- SCRIPT RUN
def load_script(key):
    csv, _ = SCRIPTS[key]
    df = pd.read_csv(WORKBOOK_DIR / csv)
    missing = REQUIRED_COLS[key] - set(df.columns)
    assert not missing, f"{csv} missing columns {missing}"
    assert df["question"].notna().all(), f"{csv}: empty question cell"
    return df.reset_index(drop=True)


def est_tokens(history, text):
    chars = sum(len(m["content"]) for m in history) + len(text)
    return int(chars / CHARS_PER_TOKEN)


def run_script(key, df, bot, run, limit=None):
    policy = SCRIPTS[key][1]
    send = get_sender(bot)
    print(f"\n{'=' * 60}\n  RUN {run} | {key} -> {bot} | history={policy}\n{'=' * 60}")
    rows, history, gap = [], [], 0
    sender = f"r{run}-{key}-{bot}-{int(time.time())}"
    prev_turn = None
    for idx, row in df.iterrows():
        if limit and idx >= limit:
            break
        turn = int(row["turn_number"])
        q = str(row["question"])
        if policy == "per_case" and turn != prev_turn:
            history, sender = [], f"{sender.rsplit('-', 1)[0]}-{int(time.time() * 1000)}"
        prev_turn = turn
        n_before = len(history)
        tok = est_tokens(history, q)
        res = send(q, history, sender, run, idx)
        failed = str(res["response"]).startswith("[ERROR") or res["latency_ms"] < 0
        if failed:
            gap += 1
        elif bot != "Rasa":
            history += [{"role": "user", "content": q}, {"role": "assistant", "content": res["response"]}]
        m = res["meta"]
        entry = {"run": run, "script": key, "bot": bot, "row_index": idx, "turn_number": turn,
                 "question": q, "response": res["response"], "latency_ms": res["latency_ms"],
                 "history_msgs": n_before, "history_gap": gap,
                 "est_prompt_tokens": tok, "ctx_risk": tok > CTX_RISK_FRACTION * NUM_CTX,
                 "eval_count": m.get("eval_count"), "prompt_eval_count": m.get("prompt_eval_count"),
                 "load_ms": round((m.get("load_duration") or 0) / 1e6, 1),
                 "gen_ms": round((m.get("eval_duration") or 0) / 1e6, 1),
                 "done_reason": m.get("done_reason"),
                 "timestamp": datetime.now().isoformat(), "environment": os.getenv("ENV_NAME", "Colab_T4")}
        for c in df.columns:                   # carry every CSV column (criteria, types ...)
            if c not in entry:
                entry[c] = row[c]
        rows.append(entry)
        flag = " CTX-RISK!" if entry["ctx_risk"] else ""
        print(f"  [{key} {turn:>2}] {res['latency_ms']:>9}ms  hist={n_before:>2}  "
              f"~{tok}tok{flag}  {str(res['response'])[:90]!r}")
        nxt = df.iloc[idx + 1]["turn_type"] if idx + 1 < len(df) and "turn_type" in df.columns else ""
        cur = row.get("turn_type", "")
        burst = str(cur).startswith("rapid_burst") and str(nxt).startswith("rapid_burst")
        time.sleep(0 if burst else 1)
    return rows


def save_combo(rows, run, key, bot):
    d = LOGS_DIR / f"run{run}"
    d.mkdir(parents=True, exist_ok=True)
    stem = d / f"RUN{run}_{key}_{bot}"
    df = pd.DataFrame(rows)
    df.to_csv(f"{stem}.csv", index=False, encoding="utf-8")
    Path(f"{stem}.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return stem


def combo_done(run, key, bot, expected):
    p = LOGS_DIR / f"run{run}" / f"RUN{run}_{key}_{bot}.csv"
    if not p.exists():
        return False
    try:
        d = pd.read_csv(p)
        return len(d) == expected and not d["response"].astype(str).str.startswith("[ERROR").any()
    except Exception:
        return False


def merge_run(run, scripts, bots):
    parts = []
    for key in scripts:
        for bot in bots:
            p = LOGS_DIR / f"run{run}" / f"RUN{run}_{key}_{bot}.csv"
            if p.exists():
                parts.append(pd.read_csv(p))
    if parts:
        full = pd.concat(parts, ignore_index=True)
        full.to_csv(LOGS_DIR / f"RUN{run}_FULL.csv", index=False, encoding="utf-8")
        n_err = int(full["response"].astype(str).str.startswith("[ERROR").sum())
        print(f"\n  RUN{run}_FULL.csv: {len(full)} rows, {n_err} error rows, "
              f"{int(full['ctx_risk'].sum())} ctx-risk rows")


# --------------------------------------------------------------------------- SMOKE TEST
def smoke(bot):
    """B's step 2: first 5 CR turns; turn 5 must NOT deny memory and must not suggest meat."""
    prepare_bot(bot)
    df = load_script("CR")
    rows = run_script("CR", df, bot, run=0, limit=5)
    r5 = rows[-1]["response"].lower()
    denial = any(p in r5 for p in ("haven't discussed", "have not discussed", "start of our conversation",
                                   "beginning of our conversation", "first message"))
    meat = [w for w in ("chicken", "beef", "pork", "fish", "salmon", "steak", "bacon", "shrimp") if w in r5]
    print("\nSMOKE RESULT")
    print("  turn-5 history_msgs      :", rows[-1]["history_msgs"], "(must be 8)")
    print("  memory denial            :", denial)
    print("  meat words in turn-5 dinner:", meat)
    ok = rows[-1]["history_msgs"] == 8 and not denial
    print("  =>", "OK to start the full run" if ok else "STOP: history is not reaching the model")
    return 0 if ok else 1


# --------------------------------------------------------------------------- MAIN
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--bots", nargs="+", default=BOT_ORDER)
    ap.add_argument("--scripts", nargs="+", default=list(SCRIPTS))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-bot", default="Llama3")
    ap.add_argument("--force", action="store_true", help="redo combos that already exist")
    args = ap.parse_args()

    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    if args.smoke:
        sys.exit(smoke(args.smoke_bot))
    write_manifest(args)

    data = {k: load_script(k) for k in args.scripts}
    print("Rows per script:", {k: len(v) for k, v in data.items()}, "-> total", sum(map(len, data.values())))

    for run in args.runs:
        print(f"\n{'#' * 60}\n  RUN {run}\n{'#' * 60}")
        for bot in args.bots:
            todo = [k for k in args.scripts if args.force or not combo_done(run, k, bot, len(data[k]))]
            if not todo:
                print(f"  {bot}: all scripts already done for run {run}")
                continue
            prepare_bot(bot)
            for key in todo:
                rows = run_script(key, data[key], bot, run)
                stem = save_combo(rows, run, key, bot)
                print(f"  saved {stem}.csv")
                time.sleep(2)
        merge_run(run, list(SCRIPTS), BOT_ORDER)     # always merge every bot found on disk
    print("\nAll runs complete. Next: prometheus_v2.py --run 1 2 3, then metrics_v2.py")


if __name__ == "__main__":
    main()
