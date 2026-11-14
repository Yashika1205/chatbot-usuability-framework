"""
pipeline.py — Automated Usability Testing Framework
Sends scripted conversations to chatbots, logs all responses.
Domain-agnostic: works with any CSV script on any topic.
"""

import requests, time, json, pandas as pd, nltk
from datetime import datetime
from pathlib import Path

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

# ── CONFIG ────────────────────────────────────────────────────────────
RASA_URL      = "http://localhost:5005/webhooks/rest/webhook"
OLLAMA_URL    = "http://localhost:11434/api/generate"
LLAMA_MODEL   = "llama3.1"
MISTRAL_MODEL = "mistral"
WORKBOOK_DIR  = Path("workbook")
LOGS_DIR      = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

# ── MATTR: computed on every turn regardless of script ────────────────
def compute_mattr(text, window=50):
    """
    Moving Average Type-Token Ratio (MATTR-50).
    Script-agnostic: measures vocabulary richness on any text.
    Returns None if response is too short or degenerate.
    Citation: Covington & McFall (2010) J.Quantitative Linguistics 17(2).
    """
    r = str(text).strip().lower()
    if not r or r.startswith("[error") or r in ["[no response]", "nan"]:
        return None

    try:
        tokens = nltk.word_tokenize(r)
    except Exception:
        tokens = r.split()

    if len(tokens) < 20:
        return None                          # too short to score reliably

    unique_ratio = len(set(tokens)) / len(tokens)
    if unique_ratio < 0.1:
        return None                          # degenerate repetition

    if len(tokens) < window:
        return round(len(set(tokens)) / len(tokens), 4)

    scores = [len(set(tokens[i:i + window])) / window
              for i in range(len(tokens) - window + 1)]
    return round(sum(scores) / len(scores), 4)


# ── BOT SENDERS ───────────────────────────────────────────────────────
def send_rasa(text, sender="tester"):
    t = datetime.now()
    try:
        r = requests.post(RASA_URL,
                          json={"sender": sender, "message": text},
                          timeout=30)
        recv = datetime.now()
        replies = [x["text"] for x in r.json() if "text" in x]
        return {"response":   " ".join(replies) or "[no response]",
                "latency_ms": round((recv - t).total_seconds() * 1000, 2)}
    except Exception as e:
        return {"response": f"[ERROR: {e}]", "latency_ms": -1}


def send_ollama(model, text, retries=2):
    for attempt in range(retries + 1):
        t = datetime.now()
        try:
            r = requests.post(OLLAMA_URL,
                              json={"model": model,
                                    "prompt": text,
                                    "stream": False},
                              timeout=600)
            r.raise_for_status()
            recv = datetime.now()
            return {"response":   r.json().get("response", "[no response]"),
                    "latency_ms": round((recv - t).total_seconds() * 1000, 2)}
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"    Timeout — retry {attempt + 1}/{retries}...")
                time.sleep(5)
            else:
                return {"response": "[ERROR: timeout]", "latency_ms": -1}
        except Exception as e:
            return {"response": f"[ERROR: {e}]", "latency_ms": -1}


def send_llama(text):   return send_ollama(LLAMA_MODEL,   text)
def send_mistral(text): return send_ollama(MISTRAL_MODEL, text)


# ── WARMUP ────────────────────────────────────────────────────────────
def warmup(model):
    print(f"  Warming up {model}...")
    try:
        requests.post(OLLAMA_URL,
                      json={"model": model, "prompt": "hi", "stream": False},
                      timeout=300)
        print(f"  ✓ {model} ready.")
    except Exception as e:
        print(f"  ✗ {model} failed: {e}")


def warmup_rasa():
    print("  Warming up Rasa...")
    try:
        r = requests.post(RASA_URL,
                          json={"sender": "warmup", "message": "hi"},
                          timeout=30)
        print(f"  ✓ Rasa ready: {r.json()}")
    except Exception as e:
        print(f"  ✗ Rasa failed: {e}")


# ── LOAD CSV ──────────────────────────────────────────────────────────
def load_script(csv_name):
    path = WORKBOOK_DIR / csv_name
    df   = pd.read_csv(path)
    assert "question" in df.columns, f"{csv_name} missing 'question' column"
    assert "turn_number" in df.columns, f"{csv_name} missing 'turn_number' column"
    return df


# ── RUN ONE SCRIPT ON ONE BOT ─────────────────────────────────────────
def run_script(script_key, df, bot_name, send_fn):
    print(f"\n{'='*55}\n  {script_key} → {bot_name}\n{'='*55}")
    results = []

    for _, row in df.iterrows():
        question = str(row["question"])
        turn_num = int(row["turn_number"])

        print(f"  Turn {turn_num}: {question[:70]}...")
        result = send_fn(question)
        mattr  = compute_mattr(result["response"])

        entry = {
            "script":      script_key,
            "bot":         bot_name,
            "turn_number": turn_num,
            "question":    question,
            "response":    result["response"],
            "latency_ms":  result["latency_ms"],   # RT logged on EVERY turn
            "mattr":       mattr,                  # MATTR logged on EVERY turn
            "timestamp":   datetime.now().isoformat(),
            "environment": "Google_Colab_T4",
        }
        # carry ALL CSV columns forward (pass_criteria, turn_type, etc.)
        for col in df.columns:
            if col not in entry:
                entry[col] = row[col]

        results.append(entry)
        print(f"    Latency: {result['latency_ms']}ms | MATTR: {mattr}")
        print(f"    Response: {result['response'][:120]}")
        time.sleep(1)

    return results


# ── SAVE (immediately after each script-bot combo) ────────────────────
def save(results, tag):
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{tag}_{ts}"
    with open(LOGS_DIR / f"{name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    pd.DataFrame(results).to_csv(LOGS_DIR / f"{name}.csv", index=False,
                                 encoding="utf-8")
    print(f"  ✓ Saved: logs/{name}.csv")
    return name


# ── MAIN ──────────────────────────────────────────────────────────────
def main(num_runs=3):
    scripts = {
        "CR":    "cr_script.csv",
        "RT":    "rt_script.csv",
        "HR":    "hr_script.csv",
        "MATTR": "mattr_script.csv",
        "ER":    "er_script.csv",
    }
    bots = {
        "Rasa":    send_rasa,
        "Llama3":  send_llama,
        "Mistral": send_mistral,
    }

    print("Warming up all bots...")
    warmup_rasa()
    warmup(LLAMA_MODEL)
    warmup(MISTRAL_MODEL)
    print("Waiting 15s for models to settle...")
    time.sleep(15)

    for run_num in range(1, num_runs + 1):
        print(f"\n{'#'*55}\n  RUN {run_num} of {num_runs}\n{'#'*55}")
        all_results = []

        for script_key, csv_file in scripts.items():
            print(f"\nLoading {csv_file}...")
            try:
                df = load_script(csv_file)
                print(f"  → {len(df)} rows loaded")
            except Exception as e:
                print(f"  → FAILED: {e}")
                continue

            for bot_name, send_fn in bots.items():
                results = run_script(script_key, df, bot_name, send_fn)
                all_results.extend(results)
                # Save immediately — protects against Colab disconnection
                save(results, f"RUN{run_num}_{script_key}_{bot_name}")
                time.sleep(2)

        save(all_results, f"RUN{run_num}_FULL")
        print(f"\n✅ Run {run_num} complete.")

    print("\n✅ All runs complete. Run metrics.py next.")


if __name__ == "__main__":
    main(num_runs=3)
