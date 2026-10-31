"""
main.py
-------
Main entry point for the Chatbot Usability Testing Framework.

Usage
-----
# Full automated test run (both chatbots, all 5 scripts):
    py main.py --mode full

# Pilot run (3 turns of context_retention for smoke test):
    py main.py --mode pilot

# Single chatbot, single script:
    py main.py --mode script --chatbot ollama --script error_rate

# Run MATTR pilot calibration:
    py main.py --mode calibrate

# Run SUS human study collection (interactive CLI):
    py main.py --mode sus --chatbot rasa

# Run correlation + regression analysis after SUS study:
    py main.py --mode analysis

# Mock mode (no live chatbot required):
    py main.py --mode full --mock

Environment variables:
    RASA_URL           Rasa server URL (default: http://localhost:5005)
    OLLAMA_URL         Ollama server URL (default: http://localhost:11434)
    OLLAMA_MODEL       Ollama model (default: llama3.1:8b)
    RASA_MOCK=1        Force Rasa mock mode
    OLLAMA_MOCK=1      Force Ollama mock mode
    PROMETHEUS_MOCK=1  Force Prometheus 2 mock mode
    PROMETHEUS_MODEL   Judge model name (default: llama3.1:8b)
"""

import argparse
import os
import sys
import json

# Make sure the project root is in the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from adapters.rasa_adapter import RasaAdapter
from adapters.ollama_adapter import OllamaAdapter
from orchestrator import Orchestrator
from metrics.metric_engine import (
    compute_context_retention,
    compute_response_time,
    compute_helpful_response,
    compute_lexical_diversity,
    compute_error_rate,
)
from metrics.prometheus_judge import judge_helpful_response_turns
from metrics.kappa_calculator import compute_kappa_from_scores, collect_human_scores_interactive
from metrics.mattr_calculator import pilot_calibrate
from scoring import compute_cus, compute_cus_comparison
from sus_study import (
    collect_sus_interactive,
    load_sus_from_csv,
    save_sus_records,
    aggregate_sus,
    load_all_sus_for_chatbot,
)
from analysis import run_full_analysis
from report import generate_report

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHATBOT_NAMES = ["rasa", "ollama"]
ALL_SCRIPTS = [
    "context_retention",
    "response_time",
    "helpful_response",
    "lexical_diversity",
    "error_rate",
]

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


# ---------------------------------------------------------------------------
# Adapter factory
# ---------------------------------------------------------------------------
def make_adapter(chatbot_name: str, mock: bool = False):
    if chatbot_name == "rasa":
        adapter = RasaAdapter(use_mock=mock)
        if not mock and not adapter.health_check():
            print(f"  [Warning] Rasa server not reachable. Switching to mock mode.")
            adapter = RasaAdapter(use_mock=True)
    elif chatbot_name == "ollama":
        adapter = OllamaAdapter(use_mock=mock)
        if not mock and not adapter.health_check():
            print(f"  [Warning] Ollama server not reachable. Switching to mock mode.")
            adapter = OllamaAdapter(use_mock=True)
    else:
        raise ValueError(f"Unknown chatbot: {chatbot_name!r}")
    return adapter


# ---------------------------------------------------------------------------
# Metric runner
# ---------------------------------------------------------------------------
def run_metrics_on_log(
    script_name: str,
    log: list[dict],
    use_prometheus_mock: bool = False,
) -> dict:
    """
    Run the appropriate metric module(s) on a conversation log.

    Returns a dict of {metric_name: result_dict}.
    """
    results = {}

    if script_name == "context_retention":
        results["context_retention"] = compute_context_retention(log)

    elif script_name == "response_time":
        results["response_time"] = compute_response_time(log)

    elif script_name == "helpful_response":
        # Step 1: Get Prometheus 2 scores for judge turns
        print("  Running Prometheus 2 judge...")
        judge_result = judge_helpful_response_turns(log, use_mock=use_prometheus_mock)
        prometheus_scores = judge_result["prometheus_scores"]
        # Step 2: Compute the full helpful response ratio
        results["helpful_response"] = compute_helpful_response(log, prometheus_scores=prometheus_scores)
        results["_prometheus_details"] = judge_result

    elif script_name == "lexical_diversity":
        results["lexical_diversity"] = compute_lexical_diversity(log)

    elif script_name == "error_rate":
        results["error_rate"] = compute_error_rate(log)

    # Always compute RT and MATTR on every script (cross-validation)
    if script_name not in ("response_time",):
        rt = compute_response_time(log)
        results[f"response_time_xval_{script_name}"] = rt

    if script_name not in ("lexical_diversity",):
        ld = compute_lexical_diversity(log)
        results[f"lexical_diversity_xval_{script_name}"] = ld

    return results


def _save_metric_results(chatbot_name: str, all_metric_results: dict):
    path = os.path.join(RESULTS_DIR, f"{chatbot_name}_metric_results.json")
    # Convert to serialisable form (remove large per_turn lists if needed)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(all_metric_results, f, indent=2, default=str)
    print(f"  [Main] Metric results saved to: {path}")


# ---------------------------------------------------------------------------
# Mode: pilot
# ---------------------------------------------------------------------------
def mode_pilot(mock: bool):
    print("\n=== PILOT RUN (3 turns, context_retention) ===")
    for chatbot_name in CHATBOT_NAMES:
        adapter = make_adapter(chatbot_name, mock=mock)
        orch = Orchestrator(adapter=adapter, chatbot_name=chatbot_name, inter_turn_delay=0)
        log = orch.run_pilot("context_retention", num_turns=3)
        print(f"  {chatbot_name}: {len(log)} turns logged. Pilot OK.")
    print("\nPilot complete. Check results/logs/ for output files.")


# ---------------------------------------------------------------------------
# Mode: calibrate (MATTR pilot calibration)
# ---------------------------------------------------------------------------
def mode_calibrate():
    print("\n=== MATTR-50 PILOT CALIBRATION ===")
    print("You will enter 3 known-good and 3 known-padded responses.")
    print("Type or paste each response, then press Enter twice to finish.\n")

    def read_multiline(prompt: str) -> str:
        print(prompt)
        lines = []
        while True:
            line = input()
            if line == "" and lines:
                break
            lines.append(line)
        return " ".join(lines)

    good_responses = []
    for i in range(3):
        r = read_multiline(f"Good response {i+1} (press Enter twice when done):")
        good_responses.append(r)

    padded_responses = []
    for i in range(3):
        r = read_multiline(f"Padded response {i+1} (press Enter twice when done):")
        padded_responses.append(r)

    thresholds = pilot_calibrate(good_responses, padded_responses)
    print(f"\nCalibrated thresholds:")
    print(f"  Good ≥ {thresholds['good_min']}")
    print(f"  Moderate ≥ {thresholds['moderate_min']}")
    print(f"  Poor < {thresholds['moderate_min']}")
    print(f"  Saved to: results/mattr_thresholds.json")


# ---------------------------------------------------------------------------
# Mode: single script
# ---------------------------------------------------------------------------
def mode_script(chatbot_name: str, script_name: str, mock: bool):
    print(f"\n=== SCRIPT RUN: {chatbot_name.upper()} / {script_name} ===")
    adapter = make_adapter(chatbot_name, mock=mock)
    orch = Orchestrator(adapter=adapter, chatbot_name=chatbot_name, inter_turn_delay=0.3)
    log = orch.run_script(script_name)
    use_prom_mock = os.environ.get("PROMETHEUS_MOCK", "0") == "1" or mock
    metric_results = run_metrics_on_log(script_name, log, use_prometheus_mock=use_prom_mock)

    print(f"\nMetric results for {chatbot_name} / {script_name}:")
    for k, v in metric_results.items():
        if not k.startswith("_"):
            score = v.get("normalized_score")
            clf = v.get("classification", "N/A")
            print(f"  {k}: {score:.1f} / 100 ({clf})" if score is not None else f"  {k}: N/A")


# ---------------------------------------------------------------------------
# Mode: full run (all scripts, both chatbots)
# ---------------------------------------------------------------------------
def mode_full(mock: bool):
    print("\n=== FULL RUN: ALL SCRIPTS, BOTH CHATBOTS ===")
    use_prom_mock = os.environ.get("PROMETHEUS_MOCK", "0") == "1" or mock

    all_results = {}

    for chatbot_name in CHATBOT_NAMES:
        print(f"\n{'#'*60}")
        print(f"# CHATBOT: {chatbot_name.upper()}")
        print(f"{'#'*60}")

        adapter = make_adapter(chatbot_name, mock=mock)
        orch = Orchestrator(adapter=adapter, chatbot_name=chatbot_name, inter_turn_delay=0.3)

        chatbot_metric_results: dict = {}

        for script_name in ALL_SCRIPTS:
            log = orch.run_script(script_name)
            script_metrics = run_metrics_on_log(script_name, log, use_prometheus_mock=use_prom_mock)
            chatbot_metric_results.update(script_metrics)

        # Collect the 5 primary metrics for CUS
        primary_metrics = {
            k: chatbot_metric_results[k]
            for k in ["context_retention", "response_time", "helpful_response",
                      "lexical_diversity", "error_rate"]
            if k in chatbot_metric_results
        }

        cus_result = compute_cus(primary_metrics)
        print(f"\n  CUS (equal-weighted): {cus_result['cus_equal']} / 100 ({cus_result['classification']})")

        # Generate report
        report = generate_report(
            chatbot_name=chatbot_name,
            metric_results=primary_metrics,
            cus_result=cus_result,
        )
        print(report[:800])  # Preview first 800 chars

        _save_metric_results(chatbot_name, chatbot_metric_results)
        all_results[chatbot_name] = {
            "metric_results": primary_metrics,
            "cus": cus_result,
        }

    # Compare both chatbots
    if "rasa" in all_results and "ollama" in all_results:
        comparison = compute_cus_comparison(
            all_results["rasa"]["metric_results"],
            all_results["ollama"]["metric_results"],
        )
        print(f"\n{'='*60}")
        print(f"  COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"  Rasa CUS:   {comparison['rasa']['cus_equal']} ({comparison['rasa']['classification']})")
        print(f"  Ollama CUS: {comparison['ollama']['cus_equal']} ({comparison['ollama']['classification']})")
        print(f"  Winner:     {comparison['winner'].upper()}  (delta = {comparison['delta']})")

    print("\nFull run complete. Check results/ for all outputs.")


# ---------------------------------------------------------------------------
# Mode: SUS study
# ---------------------------------------------------------------------------
def mode_sus(chatbot_name: str):
    print(f"\n=== SUS STUDY COLLECTION: {chatbot_name.upper()} ===")
    print("How many participants will you record now?")
    n = int(input("Number of participants: ").strip())

    records = []
    for i in range(1, n + 1):
        pid = f"P{i:02d}"
        record = collect_sus_interactive(pid, chatbot_name)
        records.append(record)
        print(f"  Recorded P{i:02d}: SUS = {record['sus_score']}")

    save_sus_records(records, chatbot_name)
    agg = aggregate_sus(records)
    print(f"\nSession aggregate: n={agg['n']}  mean={agg['mean_sus']}  std={agg['std_sus']}  [{agg['classification']}]")

    # Optionally collect Kappa human scores
    kappa_q = input("\nAlso record Kappa human scores for Prometheus validation? (y/n): ").strip().lower()
    if kappa_q == "y":
        human_kappa = collect_human_scores_interactive()
        kappa_path = os.path.join(RESULTS_DIR, f"human_kappa_scores_{chatbot_name}.json")
        with open(kappa_path, "w", encoding="utf-8") as f:
            json.dump({str(k): v for k, v in human_kappa.items()}, f, indent=2)
        print(f"  Kappa scores saved to: {kappa_path}")


# ---------------------------------------------------------------------------
# Mode: analysis
# ---------------------------------------------------------------------------
def mode_analysis():
    print("\n=== CORRELATION + REGRESSION ANALYSIS ===")

    # Load metric results
    combined_cus = []
    combined_sus = []
    combined_metrics = []

    for chatbot_name in CHATBOT_NAMES:
        metric_path = os.path.join(RESULTS_DIR, f"{chatbot_name}_metric_results.json")
        if not os.path.exists(metric_path):
            print(f"  [Warning] No metric results for {chatbot_name}. Run --mode full first.")
            continue

        with open(metric_path, "r", encoding="utf-8") as f:
            metric_data = json.load(f)

        primary = {
            k: metric_data[k]
            for k in ["context_retention", "response_time", "helpful_response",
                      "lexical_diversity", "error_rate"]
            if k in metric_data
        }
        cus_result = compute_cus(primary)
        cus_score = cus_result["cus_equal"]

        # Load SUS records
        sus_records = load_all_sus_for_chatbot(chatbot_name)
        if not sus_records:
            print(f"  [Warning] No SUS records for {chatbot_name}. Run --mode sus first.")
            continue

        sus_scores = [r["sus_score"] for r in sus_records]

        # Each participant gets the same CUS (one chatbot run)
        n = len(sus_scores)
        cus_list = [cus_score] * n
        combined_cus.extend(cus_list)
        combined_sus.extend(sus_scores)

        # Build metric score matrix
        metric_row = [
            primary.get("context_retention", {}).get("normalized_score", 0),
            primary.get("response_time", {}).get("normalized_score", 0),
            primary.get("helpful_response", {}).get("normalized_score", 0),
            primary.get("lexical_diversity", {}).get("normalized_score", 0),
            primary.get("error_rate", {}).get("normalized_score", 0),
        ]
        combined_metrics.extend([metric_row] * n)

    if len(combined_sus) < 3:
        print("  Not enough data for analysis. Need SUS data from at least 3 participants.")
        return

    run_full_analysis(
        cus_scores=combined_cus,
        sus_scores=combined_sus,
        metric_scores_matrix=combined_metrics,
        chatbot_name="combined",
    )


# ---------------------------------------------------------------------------
# Argument parser + main
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Chatbot Usability Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--mode", choices=["pilot", "calibrate", "script", "full", "sus", "analysis"],
                        default="pilot", help="Run mode")
    parser.add_argument("--chatbot", choices=CHATBOT_NAMES, default="ollama",
                        help="Chatbot to test (for script/sus modes)")
    parser.add_argument("--script", choices=ALL_SCRIPTS, default="context_retention",
                        help="Script to run (for script mode)")
    parser.add_argument("--mock", action="store_true",
                        help="Use mock adapters (no live chatbots required)")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  CHATBOT USABILITY TESTING FRAMEWORK")
    print(f"  Mode: {args.mode.upper()}")
    if args.mock:
        print(f"  *** MOCK MODE — no live chatbots required ***")
    print(f"{'='*60}")

    if args.mode == "pilot":
        mode_pilot(mock=args.mock)
    elif args.mode == "calibrate":
        mode_calibrate()
    elif args.mode == "script":
        mode_script(args.chatbot, args.script, mock=args.mock)
    elif args.mode == "full":
        mode_full(mock=args.mock)
    elif args.mode == "sus":
        mode_sus(args.chatbot)
    elif args.mode == "analysis":
        mode_analysis()


if __name__ == "__main__":
    main()
