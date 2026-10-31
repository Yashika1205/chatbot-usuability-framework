"""
orchestrator.py
---------------
Test Orchestrator for the Chatbot Usability Testing Framework.

The orchestrator drives the full automated test pipeline:
  1. Iterates over every scripted turn in a question bank script.
  2. Sends each user message to a chatbot via its adapter.
  3. Records the response + latency via the ConversationLogger.
  4. Saves the log to disk after each script completes.
  5. Returns the log for downstream metric computation.

Usage
-----
from orchestrator import Orchestrator
from adapters import OllamaAdapter

adapter = OllamaAdapter(use_mock=True)
orch = Orchestrator(adapter=adapter, chatbot_name="ollama")
log = orch.run_script("context_retention")
"""

import time
from typing import Any

from logger import ConversationLogger
from task_bank.task_bank import get_script, SCRIPTS


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
class Orchestrator:
    """
    Drives a single chatbot through one or all scripted test banks.

    Parameters
    ----------
    adapter : RasaAdapter | OllamaAdapter
        A chatbot adapter with a .send(message, session_id) method.
    chatbot_name : str
        "rasa" or "ollama" — used for log filenames.
    inter_turn_delay : float
        Seconds to wait between turns (default 0.5 s). Prevents rate-limiting.
    verbose : bool
        Print turn-by-turn output to stdout.
    """

    def __init__(
        self,
        adapter,
        chatbot_name: str,
        inter_turn_delay: float = 0.5,
        verbose: bool = True,
    ):
        self.adapter = adapter
        self.chatbot_name = chatbot_name
        self.inter_turn_delay = inter_turn_delay
        self.verbose = verbose

    # ------------------------------------------------------------------
    def run_script(self, script_name: str) -> list[dict]:
        """
        Run one complete scripted test bank against the chatbot.

        Parameters
        ----------
        script_name : str
            One of: context_retention | response_time | helpful_response |
                    lexical_diversity | error_rate

        Returns
        -------
        list[dict]
            The full conversation log (list of turn records).
        """
        script_turns = get_script(script_name)
        logger = ConversationLogger(
            chatbot_name=self.chatbot_name,
            script_name=script_name,
        )
        session_id = logger.session_id

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"SCRIPT: {script_name.upper()}")
            print(f"CHATBOT: {self.chatbot_name}  |  SESSION: {session_id}")
            print(f"TOTAL TURNS: {len(script_turns)}")
            print(f"{'='*60}")

        # Reset adapter context between scripts (Ollama only)
        if hasattr(self.adapter, "reset"):
            self.adapter.reset()

        for turn_data in script_turns:
            turn_num = turn_data["turn_number"]
            user_msg = turn_data["user_message"]
            metadata = turn_data.get("metadata", {})
            expected = turn_data.get("expected")
            if expected:
                metadata = {**metadata, "expected": expected}

            if self.verbose:
                print(f"\n  Turn {turn_num:>2}: {user_msg[:80]}{'...' if len(user_msg)>80 else ''}")

            try:
                result = self.adapter.send(
                    message=user_msg,
                    session_id=session_id,
                )
                bot_response = result["response"]
                latency_ms = result["latency_ms"]
            except (ConnectionError, TimeoutError) as e:
                # Log the failure turn and continue
                bot_response = f"[ERROR] {e}"
                latency_ms = -1.0
                if self.verbose:
                    print(f"  !! Adapter error on turn {turn_num}: {e}")

            logger.log_turn(
                turn_number=turn_num,
                user_message=user_msg,
                bot_response=bot_response,
                latency_ms=latency_ms,
                metadata=metadata,
            )

            if self.verbose:
                preview = bot_response[:100].replace("\n", " ")
                print(f"    Bot: {preview}{'...' if len(bot_response)>100 else ''}")
                print(f"    Latency: {latency_ms:.1f} ms")

            if self.inter_turn_delay > 0:
                time.sleep(self.inter_turn_delay)

        filepath = logger.save()
        if self.verbose:
            print(f"\n  Log saved to: {filepath}")
            print(f"{'='*60}\n")

        return logger.get_log()

    # ------------------------------------------------------------------
    def run_all_scripts(self) -> dict[str, list[dict]]:
        """
        Run all five scripted test banks in order.

        Returns
        -------
        dict  {script_name: log_turns}
        """
        all_logs = {}
        for script_name in SCRIPTS.keys():
            log = self.run_script(script_name)
            all_logs[script_name] = log
        return all_logs

    # ------------------------------------------------------------------
    def run_pilot(self, script_name: str = "context_retention", num_turns: int = 5) -> list[dict]:
        """
        Run only the first N turns of a script — used for end-to-end pipeline
        validation before the full test run.

        Parameters
        ----------
        script_name : str
        num_turns : int
            Number of turns to execute (default: 5).

        Returns
        -------
        list[dict]  The partial conversation log.
        """
        all_turns = get_script(script_name)
        pilot_turns = all_turns[:num_turns]

        # Temporarily override the script with the truncated version
        logger = ConversationLogger(
            chatbot_name=self.chatbot_name,
            script_name=f"{script_name}_pilot",
        )
        session_id = logger.session_id

        if self.verbose:
            print(f"\n[PILOT] {script_name} — first {num_turns} turns")
            print(f"Chatbot: {self.chatbot_name}  Session: {session_id}")

        if hasattr(self.adapter, "reset"):
            self.adapter.reset()

        for turn_data in pilot_turns:
            user_msg = turn_data["user_message"]
            metadata = turn_data.get("metadata", {})
            expected = turn_data.get("expected")
            if expected:
                metadata = {**metadata, "expected": expected}

            try:
                result = self.adapter.send(message=user_msg, session_id=session_id)
                bot_response = result["response"]
                latency_ms = result["latency_ms"]
            except (ConnectionError, TimeoutError) as e:
                bot_response = f"[ERROR] {e}"
                latency_ms = -1.0

            logger.log_turn(
                turn_number=turn_data["turn_number"],
                user_message=user_msg,
                bot_response=bot_response,
                latency_ms=latency_ms,
                metadata=metadata,
            )

            if self.verbose:
                print(f"  Turn {turn_data['turn_number']}: {bot_response[:80]}  [{latency_ms:.1f} ms]")

            if self.inter_turn_delay > 0:
                time.sleep(self.inter_turn_delay)

        filepath = logger.save()
        if self.verbose:
            print(f"  [PILOT] Saved to: {filepath}")

        return logger.get_log()

    def __repr__(self):
        return f"Orchestrator(chatbot={self.chatbot_name!r}, adapter={self.adapter!r})"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from adapters.rasa_adapter import RasaAdapter
    from adapters.ollama_adapter import OllamaAdapter

    print("=== Rasa (mock) pilot ===")
    rasa = RasaAdapter(use_mock=True)
    orch_rasa = Orchestrator(adapter=rasa, chatbot_name="rasa", inter_turn_delay=0)
    log = orch_rasa.run_pilot("context_retention", num_turns=3)
    print(f"Pilot returned {len(log)} turns.\n")

    print("=== Ollama (mock) pilot ===")
    ollama = OllamaAdapter(use_mock=True)
    orch_ollama = Orchestrator(adapter=ollama, chatbot_name="ollama", inter_turn_delay=0)
    log2 = orch_ollama.run_pilot("error_rate", num_turns=3)
    print(f"Pilot returned {len(log2)} turns.")
