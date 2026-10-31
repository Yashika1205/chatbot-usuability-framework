"""
adapters/ollama_adapter.py
--------------------------
Adapter for a locally running Ollama server (LLM-backed chatbot).

Ollama REST endpoint: POST http://localhost:11434/api/chat
Model: llama3.1:8b  (or mistral:7b — set via OLLAMA_MODEL env var)

If Ollama is not running, send() raises ConnectionError.
A mock mode is available for offline testing: set use_mock=True or
set the environment variable OLLAMA_MOCK=1.

The adapter maintains a per-session message history to support multi-turn
conversations (context window). Clear it between test scripts with reset().
"""

import os
import time
import json
import requests
from typing import Optional


class OllamaAdapter:
    """
    Sends a user message to a running Ollama server (multi-turn aware)
    and returns the bot response with round-trip latency.

    Parameters
    ----------
    model : str
        Ollama model tag (default: llama3.1:8b or OLLAMA_MODEL env var).
    base_url : str
        Base URL of the Ollama server (default: http://localhost:11434).
    system_prompt : str
        Optional system instruction prepended to every conversation.
    use_mock : bool
        Skip actual HTTP call and return a canned mock response.
    timeout : int
        HTTP request timeout in seconds (default: 60 — LLMs can be slow).
    """

    DEFAULT_URL = "http://localhost:11434"
    DEFAULT_MODEL = "llama3.1:8b"

    def __init__(
        self,
        model: str = None,
        base_url: str = None,
        system_prompt: str = None,
        use_mock: bool = False,
        timeout: int = 60,
    ):
        env_mock = os.environ.get("OLLAMA_MOCK", "0").strip() == "1"
        self.use_mock = use_mock or env_mock
        self.base_url = (base_url or os.environ.get("OLLAMA_URL", self.DEFAULT_URL)).rstrip("/")
        self.model = model or os.environ.get("OLLAMA_MODEL", self.DEFAULT_MODEL)
        self.timeout = timeout
        self.endpoint = f"{self.base_url}/api/chat"

        # Conversation history — maintains multi-turn context
        self._messages: list[dict] = []
        if system_prompt:
            self._messages.append({"role": "system", "content": system_prompt})

    # ------------------------------------------------------------------
    def send(self, message: str, session_id: str = "test_user") -> dict:
        """
        Send a message to Ollama and return the response.

        Returns
        -------
        dict with keys:
            response   : str   — bot response text
            latency_ms : float — round-trip time in milliseconds
            raw        : dict  — raw Ollama response JSON
        """
        if self.use_mock:
            return self._mock_response(message)

        self._messages.append({"role": "user", "content": message})

        payload = {
            "model": self.model,
            "messages": self._messages,
            "stream": False,
        }

        t_start = time.perf_counter()
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            # Remove the user message we just appended (failed call)
            self._messages.pop()
            raise ConnectionError(
                f"Ollama server not reachable at {self.endpoint}. "
                "Start Ollama with: ollama serve\n"
                f"Original error: {e}"
            )
        except requests.exceptions.Timeout:
            self._messages.pop()
            raise TimeoutError(
                f"Ollama server timed out after {self.timeout}s for model {self.model!r}"
            )
        latency_ms = (time.perf_counter() - t_start) * 1000

        raw = resp.json()
        response_text = raw.get("message", {}).get("content", "[no response]").strip()

        # Append assistant turn to history for context continuity
        self._messages.append({"role": "assistant", "content": response_text})

        return {
            "response": response_text,
            "latency_ms": round(latency_ms, 2),
            "raw": raw,
        }

    # ------------------------------------------------------------------
    def reset(self, keep_system_prompt: bool = True):
        """
        Clear the conversation history.
        Call this between test scripts to avoid cross-script context bleed.
        """
        if keep_system_prompt and self._messages and self._messages[0]["role"] == "system":
            self._messages = [self._messages[0]]
        else:
            self._messages = []

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the Ollama server is reachable."""
        if self.use_mock:
            return True
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _mock_response(self, message: str) -> dict:
        """Return a deterministic mock response for offline testing."""
        time.sleep(0.2)
        self._messages.append({"role": "user", "content": message})
        text = f"[OLLAMA MOCK] I received your message about: {message[:60]}"
        self._messages.append({"role": "assistant", "content": text})
        return {
            "response": text,
            "latency_ms": 200.0,
            "raw": {"model": self.model, "message": {"role": "assistant", "content": text}},
        }

    def __repr__(self):
        mode = "MOCK" if self.use_mock else f"live@{self.base_url}"
        return f"OllamaAdapter(model={self.model!r}, {mode})"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    adapter = OllamaAdapter(use_mock=True)
    r1 = adapter.send("Hi, I am vegetarian.")
    print(r1)
    r2 = adapter.send("What should I eat for dinner?")
    print(r2)
    adapter.reset()
    print(f"History after reset: {len(adapter._messages)} messages")
