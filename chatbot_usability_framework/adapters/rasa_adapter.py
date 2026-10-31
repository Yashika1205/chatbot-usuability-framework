"""
adapters/rasa_adapter.py
------------------------
Adapter for a locally running Rasa server.

Rasa REST endpoint (default): POST http://localhost:5005/webhooks/rest/webhook
Payload: {"sender": "<session_id>", "message": "<text>"}
Response: [{"recipient_id": ..., "text": "..."}]

If Rasa is not running, send() raises ConnectionError.
A mock mode is available for offline testing: set use_mock=True or
set the environment variable RASA_MOCK=1.
"""

import os
import time
import requests
from typing import Optional


class RasaAdapter:
    """
    Sends a user message to a running Rasa server and returns the bot response
    along with the round-trip latency in milliseconds.

    Parameters
    ----------
    base_url : str
        Base URL of the Rasa REST server (default: http://localhost:5005).
    use_mock : bool
        If True, skip actual HTTP call and return a canned mock response.
        Useful for pipeline development when Rasa is not running.
    timeout : int
        HTTP request timeout in seconds (default: 30).
    """

    DEFAULT_URL = "http://localhost:5005"

    def __init__(
        self,
        base_url: str = None,
        use_mock: bool = False,
        timeout: int = 30,
    ):
        env_mock = os.environ.get("RASA_MOCK", "0").strip() == "1"
        self.use_mock = use_mock or env_mock
        self.base_url = (base_url or os.environ.get("RASA_URL", self.DEFAULT_URL)).rstrip("/")
        self.timeout = timeout
        self.endpoint = f"{self.base_url}/webhooks/rest/webhook"

    # ------------------------------------------------------------------
    def send(self, message: str, session_id: str = "test_user") -> dict:
        """
        Send a message to Rasa and return the response.

        Returns
        -------
        dict with keys:
            response   : str   — bot response text (joined if multiple)
            latency_ms : float — round-trip time in milliseconds
            raw        : list  — raw Rasa response JSON
        """
        if self.use_mock:
            return self._mock_response(message)

        payload = {"sender": session_id, "message": message}
        t_start = time.perf_counter()
        try:
            resp = requests.post(
                self.endpoint,
                json=payload,
                timeout=self.timeout,
            )
            resp.raise_for_status()
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"Rasa server not reachable at {self.endpoint}. "
                "Start Rasa with: rasa run --enable-api\n"
                f"Original error: {e}"
            )
        except requests.exceptions.Timeout:
            raise TimeoutError(
                f"Rasa server timed out after {self.timeout}s at {self.endpoint}"
            )
        latency_ms = (time.perf_counter() - t_start) * 1000

        raw = resp.json()
        # Rasa returns a list of message objects; join all text fields
        texts = [msg.get("text", "") for msg in raw if "text" in msg]
        response_text = " ".join(texts).strip() if texts else "[no response]"

        return {
            "response": response_text,
            "latency_ms": round(latency_ms, 2),
            "raw": raw,
        }

    # ------------------------------------------------------------------
    def health_check(self) -> bool:
        """Return True if the Rasa server is reachable."""
        if self.use_mock:
            return True
        try:
            resp = requests.get(f"{self.base_url}/", timeout=5)
            return resp.status_code == 200
        except Exception:
            return False

    # ------------------------------------------------------------------
    def _mock_response(self, message: str) -> dict:
        """
        Return a deterministic mock response for offline testing.
        Simulates ~150 ms latency.
        """
        time.sleep(0.15)
        return {
            "response": f"[RASA MOCK] Received: {message[:60]}",
            "latency_ms": 150.0,
            "raw": [{"recipient_id": "test_user", "text": f"[RASA MOCK] Received: {message[:60]}"}],
        }

    def __repr__(self):
        mode = "MOCK" if self.use_mock else f"live@{self.base_url}"
        return f"RasaAdapter({mode})"


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    adapter = RasaAdapter(use_mock=True)
    result = adapter.send("Hi, I am vegetarian.")
    print(result)
