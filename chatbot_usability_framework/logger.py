"""
logger.py
---------
Conversation logger for the Chatbot Usability Testing Framework.

Stores every exchange (message, response, timestamp, latency_ms, turn_number,
session_id, chatbot_name, script_name) as the raw dataset that all metric
modules read from.

Output format: JSONL (one JSON object per turn) written to
results/logs/<session_id>.jsonl
"""

import json
import os
import uuid
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_LOGS_DIR = os.path.join(_BASE_DIR, "results", "logs")
os.makedirs(_LOGS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# ConversationLogger
# ---------------------------------------------------------------------------
class ConversationLogger:
    """
    One logger instance per test session.

    Usage
    -----
    logger = ConversationLogger(chatbot_name="rasa", script_name="context_retention")
    logger.log_turn(turn_number=1, user_message="Hello", bot_response="Hi!", latency_ms=120)
    logger.save()
    session_log = logger.get_log()
    """

    def __init__(self, chatbot_name: str, script_name: str, session_id: str = None):
        """
        Parameters
        ----------
        chatbot_name : str
            "rasa" or "ollama"
        script_name : str
            One of: context_retention | response_time | helpful_response |
                    lexical_diversity | error_rate
        session_id : str, optional
            Auto-generated UUID if not provided.
        """
        self.chatbot_name = chatbot_name
        self.script_name = script_name
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.turns: list[dict] = []
        self.started_at = datetime.now(timezone.utc).isoformat()

    # ------------------------------------------------------------------
    def log_turn(
        self,
        turn_number: int,
        user_message: str,
        bot_response: str,
        latency_ms: float,
        metadata: dict = None,
    ) -> dict:
        """
        Record a single conversation turn.

        Parameters
        ----------
        turn_number : int
            1-indexed turn within the script.
        user_message : str
            The scripted user input sent to the bot.
        bot_response : str
            The raw response string returned by the bot.
        latency_ms : float
            Round-trip latency in milliseconds.
        metadata : dict, optional
            Any extra data (e.g. expected_keyword, checkpoint_id).

        Returns
        -------
        dict  The turn record that was appended.
        """
        record = {
            "session_id": self.session_id,
            "chatbot_name": self.chatbot_name,
            "script_name": self.script_name,
            "turn_number": turn_number,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "user_message": user_message,
            "bot_response": bot_response,
            "latency_ms": round(latency_ms, 2),
        }
        if metadata:
            record["metadata"] = metadata
        self.turns.append(record)
        return record

    # ------------------------------------------------------------------
    def save(self) -> str:
        """
        Write all logged turns to a JSONL file and return the file path.
        """
        filename = f"{self.chatbot_name}_{self.script_name}_{self.session_id}.jsonl"
        filepath = os.path.join(_LOGS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            for turn in self.turns:
                f.write(json.dumps(turn, ensure_ascii=False) + "\n")
        return filepath

    # ------------------------------------------------------------------
    def get_log(self) -> list[dict]:
        """Return the in-memory list of turn records."""
        return self.turns

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.turns)

    def __repr__(self):
        return (
            f"ConversationLogger(chatbot={self.chatbot_name!r}, "
            f"script={self.script_name!r}, turns={len(self.turns)})"
        )


# ---------------------------------------------------------------------------
# Utility: load a saved JSONL log
# ---------------------------------------------------------------------------
def load_log(filepath: str) -> list[dict]:
    """
    Load a previously saved JSONL conversation log.

    Parameters
    ----------
    filepath : str
        Path to the .jsonl file.

    Returns
    -------
    list[dict]  Ordered list of turn records.
    """
    turns = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                turns.append(json.loads(line))
    return turns


def load_latest_log(chatbot_name: str, script_name: str) -> list[dict]:
    """
    Load the most recently modified log matching chatbot_name and script_name.
    """
    prefix = f"{chatbot_name}_{script_name}_"
    candidates = [
        f for f in os.listdir(_LOGS_DIR)
        if f.startswith(prefix) and f.endswith(".jsonl")
    ]
    if not candidates:
        raise FileNotFoundError(
            f"No log found for chatbot={chatbot_name!r}, script={script_name!r} "
            f"in {_LOGS_DIR}"
        )
    candidates.sort(
        key=lambda f: os.path.getmtime(os.path.join(_LOGS_DIR, f)),
        reverse=True,
    )
    return load_log(os.path.join(_LOGS_DIR, candidates[0]))


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    lg = ConversationLogger(chatbot_name="rasa", script_name="context_retention")
    lg.log_turn(1, "Hi, I'm vegetarian.", "Hello! I'll keep that in mind.", 142.3)
    lg.log_turn(2, "Suggest a pasta dish.", "How about pasta primavera?", 210.7)
    path = lg.save()
    print(f"Saved to: {path}")
    loaded = load_log(path)
    print(f"Loaded {len(loaded)} turns.")
    print(loaded[0])
