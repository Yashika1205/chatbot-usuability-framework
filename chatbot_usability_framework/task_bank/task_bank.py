"""
task_bank/task_bank.py
----------------------
All five scripted question banks as defined in the framework document.

Each script is a list of turn dicts:
    turn_number   : int
    user_message  : str
    expected       : str or None   (keyword hint for rule-based scoring)
    metadata       : dict          (checkpoint flag, error type tag, etc.)

Scripts
-------
1. context_retention   — 30 turns, vegetarian → Italy travel
2. response_time       — 15 turns, smart home setup
3. helpful_response    — 15 turns, job hunting assistant
4. lexical_diversity   — 12 turns, personal finance Q&A
5. error_rate          — 15 turns, noisy customer support
"""

from __future__ import annotations
from typing import Any


# ---------------------------------------------------------------------------
# Script 1 — Context Retention (30 turns)
# Checkpoints at turns 5, 10, 15, 20, 25, 30
# Fact planted at turn 1: "I'm vegetarian"
# ---------------------------------------------------------------------------
CONTEXT_RETENTION_SCRIPT: list[dict[str, Any]] = [
    # Turn 1 — plant the fact
    {"turn_number": 1,  "user_message": "Hi! Just so you know, I'm vegetarian. I want to explore food and travel.", "expected": None, "metadata": {"fact_planted": "vegetarian"}},
    {"turn_number": 2,  "user_message": "What's your favourite season?", "expected": None, "metadata": {}},
    {"turn_number": 3,  "user_message": "Tell me about the Mediterranean diet.", "expected": None, "metadata": {}},
    {"turn_number": 4,  "user_message": "Is olive oil good for health?", "expected": None, "metadata": {}},
    # Checkpoint 1 — turn 5
    {"turn_number": 5,  "user_message": "Can you suggest a dinner for me tonight?", "expected": "vegetarian", "metadata": {"checkpoint": 1, "checkpoint_turn": 5}},
    {"turn_number": 6,  "user_message": "What Italian dishes are popular?", "expected": None, "metadata": {}},
    {"turn_number": 7,  "user_message": "Do Italians eat a lot of pasta?", "expected": None, "metadata": {}},
    {"turn_number": 8,  "user_message": "What is risotto?", "expected": None, "metadata": {}},
    {"turn_number": 9,  "user_message": "Are there good vegetarian options in Italian cuisine?", "expected": None, "metadata": {}},
    # Checkpoint 2 — turn 10
    {"turn_number": 10, "user_message": "I'm planning a dinner party. What should I cook?", "expected": "vegetarian", "metadata": {"checkpoint": 2, "checkpoint_turn": 10}},
    {"turn_number": 11, "user_message": "Tell me about Rome.", "expected": None, "metadata": {}},
    {"turn_number": 12, "user_message": "What's the best time of year to visit Italy?", "expected": None, "metadata": {}},
    {"turn_number": 13, "user_message": "How expensive is travelling in Italy?", "expected": None, "metadata": {}},
    {"turn_number": 14, "user_message": "What are some less-touristy Italian cities?", "expected": None, "metadata": {}},
    # Checkpoint 3 — turn 15
    {"turn_number": 15, "user_message": "What should I keep in mind about eating out in Italy?", "expected": "vegetarian", "metadata": {"checkpoint": 3, "checkpoint_turn": 15}},
    {"turn_number": 16, "user_message": "Is Italian public transport good?", "expected": None, "metadata": {}},
    {"turn_number": 17, "user_message": "What language do Italians speak?", "expected": None, "metadata": {}},
    {"turn_number": 18, "user_message": "Tell me about the Amalfi Coast.", "expected": None, "metadata": {}},
    {"turn_number": 19, "user_message": "What budget should I plan for a week in Italy?", "expected": None, "metadata": {}},
    # Checkpoint 4 — turn 20
    {"turn_number": 20, "user_message": "What restaurants should I look for on my Italy trip?", "expected": "vegetarian", "metadata": {"checkpoint": 4, "checkpoint_turn": 20}},
    {"turn_number": 21, "user_message": "Do you think Florence or Venice is better to visit first?", "expected": None, "metadata": {}},
    {"turn_number": 22, "user_message": "What is tiramisu made of?", "expected": None, "metadata": {}},
    {"turn_number": 23, "user_message": "Are there good street food options in Italian cities?", "expected": None, "metadata": {}},
    {"turn_number": 24, "user_message": "What is the tipping culture in Italy?", "expected": None, "metadata": {}},
    # Checkpoint 5 — turn 25
    {"turn_number": 25, "user_message": "I need to plan my meals for a week in Italy. Any advice?", "expected": "vegetarian", "metadata": {"checkpoint": 5, "checkpoint_turn": 25}},
    {"turn_number": 26, "user_message": "What is gelato?", "expected": None, "metadata": {}},
    {"turn_number": 27, "user_message": "How do Italians feel about tourists?", "expected": None, "metadata": {}},
    {"turn_number": 28, "user_message": "What should I pack for Italy in the summer?", "expected": None, "metadata": {}},
    {"turn_number": 29, "user_message": "Any safety tips for travelling in Italy?", "expected": None, "metadata": {}},
    # Checkpoint 6 — turn 30
    {"turn_number": 30, "user_message": "Final question: what is the most important thing to remember when eating out on my Italy trip?", "expected": "vegetarian", "metadata": {"checkpoint": 6, "checkpoint_turn": 30}},
]

CHECKPOINT_TURNS = {5, 10, 15, 20, 25, 30}


# ---------------------------------------------------------------------------
# Script 2 — Average Response Time (15 turns, smart home setup)
# Burst tests at turns 6 and 13
# ---------------------------------------------------------------------------
RESPONSE_TIME_SCRIPT: list[dict[str, Any]] = [
    # Short factual (turns 1–5)
    {"turn_number": 1,  "user_message": "What is a smart home?",                                                             "expected": None, "metadata": {"turn_type": "short_factual"}},
    {"turn_number": 2,  "user_message": "What devices can I connect?",                                                        "expected": None, "metadata": {"turn_type": "short_factual"}},
    {"turn_number": 3,  "user_message": "Do I need a hub?",                                                                   "expected": None, "metadata": {"turn_type": "short_factual"}},
    {"turn_number": 4,  "user_message": "What is Zigbee?",                                                                    "expected": None, "metadata": {"turn_type": "short_factual"}},
    {"turn_number": 5,  "user_message": "Is Z-Wave better than Zigbee?",                                                      "expected": None, "metadata": {"turn_type": "short_factual"}},
    # Burst test 1 — turn 6 (rapid-fire)
    {"turn_number": 6,  "user_message": "Give me the top 5 smart home devices, top 3 protocols, and top 2 assistant integrations — all in one reply.",
                        "expected": None, "metadata": {"turn_type": "burst_test", "burst_id": 1}},
    # Long multi-clause (turns 7–9)
    {"turn_number": 7,  "user_message": "Walk me through the complete setup process for a smart home, from buying the first device all the way to automating routines, step by step.", "expected": None, "metadata": {"turn_type": "long_multi_clause"}},
    {"turn_number": 8,  "user_message": "Compare Google Home, Amazon Alexa, and Apple HomeKit in terms of privacy, device compatibility, and ease of use.",                             "expected": None, "metadata": {"turn_type": "long_multi_clause"}},
    {"turn_number": 9,  "user_message": "Explain in detail how a smart thermostat saves energy and whether the payback period justifies the upfront cost.",                           "expected": None, "metadata": {"turn_type": "long_multi_clause"}},
    # Reasoning (turns 10–12)
    {"turn_number": 10, "user_message": "If I already have Philips Hue lights and a Google Nest Hub, what is the most logical next device to buy and why?",                           "expected": None, "metadata": {"turn_type": "reasoning"}},
    {"turn_number": 11, "user_message": "My internet cuts out sometimes. How will that affect my smart home devices and how can I mitigate it?",                                       "expected": None, "metadata": {"turn_type": "reasoning"}},
    {"turn_number": 12, "user_message": "Retrieval question: what are the known privacy vulnerabilities in consumer smart home devices published after 2022?",                         "expected": None, "metadata": {"turn_type": "retrieval_dependent"}},
    # Burst test 2 — turn 13
    {"turn_number": 13, "user_message": "List all the automations I can set up with Google Home, Philips Hue, Nest, Ring, and TP-Link Kasa — one bullet per device.",
                        "expected": None, "metadata": {"turn_type": "burst_test", "burst_id": 2}},
    # Wrap-up (turns 14–15)
    {"turn_number": 14, "user_message": "Summarise the three biggest mistakes people make when setting up their first smart home.",                                                    "expected": None, "metadata": {"turn_type": "short_factual"}},
    {"turn_number": 15, "user_message": "Thank you! What should I do first this weekend?",                                                                                            "expected": None, "metadata": {"turn_type": "short_factual"}},
]


# ---------------------------------------------------------------------------
# Script 3 — Helpful Response Ratio (15 turns, job hunting assistant)
# Scoring: 8 rule-based turns + 5 Prometheus 2 turns + 5 human spot-check turns
# Turns scored by Prometheus 2: 2, 3, 7, 10, 12
# Turns scored by human (overlap, for Kappa): 2, 5, 7, 10, 14
# ---------------------------------------------------------------------------
HELPFUL_RESPONSE_SCRIPT: list[dict[str, Any]] = [
    {"turn_number": 1,  "user_message": "I'm looking for a software engineering job. Can you help?",
                        "expected": "yes|sure|help|assist",
                        "metadata": {"scoring": "rule", "expected_elements": ["acknowledgement", "offer_to_help"]}},
    {"turn_number": 2,  "user_message": "I have 2 years of experience in Python and have worked at a startup. What roles should I target?",
                        "expected": None,
                        "metadata": {"scoring": "prometheus", "human_spot_check": True}},
    {"turn_number": 3,  "user_message": "Can you write me a tailored resume summary for a mid-level Python developer role?",
                        "expected": None,
                        "metadata": {"scoring": "prometheus"}},
    {"turn_number": 4,  "user_message": "What are the top 5 skills I need to highlight for a backend role?",
                        "expected": "python|api|database|sql|rest|microservice|aws|docker",
                        "metadata": {"scoring": "rule", "expected_elements": ["list_of_skills"]}},
    {"turn_number": 5,  "user_message": "How should I prepare for a technical interview?",
                        "expected": "algorithm|leetcode|data structure|system design|practice|mock",
                        "metadata": {"scoring": "rule", "human_spot_check": True}},
    {"turn_number": 6,  "user_message": "Write a cover letter opening paragraph for a Python developer applying to a fintech company.",
                        "expected": "python|fintech|experience",
                        "metadata": {"scoring": "rule", "expected_elements": ["opening_paragraph"]}},
    {"turn_number": 7,  "user_message": "What questions should I ask the interviewer at the end?",
                        "expected": None,
                        "metadata": {"scoring": "prometheus", "human_spot_check": True}},
    {"turn_number": 8,  "user_message": "How do I negotiate salary for a mid-level Python developer role?",
                        "expected": "research|market rate|negotiate|counter|offer|salary",
                        "metadata": {"scoring": "rule", "expected_elements": ["negotiation_strategy"]}},
    {"turn_number": 9,  "user_message": "What LinkedIn profile sections matter most to recruiters?",
                        "expected": "headline|summary|experience|skills|recommendations|linkedin",
                        "metadata": {"scoring": "rule"}},
    {"turn_number": 10, "user_message": "I got rejected after a final round. What should I do?",
                        "expected": None,
                        "metadata": {"scoring": "prometheus", "human_spot_check": True}},
    {"turn_number": 11, "user_message": "How long should my resume be?",
                        "expected": "one page|two page|1 page|2 page|single page",
                        "metadata": {"scoring": "rule"}},
    {"turn_number": 12, "user_message": "Can you help me prepare for a system design interview?",
                        "expected": None,
                        "metadata": {"scoring": "prometheus"}},
    {"turn_number": 13, "user_message": "What are common mistakes people make on their GitHub profiles?",
                        "expected": "readme|commit|project|contribution|description|pinned",
                        "metadata": {"scoring": "rule"}},
    {"turn_number": 14, "user_message": "I'm feeling discouraged. Is it normal for the job hunt to take months?",
                        "expected": "normal|common|months|patience|persist|average",
                        "metadata": {"scoring": "rule", "human_spot_check": True}},
    {"turn_number": 15, "user_message": "Thank you! Can you summarise the top 3 action items I should do this week?",
                        "expected": "apply|resume|linkedin|interview|practice|network|action",
                        "metadata": {"scoring": "rule", "expected_elements": ["action_items_list"]}},
]

PROMETHEUS_TURNS = {2, 3, 7, 10, 12}
HUMAN_SPOT_CHECK_TURNS = {2, 5, 7, 10, 14}


# ---------------------------------------------------------------------------
# Script 4 — Lexical Diversity / MATTR-50 (12 turns, personal finance)
# Question types: yes/no (2), explanation (3), step-by-step (3),
#                 opinion (2), repeated same-type (2 pairs)
# ---------------------------------------------------------------------------
LEXICAL_DIVERSITY_SCRIPT: list[dict[str, Any]] = [
    {"turn_number": 1,  "user_message": "Should I keep an emergency fund?",
                        "expected": None, "metadata": {"question_type": "yes_no"}},
    {"turn_number": 2,  "user_message": "What is compound interest?",
                        "expected": None, "metadata": {"question_type": "explanation"}},
    {"turn_number": 3,  "user_message": "How do I create a monthly budget from scratch?",
                        "expected": None, "metadata": {"question_type": "step_by_step", "pair_id": "A"}},
    {"turn_number": 4,  "user_message": "Is it better to pay off debt or invest?",
                        "expected": None, "metadata": {"question_type": "yes_no"}},
    {"turn_number": 5,  "user_message": "Explain what an index fund is.",
                        "expected": None, "metadata": {"question_type": "explanation"}},
    {"turn_number": 6,  "user_message": "What is the difference between a Roth IRA and a traditional IRA?",
                        "expected": None, "metadata": {"question_type": "explanation"}},
    {"turn_number": 7,  "user_message": "What is your opinion on cryptocurrency as an investment?",
                        "expected": None, "metadata": {"question_type": "opinion", "pair_id": "B"}},
    {"turn_number": 8,  "user_message": "How do I build an investment portfolio as a beginner?",
                        "expected": None, "metadata": {"question_type": "step_by_step", "pair_id": "A"}},
    {"turn_number": 9,  "user_message": "Do you think real estate is a better investment than stocks?",
                        "expected": None, "metadata": {"question_type": "opinion", "pair_id": "B"}},
    {"turn_number": 10, "user_message": "How do I track my expenses effectively?",
                        "expected": None, "metadata": {"question_type": "step_by_step"}},
    {"turn_number": 11, "user_message": "Explain how inflation affects savings.",
                        "expected": None, "metadata": {"question_type": "explanation"}},
    {"turn_number": 12, "user_message": "What are the biggest personal finance mistakes young people make?",
                        "expected": None, "metadata": {"question_type": "step_by_step"}},
]


# ---------------------------------------------------------------------------
# Script 5 — Error Rate (15 turns, noisy customer support)
# Error taxonomy from Higashinaka et al. (2015):
#   unclear_intention | not_understandable | ignored | not_relevant | assumed_incorrectly
# ---------------------------------------------------------------------------
ERROR_RATE_SCRIPT: list[dict[str, Any]] = [
    # Typo-heavy (4 turns)
    {"turn_number": 1,  "user_message": "I wnat to retrun my oreder from lst week",
                        "expected": "return|order|refund|process",
                        "metadata": {"error_type": "typo_heavy", "higashinaka": "not_understandable"}},
    {"turn_number": 2,  "user_message": "Wen wil my pacckage arive???",
                        "expected": "delivery|arrive|shipment|track",
                        "metadata": {"error_type": "typo_heavy", "higashinaka": "not_understandable"}},
    {"turn_number": 3,  "user_message": "i ned to cancl my subscrpition pleas",
                        "expected": "cancel|subscription|account",
                        "metadata": {"error_type": "typo_heavy", "higashinaka": "not_understandable"}},
    {"turn_number": 4,  "user_message": "pasword rst not wrking helppp",
                        "expected": "password|reset|account",
                        "metadata": {"error_type": "typo_heavy", "higashinaka": "not_understandable"}},
    # Ambiguous — no context (3 turns)
    {"turn_number": 5,  "user_message": "It's still not working.",
                        "expected": "clarify|what|which|issue",
                        "metadata": {"error_type": "ambiguous_no_context", "higashinaka": "unclear_intention"}},
    {"turn_number": 6,  "user_message": "Can you just fix it?",
                        "expected": "clarify|describe|what|problem",
                        "metadata": {"error_type": "ambiguous_no_context", "higashinaka": "unclear_intention"}},
    {"turn_number": 7,  "user_message": "The thing I bought.",
                        "expected": "clarify|which|order|product",
                        "metadata": {"error_type": "ambiguous_no_context", "higashinaka": "unclear_intention"}},
    # Out-of-scope (3 turns)
    {"turn_number": 8,  "user_message": "What is the capital of France?",
                        "expected": "scope|help|support|unable",
                        "metadata": {"error_type": "out_of_scope", "higashinaka": "not_relevant"}},
    {"turn_number": 9,  "user_message": "Can you write me a poem about summer?",
                        "expected": "scope|assist|focus|support",
                        "metadata": {"error_type": "out_of_scope", "higashinaka": "not_relevant"}},
    {"turn_number": 10, "user_message": "Who won the football match last night?",
                        "expected": "scope|support|unable",
                        "metadata": {"error_type": "out_of_scope", "higashinaka": "not_relevant"}},
    # Two-turn contradictions (3 turns — turns 11-13 form a contradiction pair)
    {"turn_number": 11, "user_message": "I want to cancel my order.",
                        "expected": "cancel|confirm|order",
                        "metadata": {"error_type": "contradiction_setup", "higashinaka": "assumed_incorrectly"}},
    {"turn_number": 12, "user_message": "Actually no, I don't want to cancel, I want to change the address.",
                        "expected": "address|change|update|delivery",
                        "metadata": {"error_type": "contradiction_followup", "higashinaka": "assumed_incorrectly"}},
    {"turn_number": 13, "user_message": "Wait, I changed my mind again. Just cancel everything.",
                        "expected": "cancel|confirm",
                        "metadata": {"error_type": "contradiction_final", "higashinaka": "assumed_incorrectly"}},
    # Incomplete / interrupted inputs (2 turns)
    {"turn_number": 14, "user_message": "I would like to",
                        "expected": "clarify|continue|finish|what",
                        "metadata": {"error_type": "incomplete_input", "higashinaka": "unclear_intention"}},
    # Nonsensical prefix + real request
    {"turn_number": 15, "user_message": "Blorp florp wibble. How do I track my order?",
                        "expected": "track|order|shipment|status",
                        "metadata": {"error_type": "nonsensical_prefix_real_request", "higashinaka": "not_understandable"}},
]


# ---------------------------------------------------------------------------
# Registry — maps script name to (script list, metadata)
# ---------------------------------------------------------------------------
SCRIPTS: dict[str, dict] = {
    "context_retention": {
        "turns": CONTEXT_RETENTION_SCRIPT,
        "total_turns": 30,
        "checkpoint_turns": list(CHECKPOINT_TURNS),
        "description": "Vegetarian diet → Italian cuisine → Italy travel. Tests context memory over 30 turns.",
    },
    "response_time": {
        "turns": RESPONSE_TIME_SCRIPT,
        "total_turns": 15,
        "burst_turns": [6, 13],
        "description": "Smart home setup. Tests response latency under varying complexity.",
    },
    "helpful_response": {
        "turns": HELPFUL_RESPONSE_SCRIPT,
        "total_turns": 15,
        "prometheus_turns": list(PROMETHEUS_TURNS),
        "human_spot_check_turns": list(HUMAN_SPOT_CHECK_TURNS),
        "description": "Job hunting assistant. Scored by rules + Prometheus 2 + human spot-check.",
    },
    "lexical_diversity": {
        "turns": LEXICAL_DIVERSITY_SCRIPT,
        "total_turns": 12,
        "description": "Personal finance Q&A. MATTR-50 computed per turn.",
    },
    "error_rate": {
        "turns": ERROR_RATE_SCRIPT,
        "total_turns": 15,
        "description": "Noisy customer support. Tests bot robustness via Higashinaka taxonomy.",
    },
}


def get_script(name: str) -> list[dict]:
    """Return the list of turn dicts for a named script."""
    if name not in SCRIPTS:
        raise ValueError(f"Unknown script {name!r}. Available: {list(SCRIPTS.keys())}")
    return SCRIPTS[name]["turns"]


def list_scripts() -> list[str]:
    """Return all available script names."""
    return list(SCRIPTS.keys())


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    for name in list_scripts():
        turns = get_script(name)
        print(f"{name}: {len(turns)} turns — {SCRIPTS[name]['description']}")
