# Chatbot Usability Testing Framework

**Automatic Usability Testing Framework for Chatbots — B.Tech Final Year Project**

The first fully automated, multi-dimensional chatbot usability testing framework.
Scores chatbots on five objective metrics from conversation logs alone — no human
raters needed — and validates those scores against a human SUS study (n ≥ 30).

---

## Novelty

1. **Fully automated usability scoring** — zero human raters to produce a CUS score.
2. **Five usability dimensions** — not just correctness: context retention, response
   time, helpfulness, lexical diversity, and error rate.
3. **Validated against human perception** — CUS correlated against SUS (n ≥ 30);
   regression-derived weights identify which metric best predicts perceived quality.

---

## Project Structure

```
chatbot_usability_framework/
├── main.py                  ← Entry point (all modes)
├── orchestrator.py          ← Test orchestrator
├── logger.py                ← Conversation logger (JSONL)
├── scoring.py               ← CUS engine (equal + regression weighted)
├── analysis.py              ← Pearson r, Spearman ρ, OLS regression
├── report.py                ← Human-readable report generator
├── sus_study.py             ← SUS data collector + calculator
│
├── adapters/
│   ├── rasa_adapter.py      ← Rasa REST adapter (mock mode supported)
│   └── ollama_adapter.py    ← Ollama LLM adapter (mock mode supported)
│
├── task_bank/
│   └── task_bank.py         ← All 5 scripted question banks (87 turns total)
│
├── metrics/
│   ├── metric_engine.py     ← All 5 metric modules
│   ├── mattr_calculator.py  ← MATTR-50 implementation (Covington & McFall 2010)
│   ├── prometheus_judge.py  ← Prometheus 2 LLM-as-judge integration
│   └── kappa_calculator.py  ← Cohen's Kappa (Landis & Koch 1977)
│
└── results/
    ├── logs/                ← JSONL conversation logs
    ├── reports/             ← Generated text reports
    └── sus_data/            ← SUS participant records (JSON)
```

---

## Quick Start

### 1. Install dependencies
```bash
py -m pip install -r requirements.txt
```

NLTK punkt tokenizer (run once):
```python
import nltk; nltk.download('punkt_tab')
```

### 2. Smoke test (no live chatbots needed)
```bash
py main.py --mode pilot --mock
```

### 3. Full run in mock mode
```bash
py main.py --mode full --mock
```

### 4. Run with live Ollama
Start Ollama: `ollama serve` and pull a model: `ollama pull llama3.1:8b`
```bash
py main.py --mode full --chatbot ollama
```

### 5. Run with live Rasa
Start Rasa: `rasa run --enable-api`
```bash
py main.py --mode full --chatbot rasa
```

### 6. Single script run
```bash
py main.py --mode script --chatbot ollama --script error_rate
```

---

## MATTR-50 Calibration (required before main test run)

```bash
py main.py --mode calibrate
```

Enter 3 known-good and 3 known-padded responses when prompted.
Thresholds are saved to `results/mattr_thresholds.json`.

---

## Human SUS Study (n ≥ 30)

Collect participant responses interactively:
```bash
py main.py --mode sus --chatbot rasa
py main.py --mode sus --chatbot ollama
```

Or import from a CSV:
```python
from sus_study import load_sus_from_csv, aggregate_sus
records = load_sus_from_csv("my_data.csv")
# CSV columns: participant_id, chatbot_name, q1–q10
```

---

## Correlation + Regression Analysis

After the SUS study is complete:
```bash
py main.py --mode analysis
```

Results (Pearson r, Spearman ρ, regression weights) are saved to
`results/analysis_results.json` and the regression weights are automatically
loaded by `scoring.py` for the regression-weighted CUS.

---

## Chatbots Under Test

| Chatbot | Type | Default endpoint |
|---------|------|-----------------|
| Rasa | Rule/intent-based | http://localhost:5005 |
| Ollama | LLM-backed (llama3.1:8b) | http://localhost:11434 |

Override via environment variables:
```
RASA_URL=http://localhost:5005
OLLAMA_URL=http://localhost:11434
OLLAMA_MODEL=mistral:7b
```

---

## Mock Mode

All adapters support offline mock mode. Set env vars or use `--mock`:
```
RASA_MOCK=1
OLLAMA_MOCK=1
PROMETHEUS_MOCK=1
```

---

## Five Metrics

| Metric | Script Turns | ISO Dimension | Threshold (Good) |
|--------|-------------|---------------|-----------------|
| Context Retention | 30 | Effectiveness | ≥ 83% checkpoints |
| Response Time | 15 | Efficiency | < 1000 ms |
| Helpful Response Ratio | 15 | Effectiveness | > 80% helpful |
| Lexical Diversity (MATTR-50) | 12 | — | Pilot-calibrated |
| Error Rate | 15 | Robustness | < 10% breakdowns |

**Total: 87 turns per chatbot per full run.**

---

## Prometheus 2 Judge

Prometheus 2 (Kim et al., EMNLP 2024) judges 5 turns of the Helpful Response
script (turns 2, 3, 7, 10, 12). Validated against a human rater on 5 overlap
turns (turns 2, 5, 7, 10, 14) using Cohen's Kappa (target κ ≥ 0.61).

Set the judge model:
```
PROMETHEUS_MODEL=llama3.1:8b
```

---

## Key References

- Atefi & Alipour (2019) arXiv:1902.06193 — base paper
- Casas et al. (2020) ACM ICMI — gap justification
- Clerissi et al. (2026) MSR arXiv:2602.13072 — 2026 gap confirmation
- ISO 9241-11:2018 — usability definition (effectiveness, efficiency, satisfaction)
- Covington & McFall (2010) J. Quantitative Linguistics 17(2) — MATTR
- Nielsen (1993) Usability Engineering — response time thresholds
- Brooke (1996) — SUS questionnaire
- Cohen (1988) — statistical power, n ≥ 30 justification
- Landis & Koch (1977) Biometrics 33(1) — Kappa interpretation scale
- Higashinaka et al. (2015) COLING — error taxonomy
- Kim et al. (2024) EMNLP — Prometheus 2
- Zheng et al. (2023) NeurIPS — LLM judge self-preference bias

---

## Team Roles

| Member | Owns |
|--------|------|
| A | orchestrator.py, adapters/, logger.py |
| B | metrics/ (all 5 modules), mattr_calculator.py, prometheus_judge.py, kappa_calculator.py |
| C | task_bank/, sus_study.py, analysis.py, report generation |
