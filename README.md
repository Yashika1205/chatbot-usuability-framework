# Chatbot Usability Testing Framework

> **B.Tech Final Year Project** — An automated, multi-dimensional chatbot usability testing framework. Scores chatbots on five objective metrics from conversation logs, then checks the result against a human SUS study.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/Status-Research%20Prototype-yellow?style=flat-square)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Google%20Colab-lightgrey?style=flat-square)

---

## What is this?

This framework automatically evaluates chatbot usability across **5 objective dimensions** — without any human raters. It supports both **rule-based chatbots (Rasa)** and **LLM-backed chatbots (Ollama)**, and compares automated scores against a **human SUS study** (61 responses: 31 for Llama 3.1, 30 for Mistral; Rasa was not tested by humans). See [Results](#-results) and `human_study/`.

---

## Key Features

-  **Fully Automated** — Zero human raters needed to produce a CUS (Chatbot Usability Score)
-  **5 Usability Dimensions** — Context Retention, Response Time, Helpfulness, Lexical Diversity, Error Rate
-  **LLM-as-Judge** — Uses Prometheus 2 (Kim et al., EMNLP 2024) for helpfulness evaluation
-  **Human Comparison** — CUS compared with mean SUS per chatbot (rank comparison, Welch t-test as supporting detail; a correlation needs more than two chatbots)
-  **Mock Mode** — Run full pipeline offline without any live chatbot
-  **Reproducible** — 3-run protocol with mean ± SD reporting

---

##  System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        main.py (Entry Point)                    │
│              --mode [pilot|full|script|sus|analysis|calibrate]  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
              ┌─────────────▼──────────────┐
              │      orchestrator.py        │
              │  (manages test execution)   │
              └──────────┬─────────────────┘
                         │
          ┌──────────────┼──────────────────┐
          │              │                  │
   ┌──────▼──────┐  ┌────▼─────┐   ┌───────▼────────┐
   │  adapters/  │  │task_bank/│   │   metrics/     │
   │             │  │          │   │                │
   │ rasa_adapter│  │task_bank │   │ metric_engine  │
   │ ollama_adapt│  │.py       │   │ mattr_calculat │
   └──────┬──────┘  └────┬─────┘   │ prometheus_jud │
          │              │         │ kappa_calculat │
          │    ┌─────────▼────┐    └───────┬────────┘
          │    │  logger.py   │            │
          │    │ (JSONL logs) │            │
          └────►  scoring.py  ◄────────────┘
               │  (CUS Engine)│
               └──────┬───────┘
                      │
          ┌───────────▼──────────┐
          │     analysis.py      │
          │  (rank comparison)    │
          └───────────┬──────────┘
                      │
          ┌───────────▼──────────┐
          │      report.py       │
          │  (Human-readable     │
          │    PDF/TXT report)   │
          └──────────────────────┘
```

---

##  Five Usability Metrics

| # | Metric | Script Turns | ISO Dimension | Pass Threshold |
|---|--------|-------------|---------------|----------------|
| 1 | **Context Retention** | 30 | Effectiveness | ≥ 83% checkpoints |
| 2 | **Response Time** | 15 | Efficiency | < 1000 ms |
| 3 | **Helpful Response Ratio** | 15 | Effectiveness | > 80% helpful |
| 4 | **Lexical Diversity (MATTR-50)** | 12 | Diversity | Pilot-calibrated |
| 5 | **Error Rate** | 15 | Robustness | < 10% breakdowns |

> **Total: 87 turns per chatbot per full run**

---

##  Pipeline Workflow

```
Input: Chatbot Endpoint (Rasa / Ollama)
         │
         ▼
 ┌───────────────┐
 │  Task Bank    │  ← 87 scripted turns across 5 scripts
 │  (87 turns)   │
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  Conversation │  ← Logs stored as JSONL
 │  Logger       │
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  Metric       │  ← 5 metrics computed per turn
 │  Engine       │
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  Prometheus 2 │  ← LLM-as-judge for helpfulness (6 turns: 2, 3, 7, 10, 12, 13)
 │  Judge        │
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  CUS Scoring  │  ← Equal-weighted CUS
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  SUS Study    │  ← Human perception
 │ (Llama, Mistral)│
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  Correlation  │  ← Rank comparison, Welch t-test
 │  Analysis     │
 └───────┬───────┘
         │
         ▼
 ┌───────────────┐
 │  Final Report │  ← Automated text report
 └───────────────┘
```

---

##  Project Structure

```
chatbot_usability_framework/
│
├── 📄 main.py                  ← Entry point (all modes)
├── 📄 orchestrator.py          ← Test orchestrator
├── 📄 logger.py                ← Conversation logger (JSONL)
├── 📄 scoring.py               ← CUS engine (equal + regression weighted)
├── 📄 analysis.py              ← rank comparison, Welch t-test
├── 📄 report.py                ← Human-readable report generator
├── 📄 sus_study.py             ← SUS data collector + calculator
├── 📄 requirements.txt
│
├── 📂 adapters/
│   ├── rasa_adapter.py         ← Rasa REST adapter (mock mode supported)
│   └── ollama_adapter.py       ← Ollama LLM adapter (mock mode supported)
│
├── 📂 task_bank/
│   └── task_bank.py            ← All 5 scripted question banks (87 turns)
│
├── 📂 metrics/
│   ├── metric_engine.py        ← All 5 metric modules
│   ├── metrics_v2.py           ← Full scorer (pipeline_v3 + Colab compatible)
│   ├── mattr_calculator.py     ← MATTR-50 (Covington & McFall 2010)
│   ├── prometheus_judge.py     ← Prometheus 2 LLM-as-judge integration
│   ├── prometheus_v2.py        ← Prometheus v2 scoring engine
│   └── kappa_calculator.py     ← Cohen's Kappa (Landis & Koch 1977)
│
├── 📂 workbook/
│   ├── cr_script.csv           ← Context Retention script
│   ├── rt_script.csv           ← Response Time script
│   ├── hr_script.csv           ← Helpfulness script
│   ├── mattr_script.csv        ← Lexical Diversity script
│   └── er_script.csv           ← Error Rate script
│
├── 📂 human_study/            ← Human SUS study: report sections, responses, scoring sheet
│
└── 📂 results/
    ├── logs/                   ← JSONL conversation logs
    ├── reports/                ← Generated text reports
    └── sus_data/               ← SUS participant records (JSON)
```

---

##  Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/Yashika1205/chatbot-usuability-framework.git
cd chatbot-usuability-framework/chatbot_usability_framework
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

Download NLTK tokenizer (run once):
```python
import nltk
nltk.download('punkt_tab')
```

### 3. Smoke test (no live chatbot needed)
```bash
py main.py --mode pilot --mock
```

### 4. Full run in mock mode
```bash
py main.py --mode full --mock
```

---

##  Running with Live Chatbots

### Ollama (LLM-backed)
```bash
# Start Ollama and pull model
ollama serve
ollama pull llama3.1:8b

# Run framework
py main.py --mode full --chatbot ollama
```

### Rasa (Rule/intent-based)
```bash
# Start Rasa server
rasa run --enable-api

# Run framework
py main.py --mode full --chatbot rasa
```

### Single script run
```bash
py main.py --mode script --chatbot ollama --script error_rate
```

---

##  All Run Modes

| Mode | Command | Description |
|------|---------|-------------|
| Smoke test | `py main.py --mode pilot --mock` | Quick sanity check |
| Full mock run | `py main.py --mode full --mock` | Complete pipeline offline |
| Full live run | `py main.py --mode full --chatbot ollama` | Run with real chatbot |
| Single script | `py main.py --mode script --script error_rate` | One metric only |
| Calibration | `py main.py --mode calibrate` | MATTR-50 threshold setup |
| SUS study | `py main.py --mode sus --chatbot rasa` | Human survey collection |
| Analysis | `py main.py --mode analysis` | Correlation + regression |

---

##  MATTR-50 Calibration

Run once before the main test:
```bash
py main.py --mode calibrate
```
Enter 3 known-good and 3 known-padded responses when prompted. Thresholds are saved to `results/mattr_thresholds.json`.

---

##  Results

Latest team run (26 Oct, 3 runs, mean) and the human SUS study:

| Chatbot | CUS (automated, 0–100) | Mean SUS (human, 0–100) | SUS responses |
|---|---|---|---|
| Llama 3.1 8B | 59.64 | 71.0 | 31 |
| Mistral 7B | 51.54 | 70.9 | 30 |
| Rasa | about 24.5 (4 of 5 metrics; MATTR missing) | not tested by humans | n/a |

- Humans rated Llama and Mistral the same (Welch p = 0.98), both just above the common SUS benchmark of 68. The automated 8-point CUS gap is therefore **not confirmed** by testers.
- With only two chatbots in the human study, a correlation between CUS and SUS cannot be computed. We report a rank comparison only.
- Rasa ran locally and could not be offered to testers, so it is excluded from the human study.
- The Prometheus judge is **not yet validated** against human ratings (Cohen's kappa = 0.0 on 3 turns), so helpfulness scores should be treated as provisional.

---

##  Human SUS Study

Run by Member C. Testers opened a walkthrough page, sent six scripted messages to each chatbot (served from Google Colab), then filled a Google Form with the 10 SUS items. Scoring: odd items answer − 1, even items 5 − answer, sum × 2.5.

Everything is in [`human_study/`](human_study/): report sections (PDF), cleaned responses (CSV), scoring sheet (XLSX), walkthrough page and Colab notebook. See that folder's README.

`sus_study.py` is a command-line collector (`py main.py --mode sus --chatbot ollama`) that was **not** used for the final study.

---

##  Comparison Analysis

The final comparison is done in `human_study/SUS_Scoring_Analysis_Sheet_FINAL.xlsx` (mean SUS per chatbot, rank comparison with CUS, Welch t-test). `py main.py --mode analysis` runs Pearson, Spearman and OLS, which are only meaningful with many more chatbots than the two tested here.

---

##  Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `RASA_URL` | `http://localhost:5005` | Rasa server endpoint |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Ollama model name |
| `PROMETHEUS_MODEL` | `llama3.1:8b` | Judge model (final runs used `hf.co/prometheus-eval/prometheus-7b-v2.0-GGUF:Q4_K_M`) |
| `RASA_MOCK` | `0` | Enable Rasa mock mode |
| `OLLAMA_MOCK` | `0` | Enable Ollama mock mode |
| `PROMETHEUS_MOCK` | `0` | Enable Prometheus mock mode |

### Mock Mode (offline)
```bash
RASA_MOCK=1
OLLAMA_MOCK=1
PROMETHEUS_MOCK=1
```
or use `--mock` flag.

---

##  Google Colab (T4 GPU) Workflow

For running on Colab with GPU acceleration:

```python
# 1. Mount Drive
from google.colab import drive
drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/chatbot_usability')
os.environ['LOGS_DIR'] = '/content/drive/MyDrive/chatbot_usability/logs_v3'

# 2. Install
!pip -q install pandas requests

# 3. Smoke test
!python pipeline_v3.py --smoke

# 4. Full run (3 runs, resumable after disconnect)
!python pipeline_v3.py --runs 1 2 3

# 5. Score
!python metrics_v2.py --runs 1 2 3
```

---

##  Academic References

| Reference | Used For |
|-----------|----------|
| Atefi & Alipour (2019) arXiv:1902.06193 | Base paper |
| ISO 9241-11:2018 | Usability definition |
| Covington & McFall (2010) | MATTR-50 |
| Brooke (1996) | SUS questionnaire |
| Kim et al. (2024) EMNLP | Prometheus 2 |
| Cohen (1960) | Cohen's Kappa |
| Landis & Koch (1977) | Kappa interpretation |
| Nielsen (1993) | Response time thresholds |
| Higashinaka et al. (2015) | Error taxonomy |

---
