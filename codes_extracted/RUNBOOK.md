# Re-run runbook (Colab T4 + Windows laptop)

Windows laptop = only for editing files / uploading. Everything runs in Colab.

## 0. Files to put in Google Drive: `MyDrive/chatbot_usability/`
`pipeline_v3.py`, `prometheus_v2.py`, `metrics_v2.py`, `workbook/` (the 5 CSVs, named `cr_script.csv`, `rt_script.csv`, `hr_script.csv`, `mattr_script.csv`, `er_script.csv`), `mattr_pilot.py`.
Do NOT copy the old `logs/` into the new LOGS_DIR. Archive it as `logs_OLD_stateless`.

## 1. Setup cell
```python
from google.colab import drive; drive.mount('/content/drive')
import os
os.chdir('/content/drive/MyDrive/chatbot_usability')
os.environ['LOGS_DIR'] = '/content/drive/MyDrive/chatbot_usability/logs_v3'   # survives disconnects
os.environ['OLLAMA_MAX_LOADED_MODELS'] = '1'
os.environ['OLLAMA_NUM_PARALLEL'] = '1'
os.environ['OLLAMA_FLASH_ATTENTION'] = '1'
!pip -q install pandas requests
```
Start Ollama and Rasa the same way you did before (`ollama serve &`, then the Rasa server on :5005).
`ollama pull llama3.1` and `ollama pull mistral`.

## 2. Smoke test (B's step 2) - must print "OK to start the full run"
```python
!python pipeline_v3.py --smoke
```
Check: `history_msgs` = 8 at turn 5, no memory denial.

## 3. Full run (resumable: just re-run the same cell after a disconnect)
```python
!python pipeline_v3.py --runs 1 2 3
```
Per-bot-per-script files land in `logs_v3/run1/...`, merged file `logs_v3/RUN1_FULL.csv`.
After each run, look at the last line: error rows must be 0, ctx-risk rows must be 0.

## 4. Free VRAM, install judge
```python
!ollama stop llama3.1; ollama stop mistral
!ollama pull hf.co/<user>/<repo>:<quant>      # GGUF of prometheus-eval/prometheus-7b-v2.0
!ollama list                                   # copy the exact name
os.environ['PROMETHEUS_MODEL'] = '<exact name from ollama list>'
!python prometheus_v2.py --run 1 2 3
```
It stops with an error if the sanity gate fails - do not bypass it.

## 5. Score
```python
!python metrics_v2.py --runs 1 2 3
```
Outputs in LOGS_DIR: `scores_per_run.csv` (this is what Member C needs), `scores_mean_sd.csv`,
`scores_detail.json`, `audit_decisions.csv` (spot-check ~30 rows by hand and report agreement).

## 6. MATTR thresholds (do once, before reporting MATTR bands)
In `mattr_pilot.py` replace its own `compute_mattr` with `from metrics_v2 import mattr as compute_mattr`
(same tokenizer as the scorer), re-run it, and copy `mattr_thresholds.json` into LOGS_DIR.
