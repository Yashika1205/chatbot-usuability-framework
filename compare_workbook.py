import pandas as pd, os, sys
sys.path.insert(0, r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework")

WB = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\workbook_extracted\workbook"
from task_bank.task_bank import get_script

files = {
    "CR":    "cr_script.csv",
    "RT":    "rt_script.csv",
    "HR":    "hr_script.csv",
    "MATTR": "mattr_script.csv",
    "ER":    "er_script.csv",
}
script_map = {
    "CR":    "context_retention",
    "RT":    "response_time",
    "HR":    "helpful_response",
    "MATTR": "lexical_diversity",
    "ER":    "error_rate",
}

print("="*70)
print("  WORKBOOK vs MY task_bank.py — DETAILED COMPARISON")
print("="*70)

all_good = True

for key, fname in files.items():
    wb_df  = pd.read_csv(os.path.join(WB, fname))
    my_turns = get_script(script_map[key])

    wb_n  = len(wb_df["turn_number"].unique())
    my_n  = len(my_turns)
    wb_q  = wb_df["question"].tolist()
    my_q  = [t["user_message"] for t in my_turns]

    print(f"\n{'='*70}")
    print(f"  {key}  |  Workbook={len(wb_df)} rows ({wb_n} unique turns)  |  My task_bank={my_n} turns")
    print(f"{'='*70}")

    # Turn count check
    if wb_n == my_n:
        print(f"  [MATCH] Turn count: both = {wb_n}")
    else:
        print(f"  [DIFFER] Turn count: workbook={wb_n}  mine={my_n}")
        all_good = False

    # Question-by-question comparison
    print(f"\n  Question comparison (workbook vs mine):")
    max_turns = max(len(wb_df["turn_number"].unique()), my_n)
    wb_by_turn = {int(row["turn_number"]): row["question"]
                  for _, row in wb_df.drop_duplicates("turn_number").iterrows()}

    matches = 0
    differs = 0
    for t in sorted(wb_by_turn.keys()):
        wb_q_text = str(wb_by_turn[t]).strip()
        my_q_text = my_turns[t-1]["user_message"].strip() if t-1 < len(my_turns) else "MISSING"

        # Fuzzy match: first 40 chars
        wb_short = wb_q_text[:50].lower()
        my_short = my_q_text[:50].lower()

        if wb_short == my_short:
            matches += 1
            status = "SAME"
        elif any(word in my_short for word in wb_short.split()[:4] if len(word) > 3):
            matches += 1
            status = "SIMILAR"
        else:
            differs += 1
            status = "DIFFERENT"
            all_good = False

        icon = "=" if status == "SAME" else ("~" if status == "SIMILAR" else "X")
        print(f"  T{t:>2} [{icon}] WB: {wb_q_text[:60]!r}")
        if status != "SAME":
            print(f"       MY: {my_q_text[:60]!r}")

    print(f"\n  Summary: {matches} match/similar  |  {differs} different")

    # Column check
    print(f"\n  Workbook columns: {list(wb_df.columns)}")
    if "turn_type" in wb_df.columns:
        print(f"  Turn types: {sorted(wb_df['turn_type'].unique().tolist())}")
    if "judge_type" in wb_df.columns:
        print(f"  Judge types: {sorted(wb_df['judge_type'].unique().tolist())}")
    if "noise_type" in wb_df.columns:
        print(f"  Noise types: {sorted(wb_df['noise_type'].unique().tolist())}")
    if "scoring_method" in wb_df.columns:
        print(f"  Scoring methods: {sorted(wb_df['scoring_method'].unique().tolist())}")

print("\n\n" + "="*70)
print("  VERDICT")
print("="*70)

print("""
The pipeline_v3.py that ran on Colab used YOUR workbook CSVs directly
(cr_script.csv, hr_script.csv, etc.) as the source of truth.

My task_bank.py was built INDEPENDENTLY from reading the framework document
(Chatbot_Usability_Framework_v4_FINAL.docx), not from your workbook CSVs.

The questions are DIFFERENT because:
  - Your workbook has the REAL, final, agreed questions your team designed
  - My task_bank.py has questions I wrote from scratch based on the document
    descriptions (e.g. 'vegetarian diet -> Italian cuisine -> Italy travel')

What this means for the project:
  - The actual experiment (your Colab run) used YOUR workbook — CORRECT
  - My task_bank.py was never used in any real run — it was for mock testing only
  - The framework should continue using YOUR workbook CSVs as the script source
""")
