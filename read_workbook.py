import pandas as pd, os

wb = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\workbook_extracted"
fw_tb = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\task_bank\task_bank.py"

files = {
    "CR":    "cr_script.csv",
    "RT":    "rt_script.csv",
    "HR":    "hr_script.csv",
    "MATTR": "mattr_script.csv",
    "ER":    "er_script.csv",
}

print("="*70)
print("  WORKBOOK SCRIPT ANALYSIS")
print("="*70)

for key, fname in files.items():
    path = os.path.join(wb, fname)
    df = pd.read_csv(path)
    print(f"\n{'='*70}")
    print(f"  {key} — {fname}  ({len(df)} rows, {list(df.columns)})")
    print(f"{'='*70}")
    # print every row fully
    pd.set_option('display.max_colwidth', 120)
    pd.set_option('display.width', 200)
    print(df.to_string(index=False))

print("\n\n" + "="*70)
print("  COMPARISON: Workbook scripts vs my task_bank.py")
print("="*70)

# Compare turn counts
wb_turns = {}
for key, fname in files.items():
    df = pd.read_csv(os.path.join(wb, fname))
    wb_turns[key] = len(df)

my_turns = {"CR": 30, "RT": 15, "HR": 15, "MATTR": 12, "ER": 15}

for key in files:
    wb_n = wb_turns[key]
    my_n = my_turns[key]
    match = "MATCH" if wb_n == my_n else f"DIFFER (workbook={wb_n}, mine={my_n})"
    print(f"  {key}: workbook={wb_n} turns   my task_bank={my_n} turns   -> {match}")
