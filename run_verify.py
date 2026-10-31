"""run_verify.py — runs verification and saves output to file"""
import subprocess, sys, os

result = subprocess.run(
    [sys.executable,
     r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\tests\verify_e2e.py"],
    capture_output=True, text=True, encoding="utf-8",
)
out = result.stdout + result.stderr
out_path = r"C:\Users\YASHIKA\OneDrive\Desktop\AI-GTM\BTP\chatbot_usability_framework\results\verify_output.txt"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(out)
print(out)
print("Return code:", result.returncode)
