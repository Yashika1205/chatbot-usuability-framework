"""
mattr_pilot.py — Improved Version with Wider Score Gap
Computes MATTR on known-good vs known-padded responses.
Outputs thresholds to use in metrics.py.
"""

import json, nltk, re
from pathlib import Path

nltk.download("punkt",     quiet=True)
nltk.download("punkt_tab", quiet=True)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)


def compute_mattr(text, window=50):
    try:
        tokens = nltk.word_tokenize(str(text).lower())
    except Exception:
        tokens = str(text).lower().split()

    if len(tokens) < 20:
        return None
    if len(tokens) < window:
        return round(len(set(tokens)) / len(tokens), 4)
    scores = [len(set(tokens[i:i+window])) / window
              for i in range(len(tokens) - window + 1)]
    return round(sum(scores) / len(scores), 4)


# ── Known-GOOD responses: rich vocabulary & technical detail ───────────
KNOWN_GOOD = [
    ("finance",
     "A Roth IRA is funded with after-tax dollars, meaning qualified withdrawals "
     "in retirement are completely tax-free, including both contributions and earnings, "
     "provided the account has been open at least five years and you are over fifty-nine "
     "and a half years old. This makes it particularly advantageous if you expect to be "
     "in a higher tax bracket during retirement than you are currently."),

    ("how-to",
     "To build an emergency fund, start by calculating three to six months of your "
     "essential monthly expenses including rent, utilities, groceries, insurance, and "
     "minimum debt payments. Open a dedicated high-yield savings account separate from "
     "your primary account to reduce temptation. Automate a fixed monthly transfer on "
     "payday before discretionary spending, and treat it as non-negotiable until you "
     "reach your target balance."),

    ("explanation",
     "Compound interest works by adding the interest earned each period back to the "
     "principal balance, so subsequent interest is calculated on a progressively larger "
     "amount. For example, one thousand dollars invested at five percent annually becomes "
     "one thousand and fifty after the first year. In the second year, interest is earned "
     "on one thousand and fifty, not the original sum, generating fifty-two dollars and "
     "fifty cents instead of fifty, and this acceleration compounds exponentially over "
     "longer time horizons."),

    ("recommendation",
     "For a first-time visitor to Italy, I would recommend starting in Rome for at least "
     "three days to visit the Colosseum, the Vatican, and the Trevi Fountain. Then take "
     "the high-speed train to Florence for two days to see the Uffizi Gallery and the "
     "Duomo. The journey takes approximately one and a half hours and costs around "
     "twenty to forty euros if booked in advance. Autumn is an ideal time to visit as "
     "crowds are smaller and temperatures are mild."),

    ("job-advice",
     "When explaining a six-month employment gap, frame it proactively rather than "
     "defensively. Mention what you did during that time, whether it was upskilling, "
     "freelancing, caregiving, or addressing a health issue, then pivot to what you "
     "learned and how it makes you a stronger candidate. A concise example: 'I took "
     "time off to complete a Google Analytics certification and care for a family member, "
     "which strengthened both my technical skills and my ability to manage competing "
     "priorities under pressure.'"),
]

# ── Improved Known-PADDED responses: highly repetitive filler ─────────
KNOWN_PADDED = [
    ("padded-vague",
     "That is a very good question and I am happy to help you with that question. "
     "There are many things to consider when thinking about this question and this topic. "
     "It really depends on your situation and what you are trying to achieve in your situation. "
     "I would suggest looking into this further to find the best option for you and your situation. "
     "I hope this information is helpful and answers your question about this topic."),

    ("padded-hedge",
     "I understand your concern and this concern is something many people ask about. "
     "There are various factors to consider when thinking about this concern. You might want to think "
     "about your personal circumstances and what works best for you in your situation. "
     "Different people have different needs and what works for one person may not work "
     "for another person. I hope this answer is helpful for your situation."),

    ("padded-generic",
     "Sure I can help with that request. There are many ways to approach this kind of situation. "
     "It is important to consider all the options that are available to you in this situation. "
     "Many people find that thinking carefully about options helps them make a good decision. "
     "I hope this information is useful and answers your question about this topic."),

    ("padded-repeat",
     "This is a very important question and a very common question. Many people ask this "
     "very question and it is a very valid question to have. The very best thing you can "
     "do is to think very carefully about your very specific situation because every "
     "situation is very different and what works very well for one person may not work "
     "very well for another person in a very different situation."),

    ("padded-deflect",
     "I would recommend consulting a professional regarding this area as they "
     "will be best placed to give you advice regarding your needs. "
     "Everyone's situation is unique and it is always best to seek expert guidance "
     "regarding your needs rather than general information. A qualified professional "
     "will be able to provide you with advice regarding your needs."),
]


def run_pilot():
    print("=" * 55)
    print("  MATTR PILOT — Setting Lexical Diversity Thresholds")
    print("=" * 55)
    print("\nKnown-GOOD responses:")

    good_scores = []
    for domain, text in KNOWN_GOOD:
        score = compute_mattr(text)
        good_scores.append(score)
        print(f"  [{domain}]: {score}")

    print("\nKnown-PADDED responses:")
    pad_scores = []
    for label, text in KNOWN_PADDED:
        score = compute_mattr(text)
        pad_scores.append(score)
        print(f"  [{label}]: {score}")

    valid_good = [s for s in good_scores if s is not None]
    valid_pad  = [s for s in pad_scores  if s is not None]

    g_avg = round(sum(valid_good) / len(valid_good), 4)
    p_avg = round(sum(valid_pad)  / len(valid_pad),  4)

    # Threshold = midpoint between good and padded averages
    threshold    = round((g_avg + p_avg) / 2, 4)
    moderate_low = round(threshold * 0.85, 4)

    print(f"\n  Good average:   {g_avg}")
    print(f"  Padded average: {p_avg}")
    print(f"  Gap:            {round(g_avg - p_avg, 4)}")

    print(f"\n{'='*55}")
    print(f"  THRESHOLDS TO USE IN metrics.py:")
    print(f"  MATTR_GOOD_THRESHOLD     = {threshold}")
    print(f"  MATTR_MODERATE_THRESHOLD = {moderate_low}")
    print(f"{'='*55}")

    if g_avg - p_avg < 0.05:
        print("\n  ⚠ WARNING: Gap between good and padded is small (<0.05).")
        print("  MATTR may not be a strong discriminator for your scripts.")
    else:
        print("\n  ✓ Excellent separation between Good and Padded responses!")

    result = {
        "good_threshold":     threshold,
        "moderate_threshold": moderate_low,
        "good_avg":           g_avg,
        "padded_avg":         p_avg,
        "gap":                round(g_avg - p_avg, 4),
        "good_scores":        dict(zip([d for d,_ in KNOWN_GOOD],  good_scores)),
        "padded_scores":      dict(zip([d for d,_ in KNOWN_PADDED], pad_scores)),
    }

    out = LOGS_DIR / "mattr_thresholds.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  ✓ Thresholds saved to {out}")
    return result


def patch_thresholds(metrics_path="metrics.py"):
    """Auto-patches MATTR thresholds into metrics.py from pilot output."""
    t_file = LOGS_DIR / "mattr_thresholds.json"
    if not t_file.exists():
        print("Run run_pilot() first.")
        return

    with open(t_file) as f:
        t = json.load(f)

    with open(metrics_path, "r") as f:
        code = f.read()

    code = re.sub(
        r'MATTR_GOOD_THRESHOLD\s*=\s*[\d.]+',
        f'MATTR_GOOD_THRESHOLD     = {t["good_threshold"]}',
        code
    )
    code = re.sub(
        r'MATTR_MODERATE_THRESHOLD\s*=\s*[\d.]+',
        f'MATTR_MODERATE_THRESHOLD = {t["moderate_threshold"]}',
        code
    )

    with open(metrics_path, "w") as f:
        f.write(code)

    print(f"✓ metrics.py patched:")
    print(f"  MATTR_GOOD_THRESHOLD     = {t['good_threshold']}")
    print(f"  MATTR_MODERATE_THRESHOLD = {t['moderate_threshold']}")


if __name__ == "__main__":
    result = run_pilot()
    patch_thresholds()