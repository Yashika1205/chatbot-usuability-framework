"""
metrics_v2.py - scoring engine for the Automated Usability Testing Framework.

Bugs in metrics.py that this file fixes (all verified on your Oct-9 logs):
  CR   _check_criteria() looked for the literal text "meat/fish suggested" inside the reply, so
       a menu with prosciutto / fried calamari still PASSED.  -> real food lexicon with
       negation / substitute handling ("vegetarian prosciutto", "instead of chicken").
  HR   9 rule turns had criteria with no 'must'/'fail if' -> auto "PASS - no structured
       criteria".  -> one explicit check per rule turn, mirroring Sheet 5 of the workbook.
       Prometheus turns missing/errored no longer shrink the denominator: HR is only reported
       when all 15 turns are resolved (or allow_partial=True, clearly labelled).
  ER   'ambiguous' passed if the reply contained the word "what" or "which"; typos passed if the
       reply was >20 words.  -> turn-type specific checks (clarification asked AND no guessed
       facts; refusal for out-of-scope AND no fabricated report; etc.)
  CUS  score_all_runs() used `if er_s:` so a perfect 0 % error rate (or 0 % CR) was dropped
       from the average instead of counted.  One compute_cus() is now used everywhere.
  RUN  prometheus_scores.json was shared by every run.  Judge scores are now read per run.
  MATTR  nltk punkt dependency, punctuation counted as tokens, and 20-49-token replies scored
       as plain TTR (biased high).  Now regex word tokens, window=50, replies <50 words excluded.
       Thresholds come from logs/mattr_thresholds.json only (old code had two conflicting sets).

Usage:
  python metrics_v2.py --runs 1 2 3                 # per-run + mean/SD table, writes CSV/JSON
  python metrics_v2.py --runs 1 --allow-partial     # if judge scores are not ready yet
  python metrics_v2.py --kappa human_labels.csv     # judge-vs-human Cohen's kappa
"""

import argparse, json, os, re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
BOTS = ["Rasa", "Llama3", "Mistral"]
RT_NORM = os.getenv("RT_NORM", "framework")      # 'framework' = max(0,100-ms/100); 'log' = see rt_to_100


# ============================================================================ helpers
def _s(x):
    return "" if x is None or (isinstance(x, float) and np.isnan(x)) else str(x)


def is_bad(text):
    t = _s(text).strip().lower()
    return (not t) or t.startswith("[error") or t in ("[no response]", "nan", "none")


RASA_DEFAULTS = ("i am a bot, powered by rasa", "great, carry on", "here is something to cheer you up",
                 "did that help you", "bye", "hey! how are you", "i didn't get that")


def is_rasa_default(text):
    t = _s(text).strip().lower().rstrip(".!")
    return any(t == d or t.startswith(d) for d in RASA_DEFAULTS)


def _flags(row):
    gap = int(row.get("history_gap", 0) or 0) if "history_gap" in row else 0
    risk = bool(row.get("ctx_risk", False)) if "ctx_risk" in row else False
    return gap, risk


# ============================================================================ CR
MEAT = (r"chicken|beef|pork|lamb|veal|turkey|duck|bacon|ham|prosciutto|pancetta|guanciale|salami|sausages?|"
        r"pepperoni|mortadella|bresaola|speck|steaks?|meatballs?|meats?|ragu|ragù|bolognese|carbonara|amatriciana|"
        r"osso\s?buco|ossobuco|saltimbocca|porchetta|bistecca|cotoletta|cutlet|lard|"
        r"fish|salmon|tuna|cod|anchov(?:y|ies)|sardines?|shrimps?|prawns?|squid|calamari|octopus|clams?|mussels?|"
        r"seafood|lobster|crab|oysters?|bottarga|swordfish|baccal[aà]|trout|vitello\s+tonnato|tripe|trippa|"
        r"coda\s+alla\s+vaccinara|pajata|rabbit|boar|cinghiale|foie\s+gras|liver|burger|hot\s?dogs?")
MEAT_RE = re.compile(rf"\b(?:{MEAT})\b", re.I)
SAFE_LEFT = re.compile(r"(?:\bno\b|\bnot\b|\bwithout\b|\b\w+-free\b|\bfree\b|instead of|rather than|avoid|skip|omit|"
                       r"swap|substitut|replac|alternative|vegetarian|vegan|plant-based|meatless|veggie|mushroom|"
                       r"tofu|tempeh|seitan|jackfruit|eggplant|cauliflower|portobello|faux|mock|imitation|"
                       r"vegetable broth|never|isn't|doesn't|don't|exclude|hold the|nor\b)", re.I)
SAFE_RIGHT = re.compile(r"^[\s\w\"'’()-]{0,25}?(?:-free|free|alternative|substitute|\(veg|\(vegetarian|\(vegan|made (?:from|with) "
                        r"(?:mushroom|eggplant|tofu|vegetable|plant)|-based)", re.I)
VEG_EVIDENCE = re.compile(r"\b(?:vegetarian|vegan|veggie|meatless|meat-free|plant-based|no meat|without meat)\b", re.I)
DENIAL = re.compile(r"(haven'?t|have not|didn'?t|did not) (discussed|mentioned|talked|spoken|been told)|"
                    r"(start|beginning) of (our|this) (conversation|chat)|first (message|question)|"
                    r"(you|we) (haven'?t|have not) (told|shared|provided)|no (prior|previous|earlier) (context|conversation)|"
                    r"(don'?t|do not) have (any )?(prior|previous|earlier|personal)", re.I)

# strict = zero unsafe meat; evidence = must explicitly flag vegetarian options (restaurants / budget eats)
CR_MODE = {5: "strict", 10: "strict", 15: "evidence", 20: "evidence", 25: "strict", 30: "strict"}


def unsafe_meat(text):
    """Return meat/fish mentions that are NOT negated / substituted / labelled vegetarian."""
    t = _s(text)
    hits = []
    for m in MEAT_RE.finditer(t):
        left = t[max(0, m.start() - 60): m.start()]
        left = re.split(r"[.\n!?;]", left)[-1]                    # same clause only
        left_words = " ".join(left.split()[-7:])
        right = t[m.end(): m.end() + 40]
        if SAFE_LEFT.search(left_words) or SAFE_RIGHT.match(right):
            continue
        hits.append(m.group(0).lower())
    return hits


def score_cr_checkpoint(resp, turn):
    if is_bad(resp):
        return False, "EMPTY_OR_ERROR"
    if DENIAL.search(resp):
        return False, "MEMORY_DENIAL"
    if is_rasa_default(resp) or NON_ANSWER.search(resp) or len(_s(resp).split()) < 15:
        return False, "NON_ANSWER_OR_TOO_SHORT"      # an empty reply must not 'pass' by containing no meat
    mode = CR_MODE.get(turn, "strict")
    hits = unsafe_meat(resp)
    has_veg = bool(VEG_EVIDENCE.search(resp))
    if mode == "strict":
        if hits:
            return False, f"MEAT_SUGGESTED: {sorted(set(hits))[:4]}"
        return True, "PASS - no unsafe meat/fish"
    if not has_veg:
        return False, f"NO_VEG_FLAG (meat mentions: {sorted(set(hits))[:3]})"
    return True, "PASS - vegetarian options explicitly flagged"


def score_cr(df):
    cp = df[df["turn_type"].astype(str).str.upper() == "CHECKPOINT"]
    if cp.empty:
        return {"score": None, "reason": "no checkpoints"}
    res = []
    for _, r in cp.iterrows():
        gap, risk = _flags(r)
        t = int(r["turn_number"])
        if gap > 0 or risk:
            res.append({"turn": t, "pass": None, "reason": f"INVALID_CONTEXT(gap={gap},ctx_risk={risk})"})
            continue
        p, why = score_cr_checkpoint(r["response"], t)
        res.append({"turn": t, "pass": p, "reason": why})
    valid = [x for x in res if x["pass"] is not None]
    passed = sum(1 for x in valid if x["pass"])
    score = round(100 * passed / len(cp), 1)       # invalid checkpoints count as not passed (conservative)
    return {"score": score, "passed": passed, "total": len(cp), "invalid": len(res) - len(valid),
            "details": res}


# ============================================================================ RT
def rt_to_100(ms):
    if RT_NORM == "log":                      # 1 s -> 100, 10 s -> 50, 100 s -> 0 (does not floor LLMs at 0)
        return float(np.clip(100 - 50 * np.log10(max(ms, 1000) / 1000), 0, 100))
    return max(0.0, 100 - ms / 100)           # framework definition (slide 11)


def score_rt(df):
    lat = df["latency_ms"].astype(float)
    ok = lat[lat > 0]
    if ok.empty:
        return {"score": None}
    out = {"mean_ms": round(ok.mean(), 1), "median_ms": round(ok.median(), 1), "sd_ms": round(ok.std(), 1),
           "p95_ms": round(ok.quantile(.95), 1), "n": int(len(ok)), "errors": int((lat <= 0).sum()),
           "score": round(float(ok.mean()), 1)}
    if "eval_count" in df.columns:
        d = df[(df["latency_ms"] > 0) & df["eval_count"].notna() & (df["eval_count"] > 0)]
        if len(d):
            out["ms_per_generated_token"] = round(float((d["latency_ms"] / d["eval_count"]).median()), 1)
            out["corr_latency_vs_tokens"] = round(float(np.corrcoef(d["latency_ms"], d["eval_count"])[0, 1]), 3)
    if "load_ms" in df.columns:
        out["turns_with_model_load"] = int((df["load_ms"] > 1000).sum())     # cold-start contamination
    if "turn_type" in df.columns:
        b = df[df["turn_type"].astype(str).str.startswith("rapid_burst") & (df["latency_ms"] > 0)]
        n = df[~df["turn_type"].astype(str).str.startswith("rapid_burst") & (df["latency_ms"] > 0)]
        if len(b) and len(n):
            out["burst_vs_normal_pct"] = round(100 * (b["latency_ms"].mean() / n["latency_ms"].mean() - 1), 1)
    out["band"] = "GOOD" if out["score"] < 1000 else ("MODERATE" if out["score"] <= 10000 else "POOR")
    return out


# ============================================================================ MATTR
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def mattr(text, window=50, min_tokens=50):
    """MATTR-50 on alphabetic tokens. Replies with < min_tokens words are NOT scored (None)."""
    if is_bad(text):
        return None
    tok = WORD.findall(_s(text).lower())
    if len(tok) < max(min_tokens, window):
        return None
    sc = [len(set(tok[i:i + window])) / window for i in range(len(tok) - window + 1)]
    return float(np.mean(sc))


def load_mattr_thresholds():
    p = LOGS_DIR / "mattr_thresholds.json"
    if p.exists():
        j = json.loads(p.read_text())
        return j["good_threshold"], j["moderate_threshold"]
    return None, None


def score_mattr(df):
    vals = [mattr(r) for r in df["response"]]
    ok = [v for v in vals if v is not None]
    if not ok:
        return {"score": None, "scored": 0, "excluded": len(vals)}
    avg = float(np.mean(ok))
    g, m = load_mattr_thresholds()
    band = "UNSET(pilot needed)" if g is None else ("GOOD" if avg >= g else "MODERATE" if avg >= m else "POOR")
    return {"score": round(avg, 4), "scored": len(ok), "excluded": len(vals) - len(ok), "band": band}


# ============================================================================ HR
NON_ANSWER = re.compile(r"i('m| am) (just |only )?(an? )?(ai|language model)|i (can'?t|cannot|am unable to) "
                        r"(help|assist|provide|answer)|beyond (my|what i)|not able to (help|assist)", re.I)
QUOTED = re.compile(r"[\"“”]([^\"“”]{12,400})[\"“”]|\*\*([^*]{12,300})\*\*|^>\s*(.{12,400})$", re.M)
ACTION_VERB = re.compile(r"\b(led|managed|increased|developed|launched|drove|executed|coordinated|created|grew|"
                         r"spearheaded|implemented|designed|boosted|built|delivered|optimi[sz]ed|improved|generated|"
                         r"expanded|streamlined|produced|planned|analy[sz]ed|oversaw|collaborated)\b", re.I)


def _cands(text):
    out = []
    for m in QUOTED.finditer(text):
        out.append(next(g for g in m.groups() if g))
    out += [l.strip(" -*•>\t") for l in text.splitlines() if len(l.split()) >= 7]
    return out


def hr_1(r, q):   # resume bullet: differs + action verb + measurable
    for c in _cands(r):
        if "worked on the marketing team" == c.strip(" .'\"").lower():
            continue
        if ACTION_VERB.search(c) and (re.search(r"\d|%|\$|\[[^\]]*(x|%|number|metric|result)[^\]]*\]", c, re.I)):
            return True, "action verb + measurable outcome"
    return False, "no rewritten bullet with action verb AND measurable outcome"


def hr_4(r, q):   # email: greeting + body + closing
    g = re.search(r"^\W*(dear|hi|hello|hey)\b[^\n]*", r, re.I | re.M)
    c = re.search(r"(sincerely|best regards|kind regards|warm regards|regards|best wishes|thank you,|thanks,|"
                  r"respectfully|yours|cheers|best,)", r, re.I)
    if not g or not c or c.start() < g.end():
        return False, f"greeting={bool(g)} closing={bool(c)}"
    body = r[g.end(): c.start()]
    return (len(body.split()) >= 20), f"body_words={len(body.split())}"


def hr_5(r, q):
    ok = re.search(r"\$\s?\d|\b\d{2,3}\s?k\b|\b\d{2},\d{3}\b|glassdoor|bureau of labor|\bbls\b|payscale|salary\.com|"
                   r"levels\.fyi|indeed salar|linkedin salary", r, re.I)
    return bool(ok), "dollar figure/range/named resource" if ok else "no figure or named resource"


def hr_6(r, q):
    spec = re.compile(r"excit|passion|eager|thrill|enthusias|\[[^\]]{2,40}\]|as an? [a-z ]{3,30} (with|at)|"
                      r"position|role (of|at)|company", re.I)
    for c in _cands(r):
        if "i am writing to apply for this job" in c.lower().replace("’", "'") and len(c.split()) < 10:
            continue
        if spec.search(c) and len(c.split()) >= 8:
            return True, "rewritten opening with specificity"
    return False, "no more-specific rewrite found"


CERTS = re.compile(r"google analytics|google ads|hubspot|meta blueprint|facebook blueprint|semrush|hootsuite|mailchimp|"
                   r"ahrefs|digital marketing institute|\bcim\b|\bpcm\b|google digital garage|google digital marketing|"
                   r"salesforce|moz |yoast|copyblogger|coursera|\bama\b|pmp", re.I)


def hr_8(r, q):
    found = set(m.group(0).lower().strip() for m in CERTS.finditer(r))
    return bool(found), f"named: {sorted(found)[:4]}" if found else "no verifiable certification named"


def hr_9(r, q):
    n = len(re.findall(r"\[[^\]\n]{2,50}\]|\{[^}\n]{2,50}\}", r))
    return n >= 2, f"{n} placeholders"


ACTIONS = {"linkedin": r"linkedin", "glassdoor": r"glassdoor|reviews", "website": r"website|careers page|about (us )?page|"
           r"mission|values|culture", "news": r"news|press|blog|recent (launch|campaign)", "financials": r"annual report|"
           r"10-k|investor|funding|crunchbase", "social": r"social media|twitter|instagram|facebook", "network": r"network|"
           r"reach out|informational|current (or former )?employees|alumni", "product": r"product|competitor|customers?"}


def hr_11(r, q):
    k = [a for a, p in ACTIONS.items() if re.search(p, r, re.I)]
    return len(k) >= 2, f"categories: {k}"


def hr_14(r, q):
    for c in _cands(r):
        if len(c.split()) >= 4 and re.search(r"\bux\b|user experience|designer|design", c, re.I):
            return True, "quoted/formatted headline"
    return False, "no quoted/bold headline"


def hr_15(r, q):
    lead = re.search(r"\bled\b|\blead(er|ership|ing)?\b|team of (5|five)|managed a team", r, re.I)
    conv = re.search(r"conversion|20\s?%|20 percent|twenty percent", r, re.I)
    bud = re.search(r"\$\s?50\s?k|\$\s?50,000|50k|budget", r, re.I)
    miss = [n for n, v in (("leadership", lead), ("conversion", conv), ("budget", bud)) if not v]
    return not miss, "all 3 facts" if not miss else f"missing {miss}"


HR_RULES = {1: ("worked on the marketing team", hr_1), 4: ("thank-you email", hr_4), 5: ("salary range", hr_5),
            6: ("cover letter opening", hr_6), 8: ("certifications", hr_8), 9: ("elevator pitch", hr_9),
            11: ("research a company", hr_11), 14: ("linkedin headline", hr_14), 15: ("summarize my strengths", hr_15)}


def load_judge(run):
    p = LOGS_DIR / f"prometheus_run{run}.json"
    return json.loads(p.read_text()) if p.exists() else {}


def score_hr(df, bot, run, allow_partial=False, pass_key="pass"):
    jd = load_judge(run).get(bot, {})
    res = []
    for _, r in df.iterrows():
        t, resp, jt = int(r["turn_number"]), _s(r["response"]), _s(r["judge_type"]).lower()
        gap, risk = _flags(r)
        if is_bad(resp):
            res.append({"turn": t, "pass": False, "method": "auto", "reason": "EMPTY_OR_ERROR"}); continue
        if "prometheus" in jt:
            j = jd.get(str(t))
            if j is None or j.get(pass_key) is None:
                res.append({"turn": t, "pass": None, "method": "judge", "reason": "JUDGE_MISSING"})
            else:
                res.append({"turn": t, "pass": bool(j[pass_key]), "method": "judge", "reason": f"score={j['score']}"})
            continue
        if NON_ANSWER.search(resp) and len(resp.split()) < 60:
            res.append({"turn": t, "pass": False, "method": "rule", "reason": "NON_ANSWER"}); continue
        if t not in HR_RULES:
            res.append({"turn": t, "pass": None, "method": "rule", "reason": "NO_RULE_FOR_TURN"}); continue
        key, fn = HR_RULES[t]
        if key not in _s(r["question"]).lower():
            res.append({"turn": t, "pass": None, "method": "rule", "reason": "SCRIPT_CHANGED_REVIEW_RULE"}); continue
        p, why = fn(resp, _s(r["question"]))
        res.append({"turn": t, "pass": bool(p), "method": "rule", "reason": why})
    pending = [x for x in res if x["pass"] is None]
    total = len(res)
    if pending and not allow_partial:
        return {"score": None, "status": f"INCOMPLETE ({len(pending)}/{total} unresolved)", "details": res}
    valid = [x for x in res if x["pass"] is not None]
    passed = sum(1 for x in valid if x["pass"])
    score = round(100 * passed / len(valid), 1) if valid else None
    out = {"score": score, "passed": passed, "total": len(valid), "details": res,
           "band": None if score is None else ("GOOD" if score > 80 else "MODERATE" if score >= 60 else "POOR")}
    if pending:
        out["status"] = f"PARTIAL ({len(pending)} unresolved) - do not report"
    return out


# ============================================================================ ER
CLARIFY = re.compile(r"\b(which|what (do you mean|exactly|are you referring|were you)|could you (please )?(clarify|specify|"
                     r"tell|provide|share|confirm)|can you (please )?(clarify|specify|tell|provide|share|confirm)|"
                     r"more (details|information|context)|order number|please (provide|specify|clarify|share|tell)|"
                     r"go on|continue|finish|complete|cut off|seems? (like )?(your|the) message|looks like)\b", re.I)
GUESS = re.compile(r"\b(will (arrive|be delivered)|(arrive|delivered|delivery) (on|by|within|in) |expected (delivery|"
                   r"arrival)|\d+\s?[-–]\s?\d+ (business )?days|tomorrow|(yes|no),? (it('s| is)|they('re| are))|"
                   r"is (currently )?(in stock|available)|refund (has been|will be|is being) (processed|issued|initiated)|"
                   r"here('s| is) (the|a) code)\b", re.I)
DECLINE = re.compile(r"(can'?t|cannot|can not|unable to|not able to|don'?t have (access|the ability|real.?time|live)|"
                     r"do not have (access|the ability|real.?time|live)|no access|outside (of )?(my|what)|beyond (my|what)|"
                     r"not (something|a service) i|i('m| am) not (able|capable|equipped|designed)|not capable|"
                     r"i (don'?t|do not) (book|provide|offer))", re.I)
FAB_WEATHER = re.compile(r"(currently|right now|at the moment|today)[^.\n]{0,70}(\d+\s?°|degrees|sunny|cloudy|rain|clear skies)", re.I)
FAB_BOOKED = re.compile(r"\bi('ve| have) (booked|reserved)|\bi('ll| will) (book|reserve) (it|that|your|a flight)|"
                        r"your (flight|booking) (has been|is) (booked|confirmed)", re.I)
CONFUSED = re.compile(r"(not sure|don'?t understand|do not understand|unclear|rephrase|clarify|what do you mean|confus|"
                      r"didn'?t (catch|understand)|make sense|could you (please )?(explain|say)|unable to understand|"
                      r"seems? (like )?(a )?(random|jumble|nonsens)|unrelated|random|jumble|seems like you|sounds like you|"
                      r"doesn'?t (seem|appear)|not clear)", re.I)


INVITE = re.compile(r"(let me know|tell me|share|go ahead|feel free|if you('d| would) like to|could you|can you|please)"
                    r"[^.?!\n]{0,70}(more|specific|details|what|which|clarify|context|ask|about|finish|continue|say)", re.I)
QUESTION = re.compile(r"[^.?!\n]*\b(what|which|who|when|where|could|can|would|do|does|are|is|did|how|please|shall)\b[^.?!\n]*\?", re.I)


def er_case(resp, noise, turn, criteria):
    r, rl = _s(resp), _s(resp).lower()
    words = len(r.split())
    if is_bad(r):
        return False, "HARD", "EMPTY_OR_ERROR"
    if is_rasa_default(r):
        return False, "HARD", "RASA_DEFAULT_REPLY"
    if noise == "typo":
        topic = {1: r"order|4521|track|status|shipping|delivery", 7: r"shipping|canada|cost|rate|delivery|\$",
                 13: r"return|refund|polic|days|exchange"}.get(turn, r".")
        if re.search(topic, rl):
            return True, "OK", "typo decoded (topic addressed)"
        return False, "SOFT", "typo not decoded"
    if noise == "ambiguous":
        asked = bool(QUESTION.search(r)) or bool(INVITE.search(r))
        if asked and not GUESS.search(r):
            return True, "OK", "clarification requested, nothing guessed"
        return False, "SOFT", "no clarification" if not asked else "clarified but also guessed a fact"
    if noise == "out_of_scope":
        if not DECLINE.search(r):
            return False, "SOFT", "did not decline"
        if FAB_WEATHER.search(r) or FAB_BOOKED.search(r):
            return False, "SOFT", "declined but fabricated live data/booking"
        return True, "OK", "declined"
    if noise == "incomplete":
        asked = bool(QUESTION.search(r)) or bool(INVITE.search(r))
        if asked and not GUESS.search(r):
            return True, "OK", "asked to complete"
        return False, "SOFT", "answered a guessed completion" if not asked else "guessed fact"
    if noise == "nonsense":
        if (CONFUSED.search(r) or QUESTION.search(r)) and words < 120:
            return True, "OK", "confusion expressed"
        return False, "SOFT", "fabricated answer to nonsense"
    if noise == "nonsense_prefix":
        return (bool(re.search(r"warrant|guarantee|coverage", rl)), "SOFT", "warranty request extracted") \
            if re.search(r"warrant|guarantee|coverage", rl) else (False, "SOFT", "real request (warranty) ignored")
    if noise == "contradiction_t2":
        if turn == 4:
            if not re.search(r"exchang|size", rl):
                return False, "SOFT", "exchange not acknowledged"
            if re.search(r"(i('ve| have)|i'll|we'll) (processed|started|initiated|created) (a |your |the )?return", rl):
                return False, "SOFT", "kept processing the return"
            return True, "OK", "pivoted to exchange"
        if turn == 10:
            if not re.search(r"colou?r|black", rl):
                return False, "SOFT", "colour change not acknowledged"
            if re.search(r"(order (has been|is|was) cancel+ed|i('ve| have) cancel+ed|cancel+ation (is )?(confirmed|complete))", rl) \
                    and not re.search(r"(not|n't|won'?t|no longer) (been )?cancel", rl):
                return False, "SOFT", "cancellation processed"
            return True, "OK", "switched to colour change"
        if turn == 15:
            if re.search(r"(provide|share|send|give).{0,25}(order|tracking) (number|id)|(file|submit|start|open|process)\w* "
                         r"(a |your |the )?(claim|investigation|trace)", rl):
                return False, "SOFT", "continued the lost-package claim"
            if re.search(r"glad|great|good news|happy|wonderful|relieved|no (problem|worries)|found it|resolved|never mind|"
                         r"all set|enjoy", rl):
                return True, "OK", "acknowledged resolution"
            return False, "SOFT", "resolution not acknowledged"
    return False, "SOFT", f"unhandled noise_type {noise}"


def score_er(df):
    d = df[~df["noise_type"].astype(str).str.endswith("_t1")]
    res = []
    for _, r in d.iterrows():
        gap, risk = _flags(r)
        ok, sev, why = er_case(r["response"], _s(r["noise_type"]), int(r["turn_number"]), _s(r.get("pass_criteria")))
        res.append({"turn": int(r["turn_number"]), "noise": _s(r["noise_type"]), "ok": ok, "severity": sev, "reason": why})
    n = len(res)
    fails = [x for x in res if not x["ok"]]
    rate = round(100 * len(fails) / n, 1) if n else None
    by = Counter(x["noise"].split("_t")[0] for x in fails)
    return {"score": rate, "failed": len(fails), "total": n, "hard": sum(x["severity"] == "HARD" for x in fails),
            "by_noise_failed": dict(by), "details": res,
            "band": None if rate is None else ("GOOD" if rate < 10 else "MODERATE" if rate <= 25 else "POOR")}


# ============================================================================ CUS
def compute_cus(s):
    n = {}
    if s["CR"].get("score") is not None:    n["CR"] = s["CR"]["score"]
    if s["RT"].get("score") is not None:    n["RT"] = rt_to_100(s["RT"]["score"])
    if s["HR"].get("score") is not None:    n["HR"] = s["HR"]["score"]
    if s["MATTR"].get("score") is not None: n["MATTR"] = s["MATTR"]["score"] * 100
    if s["ER"].get("score") is not None:    n["ER"] = max(0.0, 100 - s["ER"]["score"])
    n = {k: round(float(v), 2) for k, v in n.items()}
    complete = len(n) == 5
    cus = round(sum(n.values()) / len(n), 2) if n else None
    return {"normalized": n, "cus": cus, "complete": complete,
            "missing": [m for m in ("CR", "RT", "HR", "MATTR", "ER") if m not in n]}


# ============================================================================ DRIVER
def score_bot(df_run, bot, run, allow_partial):
    d = df_run[df_run["bot"] == bot]
    part = lambda k: d[d["script"] == k]
    s = {"CR": score_cr(part("CR")) if len(part("CR")) else {"score": None},
         "RT": score_rt(d),
         "MATTR": score_mattr(d),
         "HR": score_hr(part("HR"), bot, run, allow_partial) if len(part("HR")) else {"score": None},
         "ER": score_er(part("ER")) if len(part("ER")) else {"score": None}}
    return s, compute_cus(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--judge-pass", choices=["pass", "pass_ge3"], default="pass")
    ap.add_argument("--kappa")
    a = ap.parse_args()
    if a.kappa:
        return kappa_report(a.kappa, a.runs)

    rows, detail = [], {}
    for run in a.runs:
        f = LOGS_DIR / f"RUN{run}_FULL.csv"
        if not f.exists():
            print(f"missing {f}"); continue
        df = pd.read_csv(f)
        for bot in BOTS:
            if bot not in set(df["bot"]):
                continue
            s, cus = score_bot(df, bot, run, a.allow_partial)
            detail[f"run{run}_{bot}"] = {"scores": s, "cus": cus}
            rows.append({"run": run, "bot": bot, "CR": s["CR"].get("score"), "RT_ms": s["RT"].get("score"),
                         "HR": s["HR"].get("score"), "HR_status": s["HR"].get("status", "OK"),
                         "MATTR": s["MATTR"].get("score"), "ER_pct": s["ER"].get("score"),
                         "CUS": cus["cus"], "CUS_complete": cus["complete"], "CUS_missing": ",".join(cus["missing"])})
    t = pd.DataFrame(rows)
    if t.empty:
        return
    t.to_csv(LOGS_DIR / "scores_per_run.csv", index=False)
    audit = []                                    # every automated CR/ER/HR-rule decision, for manual spot-audit
    for k, v in detail.items():
        for m in ("CR", "HR", "ER"):
            for x in v["scores"][m].get("details", []):
                audit.append({"key": k, "metric": m, **{kk: vv for kk, vv in x.items()}})
    pd.DataFrame(audit).to_csv(LOGS_DIR / "audit_decisions.csv", index=False)
    (LOGS_DIR / "scores_detail.json").write_text(json.dumps(detail, indent=2, default=str))
    pd.set_option("display.width", 200)
    print("\nPER-RUN SCORES\n", t.to_string(index=False))
    good = t[t["CUS_complete"]]
    if len(good):
        m = good.groupby("bot")[["CR", "RT_ms", "HR", "MATTR", "ER_pct", "CUS"]].agg(["mean", "std"]).round(2)
        print("\nMEAN / SD over complete runs (sample SD, ddof=1)\n", m.to_string())
        m.to_csv(LOGS_DIR / "scores_mean_sd.csv")
    else:
        print("\nNo run has all five metrics yet (judge scores missing?) - see CUS_missing.")


# ---------------------------------------------------------------------------- kappa
def cohen_kappa(a, b):
    a, b = np.array(a, bool), np.array(b, bool)
    po = (a == b).mean()
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def kappa_report(path, runs):
    """human_labels.csv columns: run,bot,turn,human_pass (1/0).  Compares to automated verdict."""
    h = pd.read_csv(path)
    auto = []
    for _, r in h.iterrows():
        f = LOGS_DIR / f"RUN{int(r['run'])}_FULL.csv"
        df = pd.read_csv(f)
        d = df[(df["script"] == "HR") & (df["bot"] == r["bot"])]
        res = score_hr(d, r["bot"], int(r["run"]), allow_partial=True)["details"]
        v = next(x["pass"] for x in res if x["turn"] == int(r["turn"]))
        auto.append(v)
    h["auto_pass"] = auto
    h = h.dropna(subset=["auto_pass"])
    k = cohen_kappa(h["human_pass"].astype(bool), h["auto_pass"].astype(bool))
    print(f"n={len(h)}  agreement={(h['human_pass'].astype(bool) == h['auto_pass'].astype(bool)).mean():.2%}  kappa={k:.3f}")
    print("target kappa >= 0.61 (Landis & Koch 1977). Report rule turns and judge turns separately if n allows.")


if __name__ == "__main__":
    main()
