#!/usr/bin/env python3
"""
Analysis for Experiment 9 (405B long-form, 1/2/3 layers removed).

Two metric families:

  DIVERGENCE (needs the k=0 baseline for the same prompt)
    divergence_tok    first generated-token index where the ablated run departs
                      from the baseline run. Greedy decoding, so this is caused
                      purely by the ablation.
    prefix_frac       divergence_tok / n_tokens — how far the two runs stay locked.
    uni_f1 / big_f1   unigram / bigram F1 against the baseline text: how much
                      content survives AFTER the paths split.

  INTRINSIC (no baseline needed — is the text itself still healthy?)
    distinct_1/2/3    unique n-grams / total n-grams (lexical variety)
    loop_onset        first token index that closes a repeated 20-gram, i.e. where
                      the model starts looping. n_tokens if it never loops.
    looped_frac       fraction of tokens sitting inside some repeated 10-gram.
    ttr               type/token ratio over words.
    mean_sent_len     mean sentence length in words.
    py_parse_frac     CODE ONLY. Longest prefix of the emitted code (in lines) that
                      is valid Python, / total lines. 1.0 = the whole block parses.

Writes results/longform405b/metrics.csv + summary.json.
"""
import os, sys, json, ast, csv, statistics as st
from collections import Counter, defaultdict

ROOT = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/longform405b/generations.jsonl"
OUTDIR = os.path.dirname(SRC)


# ---------------------------------------------------------------- helpers ----
def ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def distinct(seq, n):
    g = ngrams(seq, n)
    return len(set(g)) / len(g) if g else 0.0


def loop_onset(ids, n=20):
    """First token index that completes an n-gram already seen earlier."""
    seen = set()
    for i, g in enumerate(ngrams(ids, n)):
        if g in seen:
            return i + n - 1
        seen.add(g)
    return len(ids)


def looped_frac(ids, n=10):
    """Fraction of token positions covered by an n-gram that occurs more than once."""
    if len(ids) < n:
        return 0.0
    counts = Counter(ngrams(ids, n))
    covered = set()
    for i, g in enumerate(ngrams(ids, n)):
        if counts[g] > 1:
            covered.update(range(i, i + n))
    return len(covered) / len(ids)


def f1_overlap(a, b, n=1):
    """Bag-of-ngram F1 between two token-id lists (multiset intersection)."""
    ga, gb = Counter(ngrams(a, n)), Counter(ngrams(b, n))
    inter = sum((ga & gb).values())
    if not inter:
        return 0.0
    p, r = inter / max(1, sum(ga.values())), inter / max(1, sum(gb.values()))
    return 2 * p * r / (p + r)


def divergence_index(a, b):
    """(index of first differing token, did an actual mismatch occur?)

    If one run is a strict prefix of the other (the ablated run stopped early on
    EOS), there is no mismatch — it ran out. Flagging that separately matters: a
    truncated-but-matching run must not be scored as 'identical all the way'.
    """
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, True
    return min(len(a), len(b)), False


def sentences(text):
    out, cur = [], ""
    for ch in text:
        cur += ch
        if ch in ".!?" and len(cur.strip()) > 1:
            out.append(cur.strip())
            cur = ""
    if cur.strip():
        out.append(cur.strip())
    return out


def py_parse_frac(text):
    """Longest line-prefix of the emitted code that parses as Python, / total lines."""
    code = text.split("```")[0]
    lines = [l for l in code.split("\n")]
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        return 0.0
    for cut in range(len(lines), 0, -1):
        try:
            ast.parse("\n".join(lines[:cut]))
            return cut / len(lines)
        except SyntaxError:
            continue
        except Exception:
            continue
    return 0.0


# ------------------------------------------------------------------- load ----
rows = [json.loads(l) for l in open(SRC)]

# Generation is BATCHED, so every sequence in a batch is padded out to the longest
# run in that batch (pad_token == eos_token == 128001). Those pad runs are not
# model output — left in, they dominate distinct-n / loop_onset / looped_frac and
# would be reported as "the model looping". Trim each sequence at its first EOS.
EOS = 128001
for r in rows:
    ids = r["gen_ids"]
    cut = ids.index(EOS) if EOS in ids else len(ids)
    r["gen_ids"] = ids[:cut]
    r["padded_to"] = len(ids)
    r["hit_eos"] = cut < len(ids)

by_prompt = defaultdict(dict)
for r in rows:
    by_prompt[r["prompt_id"]][r["removed"]] = r
print(f"{len(rows)} generations over {len(by_prompt)} prompts")

conds = sorted({r["removed"] for r in rows})
complete = [pid for pid, d in by_prompt.items() if all(k in d for k in conds)]
print(f"conditions {conds}; {len(complete)}/{len(by_prompt)} prompts complete")

# ---------------------------------------------------------------- metrics ----
recs = []
for pid in sorted(complete):
    d = by_prompt[pid]
    base = d[0]
    bids, btxt = base["gen_ids"], base["text"]
    for k in conds:
        r = d[k]
        ids, txt = r["gen_ids"], r["text"]
        words = txt.split()
        sents = sentences(txt)
        div, mismatched = divergence_index(bids, ids)
        rec = {
            "prompt_id": pid, "category": r["category"], "removed": k,
            "n_tokens": len(ids), "n_words": len(words),
            # normalised by the BASELINE length, so a run that stops early scores
            # by how much of the baseline it actually reproduced
            "len_ratio": len(ids) / max(1, len(bids)),
            "stopped_early": 1.0 if r["hit_eos"] else 0.0,
            "divergence_tok": div if k else 0,
            "diverged": (1.0 if mismatched else 0.0) if k else 0.0,
            "prefix_frac": (div / max(1, len(bids))) if k else 1.0,
            "uni_f1": f1_overlap(bids, ids, 1) if k else 1.0,
            "big_f1": f1_overlap(bids, ids, 2) if k else 1.0,
            "distinct_1": distinct(ids, 1), "distinct_2": distinct(ids, 2),
            "distinct_3": distinct(ids, 3),
            "loop_onset": loop_onset(ids), "looped_frac": looped_frac(ids),
            "ttr": len(set(w.lower() for w in words)) / max(1, len(words)),
            "mean_sent_len": st.mean([len(s.split()) for s in sents]) if sents else 0.0,
            "py_parse_frac": py_parse_frac(txt) if r["category"] == "code" else None,
        }
        recs.append(rec)

csv_path = f"{OUTDIR}/metrics.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(recs[0].keys()))
    w.writeheader()
    w.writerows(recs)
print("wrote", csv_path)


# -------------------------------------------------------------- aggregate ----
def agg(items, field):
    vals = [x[field] for x in items if x.get(field) is not None]
    if not vals:
        return None
    return {"mean": round(st.mean(vals), 4),
            "median": round(st.median(vals), 4),
            "sd": round(st.stdev(vals), 4) if len(vals) > 1 else 0.0,
            "n": len(vals)}


FIELDS = ["divergence_tok", "prefix_frac", "diverged", "uni_f1", "big_f1",
          "distinct_1", "distinct_2", "distinct_3", "loop_onset", "looped_frac",
          "ttr", "mean_sent_len", "py_parse_frac", "n_words", "n_tokens",
          "len_ratio", "stopped_early"]

summary = {"source": SRC, "n_prompts": len(complete), "conditions": conds,
           "overall": {}, "by_category": {}}
for k in conds:
    sub = [r for r in recs if r["removed"] == k]
    summary["overall"][str(k)] = {f: agg(sub, f) for f in FIELDS}
    for cat in sorted({r["category"] for r in recs}):
        cs = [r for r in sub if r["category"] == cat]
        summary["by_category"].setdefault(cat, {})[str(k)] = {f: agg(cs, f) for f in FIELDS}

json.dump(summary, open(f"{OUTDIR}/summary.json", "w"), indent=2)
print("wrote", f"{OUTDIR}/summary.json")

# ------------------------------------------------------------------ report ---
print("\n=== OVERALL (mean) ===")
hdr = ["removed", "diverge@tok", "prefix%", "diverged%", "uni_F1", "big_F1",
       "distinct3", "loop_onset", "looped%", "TTR", "stopEarly%"]
print("  ".join(f"{h:>11}" for h in hdr))
for k in conds:
    o = summary["overall"][str(k)]
    print("  ".join(f"{v:>11}" for v in [
        k, o["divergence_tok"]["mean"], round(o["prefix_frac"]["mean"] * 100, 1),
        round(o["diverged"]["mean"] * 100, 1),
        o["uni_f1"]["mean"], o["big_f1"]["mean"], o["distinct_3"]["mean"],
        o["loop_onset"]["mean"], round(o["looped_frac"]["mean"] * 100, 1),
        o["ttr"]["mean"], round(o["stopped_early"]["mean"] * 100, 1)]))

print("\n=== BY CATEGORY: uni_F1 vs baseline / loop_onset ===")
for cat, per in summary["by_category"].items():
    line = f"{cat:>10}: "
    for k in conds:
        if k == 0:
            line += f" k0 loop@{per['0']['loop_onset']['mean']:.0f} |"
        else:
            line += (f" k{k} F1={per[str(k)]['uni_f1']['mean']:.3f} "
                     f"loop@{per[str(k)]['loop_onset']['mean']:.0f} |")
    print(line)

code0 = summary["by_category"].get("code", {})
if code0:
    print("\n=== CODE: valid-Python prefix fraction ===")
    for k in conds:
        v = code0[str(k)]["py_parse_frac"]
        if v:
            print(f"  removed {k}: mean {v['mean']:.3f}  median {v['median']:.3f} (n={v['n']})")
print("\n>>> ANALYSIS DONE")
