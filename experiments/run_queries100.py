#!/usr/bin/env python3
"""
100-query layer-ablation analysis (runs on the VM, detached).

100 prompts across 5 capability types (recall, reasoning, math, translation,
creative). For each we take the greedy baseline continuation, then measure —
teacher-forced — KL(ablated‖baseline) and top-1 agreement while:
  (A) removing each single decoder layer 0..31            -> ablation_sweep_100.png
  (B) removing a growing contiguous middle band           -> cumulative_100.png
  (C) the band-removal effect split by capability type    -> by_category.png
Saves metrics_100.json. Uses the model loader + ablation wrapper from server.py.
"""
import os, sys, json, csv, statistics as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.expanduser("~/layer-ablation"))
from server import load_model, ablate  # noqa: E402

RESULTS = os.path.expanduser("~/layer-ablation/results")
os.makedirs(RESULTS, exist_ok=True)
def log(m): print(m, flush=True)

CRIMSON, BLUE, GOOD = "#c0392b", "#2f6fb0", "#3f8f5f"
SURFACE, CONCEPT, OUTPUT = "#eaf0ea", "#fbe7c8", "#e4edf8"
CATCOLORS = {"recall": "#3f8f5f", "translation": "#2f6fb0", "math": "#e0902a",
             "reasoning": "#c0392b", "creative": "#7d3c98"}

QUERIES = [
    # --- recall (single-step factual) ---
    ("What is the capital of France?", "recall"),
    ("What is the capital of Japan?", "recall"),
    ("Who wrote Romeo and Juliet?", "recall"),
    ("What is the chemical symbol for gold?", "recall"),
    ("How many continents are there?", "recall"),
    ("What is the largest planet in our solar system?", "recall"),
    ("What year did World War II end?", "recall"),
    ("What is the freezing point of water in Celsius?", "recall"),
    ("Who painted the Mona Lisa?", "recall"),
    ("What is the tallest mountain on Earth?", "recall"),
    ("What language is mainly spoken in Brazil?", "recall"),
    ("What is the currency of Japan?", "recall"),
    ("How many legs does a spider have?", "recall"),
    ("Who was the first president of the United States?", "recall"),
    ("What is the capital of Australia?", "recall"),
    ("What gas do plants absorb from the air?", "recall"),
    ("What is the largest ocean on Earth?", "recall"),
    ("Who developed the theory of relativity?", "recall"),
    ("What planet is known as the Red Planet?", "recall"),
    ("What is the hardest natural substance?", "recall"),
    # --- reasoning (multi-hop / logic) ---
    ("If Tom is taller than Sam and Sam is taller than Al, who is shortest?", "reasoning"),
    ("All roses are flowers and all flowers need water. Do roses need water?", "reasoning"),
    ("If today is Wednesday, what day will it be in three days?", "reasoning"),
    ("A is north of B, and C is south of B. Which is furthest north?", "reasoning"),
    ("If all cats are mammals and Felix is a cat, what is Felix?", "reasoning"),
    ("Sara is older than Mia but younger than Ben. Who is the oldest?", "reasoning"),
    ("If the day before yesterday was Monday, what is today?", "reasoning"),
    ("John is behind Mary in line, and Mary is behind Sue. Who is first?", "reasoning"),
    ("Which weighs more: a kilogram of feathers or a kilogram of iron?", "reasoning"),
    ("Mary's mother has four children: April, May, June, and who?", "reasoning"),
    ("If some birds cannot fly and penguins are birds, can all birds fly?", "reasoning"),
    ("A bat and a ball cost 1.10 dollars. The bat costs 1 dollar more than the ball. How much is the ball?", "reasoning"),
    ("If you have 3 apples and take away 2, how many do you have?", "reasoning"),
    ("If all bloops are razzies and all razzies are lazzies, are all bloops lazzies?", "reasoning"),
    ("Rank fastest to slowest: a cheetah, a human, a snail.", "reasoning"),
    ("If it is raining, the ground is wet. The ground is dry. Is it raining?", "reasoning"),
    ("A clock shows 3:00. What time is it 5 hours later?", "reasoning"),
    ("There are 5 houses in a row. The red house is third. How many houses are after it?", "reasoning"),
    ("If Anna is twice as old as Ben and Ben is 6, how old is Anna?", "reasoning"),
    ("Which is furthest from the sun: Earth, Mars, or Mercury?", "reasoning"),
    # --- math (arithmetic / word problems) ---
    ("What is 17 plus 25?", "math"),
    ("What is 9 times 8?", "math"),
    ("What is 144 divided by 12?", "math"),
    ("What is 100 minus 37?", "math"),
    ("What is 15 percent of 200?", "math"),
    ("A farmer has 3 cows and buys 2 more. How many cows are there now?", "math"),
    ("What is the square root of 81?", "math"),
    ("If a book costs 12 dollars and you buy 4, what is the total?", "math"),
    ("What is 7 squared?", "math"),
    ("How many minutes are in 3 hours?", "math"),
    ("What is 250 plus 250?", "math"),
    ("If you split 20 candies among 4 kids, how many each?", "math"),
    ("What is 1000 divided by 8?", "math"),
    ("A rectangle is 5 by 8. What is its area?", "math"),
    ("What is 2 to the power of 5?", "math"),
    ("What is the sum of the first 5 positive integers?", "math"),
    ("If a train goes 60 km in 1.5 hours, what is its average speed?", "math"),
    ("What is 45 plus 55 plus 100?", "math"),
    ("How many seconds are in 5 minutes?", "math"),
    ("What is one third of 90?", "math"),
    # --- translation (surface transformation) ---
    ("Translate 'good morning' into Spanish.", "translation"),
    ("Translate 'thank you' into French.", "translation"),
    ("Translate 'hello' into German.", "translation"),
    ("Translate 'goodbye' into Italian.", "translation"),
    ("How do you say 'water' in Spanish?", "translation"),
    ("Translate 'I love you' into French.", "translation"),
    ("Translate 'cat' into German.", "translation"),
    ("Translate 'friend' into Spanish.", "translation"),
    ("Translate 'book' into French.", "translation"),
    ("Translate 'house' into German.", "translation"),
    ("Translate 'red' into Spanish.", "translation"),
    ("How do you say 'welcome' in Italian?", "translation"),
    ("Translate 'dog' into French.", "translation"),
    ("Translate 'sun' into Spanish.", "translation"),
    ("How do you say 'no' in German?", "translation"),
    ("Translate 'happy' into French.", "translation"),
    ("Translate 'school' into Spanish.", "translation"),
    ("How do you say 'family' in Italian?", "translation"),
    ("Translate 'morning' into German.", "translation"),
    ("Translate 'love' into Spanish.", "translation"),
    # --- creative (abstract / generative) ---
    ("Write a one-line metaphor for time.", "creative"),
    ("Write a haiku about the ocean.", "creative"),
    ("Give a creative name for a coffee shop.", "creative"),
    ("Describe the color blue to someone who cannot see.", "creative"),
    ("Write a one-sentence story about a lonely robot.", "creative"),
    ("Invent a name for a new planet.", "creative"),
    ("Write a short slogan for a bakery.", "creative"),
    ("Describe autumn in one poetic sentence.", "creative"),
    ("Write a two-line rhyme about the moon.", "creative"),
    ("Come up with a metaphor for the internet.", "creative"),
    ("Write a one-sentence bedtime story.", "creative"),
    ("Give a creative title for a book about dragons.", "creative"),
    ("Describe silence using a metaphor.", "creative"),
    ("Write a motivational one-liner.", "creative"),
    ("Invent a mythical creature and give it a name.", "creative"),
    ("Write a short caption for a sunset photo.", "creative"),
    ("Describe happiness without using the word happy.", "creative"),
    ("Write a one-line joke about computers.", "creative"),
    ("Come up with a band name for a jazz trio.", "creative"),
    ("Write a poetic line about rain.", "creative"),
]

log(f">>> {len(QUERIES)} queries")
log(">>> loading model")
tok, model, term = load_model(load_4bit=True)
NL = len(model.model.layers)
model.eval()

def chat_ids(p):
    out = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
    return ids.to(model.device)

log(">>> baseline generations")
base = []          # (full, prompt_len, slice, baseline_logits_cpu, category)
for i, (p, cat) in enumerate(QUERIES):
    ids = chat_ids(p); plen = ids.shape[1]
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=32, do_sample=False,
                             eos_token_id=term, pad_token_id=tok.eos_token_id)
        lb = model(gen, use_cache=False).logits[0].float()
    sl = slice(plen - 1, gen.shape[1] - 1)
    cont = tok.decode(gen[0, plen:], skip_special_tokens=True).strip().replace("\n", " ")
    base.append((gen, plen, sl, lb[sl].cpu(), cat, cont))
    if i % 20 == 0: log(f"    baseline {i}/{len(QUERIES)}")

@torch.no_grad()
def per_query(i, layer_idxs):
    full, plen, sl, lb_cpu, cat, cont = base[i]
    with ablate(model, layer_idxs):
        la = model(full, use_cache=False).logits[0].float()[sl]
    lb = lb_cpu.to(la.device)
    lpb = F.log_softmax(lb, -1); lpa = F.log_softmax(la, -1)
    kl = (lpa.exp() * (lpa - lpb)).sum(-1).mean().item()
    top = (lb.argmax(-1) == la.argmax(-1)).float().mean().item()
    return kl, top

# (A) single-layer sweep
log(">>> single-layer sweep x100")
sweep = []
pq_single = [dict() for _ in base]      # per-query: {layer: (kl, top1)}
for L in range(NL):
    vals = [per_query(i, [L]) for i in range(len(base))]
    for i, v in enumerate(vals): pq_single[i][L] = v
    kls = [v[0] for v in vals]; tops = [v[1] for v in vals]
    sweep.append({"layer": L, "kl_mean": st.mean(kls), "kl_std": st.pstdev(kls),
                  "top1_mean": st.mean(tops), "top1_std": st.pstdev(tops)})
    log(f"    L{L:2d}: KL={sweep[-1]['kl_mean']:.3f} top1={sweep[-1]['top1_mean']:.2f}")

# (B)+(C) cumulative middle-band removal + per-category
log(">>> cumulative x100 + per-category")
KS = [0, 1, 2, 4, 6, 8, 10, 12, 14]
cats = sorted(set(c for _, c in QUERIES))
cum = []; bycat = {c: [] for c in cats}
pq_cum = [dict() for _ in base]         # per-query: {k: (kl, top1)}
for k in KS:
    start = max(0, 11 - k // 2); idxs = list(range(start, start + k))
    kls, tops = [], []; ct = {c: [] for c in cats}
    for i in range(len(base)):
        kk, tt = per_query(i, idxs if k else None)
        pq_cum[i][k] = (kk, tt)
        kls.append(kk); tops.append(tt); ct[base[i][4]].append(tt)
    cum.append({"k": k, "idxs": idxs, "kl_mean": st.mean(kls), "kl_std": st.pstdev(kls),
                "top1_mean": st.mean(tops), "top1_std": st.pstdev(tops)})
    for c in cats: bycat[c].append(st.mean(ct[c]))
    log(f"    k={k:2d}: KL={cum[-1]['kl_mean']:.2f} top1={cum[-1]['top1_mean']:.2f}")

# ---------------- plots ----------------
xs = [s["layer"] for s in sweep]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
for ax in (a1, a2):
    ax.axvspan(-0.5, 4.5, color=SURFACE); ax.axvspan(4.5, 16.5, color=CONCEPT)
    ax.axvspan(16.5, NL - 0.5, color=OUTPUT)
km = np.array([s["kl_mean"] for s in sweep]); ksd = np.array([s["kl_std"] for s in sweep])
a1.plot(xs, km, "o-", color=CRIMSON, lw=2)
a1.fill_between(xs, np.clip(km - ksd, 1e-3, None), km + ksd, color=CRIMSON, alpha=0.18)
a1.set_yscale("log"); a1.set_ylabel("mean KL (log) ± std")
a1.set_title(f"Single-layer ablation over {len(QUERIES)} queries (mean ± std)")
tm = np.array([s["top1_mean"] * 100 for s in sweep]); tsd = np.array([s["top1_std"] * 100 for s in sweep])
a2.plot(xs, tm, "s-", color=BLUE, lw=2)
a2.fill_between(xs, tm - tsd, np.clip(tm + tsd, None, 100), color=BLUE, alpha=0.18)
a2.set_ylabel("top-1 agreement %"); a2.set_xlabel("decoder layer removed (0..31)"); a2.set_ylim(0, 103)
fig.tight_layout(); fig.savefig(f"{RESULTS}/ablation_sweep_100.png", dpi=150, bbox_inches="tight"); plt.close(fig)
log(">>> saved ablation_sweep_100.png")

ks = [c["k"] for c in cum]
fig, ax = plt.subplots(figsize=(9, 4.8))
km = np.array([c["kl_mean"] for c in cum]); ksd = np.array([c["kl_std"] for c in cum])
ax.plot(ks, km, "o-", color=CRIMSON, lw=2)
ax.fill_between(ks, np.clip(km - ksd, 0, None), km + ksd, color=CRIMSON, alpha=0.18)
ax.set_yscale("symlog"); ax.set_ylabel("mean KL (symlog) ± std", color=CRIMSON)
ax.set_xlabel("# contiguous middle layers removed (centered ~L11)")
axb = ax.twinx(); axb.plot(ks, [c["top1_mean"] * 100 for c in cum], "s--", color=BLUE, lw=1.8)
axb.set_ylabel("top-1 agreement %", color=BLUE); axb.set_ylim(0, 103)
ax.set_title(f"Removing the middle band over {len(QUERIES)} queries")
fig.tight_layout(); fig.savefig(f"{RESULTS}/cumulative_100.png", dpi=150, bbox_inches="tight"); plt.close(fig)
log(">>> saved cumulative_100.png")

fig, ax = plt.subplots(figsize=(9.5, 5.2))
for c in cats:
    ax.plot(KS, [v * 100 for v in bycat[c]], "o-", lw=2, color=CATCOLORS.get(c, "#555"),
            label=f"{c} (n={sum(1 for _, cc in QUERIES if cc == c)})")
ax.set_xlabel("# contiguous middle layers removed"); ax.set_ylabel("top-1 agreement % (mean)")
ax.set_ylim(0, 103); ax.grid(alpha=0.2); ax.legend(title="query type")
ax.set_title("Which capability breaks first as the workspace is removed?")
fig.tight_layout(); fig.savefig(f"{RESULTS}/by_category.png", dpi=150, bbox_inches="tight"); plt.close(fig)
log(">>> saved by_category.png")

json.dump({"n": len(QUERIES), "categories": cats, "ks": KS,
           "sweep": sweep, "cumulative": cum, "by_category": bycat},
          open(f"{RESULTS}/metrics_100.json", "w"), indent=2)
log(">>> saved metrics_100.json")

# ---------------- per-query results (JSON + CSV) ----------------
per_query_rows = []
for i, (p, cat) in enumerate(QUERIES):
    per_query_rows.append({
        "idx": i, "category": cat, "prompt": p, "baseline": base[i][5],
        "single": {L: {"kl": round(pq_single[i][L][0], 4), "top1": round(pq_single[i][L][1], 4)}
                   for L in range(NL)},
        "cumulative": {k: {"kl": round(pq_cum[i][k][0], 4), "top1": round(pq_cum[i][k][1], 4)}
                       for k in KS},
    })
json.dump(per_query_rows, open(f"{RESULTS}/per_query_results.json", "w"), indent=2)

with open(f"{RESULTS}/per_query_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["idx", "category", "prompt", "baseline",
                "kl_single_L11", "top1_single_L11",
                "top1_band_k6", "kl_band_k12", "top1_band_k12"])
    for i, (p, cat) in enumerate(QUERIES):
        w.writerow([i, cat, p, base[i][5],
                    pq_single[i][11][0], pq_single[i][11][1],
                    pq_cum[i][6][1], pq_cum[i][12][0], pq_cum[i][12][1]])
log(">>> saved per_query_results.json + per_query_results.csv")
log(">>> QUERIES100 COMPLETE")
