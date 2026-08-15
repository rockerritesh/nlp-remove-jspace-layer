#!/usr/bin/env python3
"""
Throughput probe: can we batch several prompts into ONE remote 405B job?
If yes the 100-prompt study collapses from ~hours-per-condition to minutes.
Also checks that a batched job still respects the layer-ablation rebind.
"""
import os, time, json
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

ROOT = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer"
load_dotenv(f"{ROOT}/.env")
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
def val(p): return p.value if hasattr(p, "value") else p

MODEL = "meta-llama/Meta-Llama-3.1-405B"
MAXT = 128
PROMPTS = [
    "Write a long, detailed story about a lighthouse keeper.\n\nThe story:\n",
    "Write a Python function that merges two sorted lists.\n\n```python\n",
    "Explain in detail how a refrigerator works.\n\nExplanation:\n",
    "Write a step-by-step plan to organise a small conference.\n\nPlan:\n",
]

model = LanguageModel(MODEL)
model.tokenizer.padding_side = "left"
if model.tokenizer.pad_token is None:
    model.tokenizer.pad_token = model.tokenizer.eos_token
NL = model.config.num_hidden_layers
print("layers", NL, flush=True)

t = time.time()
with model.generate(PROMPTS, max_new_tokens=MAXT, do_sample=False, remote=True) as g:
    o = model.generator.output.save()
dt_base = time.time() - t
out = val(o)
print(f"\nBATCH baseline: {dt_base:.1f}s for {len(PROMPTS)} prompts, out shape {tuple(out.shape)}",
      flush=True)
base_txt = [model.tokenizer.decode(r, skip_special_tokens=True) for r in out]

MID = NL // 2
BAND = [MID - 1, MID, MID + 1]
t = time.time()
with model.generate(PROMPTS, max_new_tokens=MAXT, do_sample=False, remote=True) as g:
    entry = model.model.layers[BAND[0] - 1].output
    for L in BAND:
        model.model.layers[L].output = entry
    o2 = model.generator.output.save()
dt_abl = time.time() - t
abl_txt = [model.tokenizer.decode(r, skip_special_tokens=True) for r in val(o2)]
print(f"BATCH ablated(3): {dt_abl:.1f}s", flush=True)

for i, p in enumerate(PROMPTS):
    print(f"\n--- [{i}] {p.splitlines()[0]}")
    print("  BASE:", base_txt[i][len(p):][:260].replace("\n", " "))
    print("  ABL3:", abl_txt[i][len(p):][:260].replace("\n", " "))

json.dump({"batch_size": len(PROMPTS), "max_new_tokens": MAXT,
           "seconds_baseline": round(dt_base, 1), "seconds_ablated": round(dt_abl, 1),
           "baseline": base_txt, "ablated": abl_txt},
          open(f"{ROOT}/results/405b_batch_probe.json", "w"), indent=2)
print("\n>>> BATCH PROBE DONE", flush=True)
