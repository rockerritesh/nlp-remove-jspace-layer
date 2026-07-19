#!/usr/bin/env python3
"""
See real generation output when a middle band is removed — on LARGE models via NDIF.
Runs on the laptop (orchestration); model executes remotely (no local GPU).

Usage:  python experiments/nnsight_remote_generate.py [MODEL] [MAX_NEW_TOKENS]
  MODEL default: meta-llama/Meta-Llama-3.1-70B   (base; NDIF pins base 8B/70B/405B)

Ablation = identity middle band via rebind: layers[L].output = layers[start-1].output.
The intervention sits directly inside `with model.generate(...)` so it applies to
every generated token (nnsight parses the block source — must be a real .py file).
"""
import os, sys, json
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

load_dotenv("/Users/sumityadav/Documents/research/nlp-remove-jspace-layer/.env")
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
def val(p): return p.value if hasattr(p, "value") else p

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Meta-Llama-3.1-70B"
MAXT = int(sys.argv[2]) if len(sys.argv) > 2 else 40
FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.25   # fraction of layers to remove

model = LanguageModel(MODEL)
NL = model.config.num_hidden_layers
W = max(6, int(NL * FRAC))          # middle band width
START = NL // 2 - W // 2
BAND = list(range(START, START + W))
print(f"model={MODEL}  layers={NL}  ablating middle band {BAND[0]}-{BAND[-1]} ({W} layers)", flush=True)

PROMPTS = [
    "The theory of relativity states that",
    "The capital of France is",
    "Step by step, to bake bread you first",
    "Once upon a time, in a distant kingdom,",
]

results = []
for p in PROMPTS:
    with model.generate(p, max_new_tokens=MAXT, remote=True) as g:
        b = model.generator.output.save()
    btxt = model.tokenizer.decode(val(b)[0], skip_special_tokens=True)

    with model.generate(p, max_new_tokens=MAXT, remote=True) as g:
        entry = model.model.layers[START - 1].output
        for L in BAND:
            model.model.layers[L].output = entry
        a = model.generator.output.save()
    atxt = model.tokenizer.decode(val(a)[0], skip_special_tokens=True)

    print("\n### PROMPT:", p, flush=True)
    print("  BASELINE:", btxt, flush=True)
    print("  ABLATED :", atxt, flush=True)
    results.append({"prompt": p, "baseline": btxt, "ablated": atxt})

out = {"model": MODEL, "layers": NL, "band": [BAND[0], BAND[-1]], "max_new_tokens": MAXT,
       "results": results}
dest = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer/blog/figures/large_model_generation.json"
json.dump(out, open(dest, "w"), indent=2)
print("\n>>> saved", dest, flush=True)
print(">>> GENERATE DONE", flush=True)
