#!/usr/bin/env python3
"""
Feasibility probe for the long-form 405B study: how long does ONE long remote
generation take, and does a 1/2/3-layer ablation survive a long roll-out?

Times: 405B baseline @ 256 tokens, then 405B with 3 middle layers removed @ 256.
Prints wall-clock so we can budget the 100-prompt run.
"""
import os, sys, time, json
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

ROOT = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer"
load_dotenv(f"{ROOT}/.env")
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
def val(p): return p.value if hasattr(p, "value") else p

MODEL = sys.argv[1] if len(sys.argv) > 1 else "meta-llama/Meta-Llama-3.1-405B"
MAXT  = int(sys.argv[2]) if len(sys.argv) > 2 else 256

PROMPT = ("Write a long, detailed story about a lighthouse keeper who discovers "
          "a message in a bottle.\n\nThe story:\n")

t0 = time.time()
model = LanguageModel(MODEL)
NL = model.config.num_hidden_layers
print(f"model={MODEL} layers={NL}  load/meta {time.time()-t0:.1f}s", flush=True)

MID = NL // 2
runs = []

for k in (0, 3):
    band = list(range(MID - k // 2, MID - k // 2 + k))
    t = time.time()
    with model.generate(PROMPT, max_new_tokens=MAXT, do_sample=False, remote=True) as g:
        if band:
            entry = model.model.layers[band[0] - 1].output
            for L in band:
                model.model.layers[L].output = entry
        o = model.generator.output.save()
    txt = model.tokenizer.decode(val(o)[0], skip_special_tokens=True)
    dt = time.time() - t
    ntok = len(val(o)[0])
    print(f"\n===== remove {k} layers (band={band}) — {dt:.1f}s, {ntok} total tokens =====",
          flush=True)
    print(txt[len(PROMPT):][:1500], flush=True)
    runs.append({"removed": k, "band": band, "seconds": round(dt, 1),
                 "total_tokens": int(ntok), "text": txt})

json.dump({"model": MODEL, "layers": NL, "max_new_tokens": MAXT, "runs": runs},
          open(f"{ROOT}/results/405b_timing_smoke.json", "w"), indent=2)
print(f"\n>>> per-generation wall clock: {[r['seconds'] for r in runs]}", flush=True)
print(">>> SMOKE DONE", flush=True)
