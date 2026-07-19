#!/usr/bin/env python3
"""Minimal NDIF remote smoke test: confirm the API key works, find the hosted
Llama-3.1-8B model id, and confirm remote layer hooking + logits."""
import os
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

load_dotenv()
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])   # value stays in-process, never printed
def val(p): return p.value if hasattr(p, "value") else p

PROMPT = "The Eiffel Tower is in the city of"
CANDIDATES = [
    "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "meta-llama/Meta-Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]

for MODEL in CANDIDATES:
    print(f"\n=== trying {MODEL} (remote) ===", flush=True)
    try:
        model = LanguageModel(MODEL)
        with model.trace(PROMPT, remote=True):
            hs = model.model.layers[15].output[0].save()
            logits = model.output.logits[0, -1, :].save()
        h = val(hs); lg = val(logits)
        print(f">>> SUCCESS model={MODEL}")
        print("    layer15 hidden shape:", tuple(h.shape))
        print("    next-token top:", repr(model.tokenizer.decode(lg.argmax())))
        print(">>> HOSTED_MODEL_ID =", MODEL)
        break
    except Exception as e:
        print(f"    FAILED {type(e).__name__}: {str(e)[:220]}", flush=True)
else:
    print(">>> NO CANDIDATE WORKED")
print(">>> SMOKE DONE")
