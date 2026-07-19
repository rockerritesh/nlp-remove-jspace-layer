#!/usr/bin/env python3
"""
Cross-model generation gallery via NDIF remote (nnsight): the SAME prompt through
Llama-3.1 8B / 70B / 405B, each at 0% / 25% / 50% middle-layer removal.
Shows how tolerance to middle-band ablation grows with model size.

Ablation = identity middle band via rebind (layers[L].output = layers[start-1].output),
applied at every generated token (edits sit directly in `with model.generate(...)`).
Saves incrementally to blog/figures/gallery.json.
"""
import os, json
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

load_dotenv("/Users/sumityadav/Documents/research/nlp-remove-jspace-layer/.env")
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
def val(p): return p.value if hasattr(p, "value") else p

DEST = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer/blog/figures/gallery.json"
PROMPT = "The theory of relativity states that"
MAXT = 40
FRACS = [0.0, 0.25, 0.5]
MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B", "8B"),
    ("meta-llama/Meta-Llama-3.1-70B", "70B"),
    ("meta-llama/Meta-Llama-3.1-405B", "405B"),
]

gallery = {}
for model_id, name in MODELS:
    print(f"\n===== {name} ({model_id}) =====", flush=True)
    try:
        model = LanguageModel(model_id)
        NL = model.config.num_hidden_layers
        outs = {}
        for frac in FRACS:
            W = int(NL * frac)
            START = max(1, NL // 2 - W // 2)
            with model.generate(PROMPT, max_new_tokens=MAXT, remote=True) as g:
                entry = model.model.layers[START - 1].output
                for L in range(START, START + W):
                    model.model.layers[L].output = entry
                o = model.generator.output.save()
            txt = model.tokenizer.decode(val(o)[0], skip_special_tokens=True)
            band = f"{START}-{START+W-1}" if W else "-"
            outs[str(frac)] = {"pct": int(frac * 100), "removed": W, "band": band, "text": txt}
            print(f"[{name} remove {int(frac*100)}% ({W}/{NL}, {band})]\n  {txt}", flush=True)
        gallery[name] = {"model": model_id, "layers": NL, "prompt": PROMPT, "outputs": outs}
        del model
    except Exception as e:
        print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:160]}", flush=True)
        gallery[name] = {"model": model_id, "error": f"{type(e).__name__}: {str(e)[:160]}"}
    json.dump(gallery, open(DEST, "w"), indent=2)   # incremental save
print("\n>>> saved", DEST, flush=True)
print(">>> GALLERY DONE", flush=True)
