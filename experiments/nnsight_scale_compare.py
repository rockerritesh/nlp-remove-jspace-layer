#!/usr/bin/env python3
"""
Comparative figure across model sizes: how tolerance to middle-band removal scales.
For Llama-3.1 8B / 70B / 405B (base, NDIF remote), remove a centered middle band of
increasing FRACTION and measure next-token top-1 agreement & KL vs the intact model,
averaged over a set of prompts. One curve per model -> the scale-robustness plot.

Outputs: blog/figures/scale_compare.png, blog/figures/metrics_scale.json
"""
import os, json, statistics as st
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from nnsight import LanguageModel, CONFIG

load_dotenv("/Users/sumityadav/Documents/research/nlp-remove-jspace-layer/.env")
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])
def val(p): return p.value if hasattr(p, "value") else p

OUT = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer/blog/figures"
PROMPTS = [
    "The capital of France is",
    "Two plus two equals",
    "The opposite of hot is",
    "Water is composed of hydrogen and",
    "The first President of the United States was",
    "The largest planet in the solar system is",
    "The theory of relativity states that",
    "The Eiffel Tower is located in the city of",
]
FRACS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
MODELS = [
    ("meta-llama/Meta-Llama-3.1-8B", "8B", "#3f8f5f"),
    ("meta-llama/Meta-Llama-3.1-70B", "70B", "#2f6fb0"),
    ("meta-llama/Meta-Llama-3.1-405B", "405B", "#c0392b"),
]

data = {}
for model_id, name, color in MODELS:
    print(f"\n===== {name} =====", flush=True)
    try:
        model = LanguageModel(model_id)
        model.tokenizer.pad_token = model.tokenizer.eos_token
        model.tokenizer.padding_side = "left"
        NL = model.config.num_hidden_layers

        with model.trace(PROMPTS, remote=True):
            base = model.output.logits[:, -1, :].save()
        base = val(base).float()

        pts = []
        for frac in FRACS:
            if frac == 0.0:
                pts.append({"frac": 0.0, "removed": 0, "kl": 0.0, "top1": 1.0})
                print(f"  {name} 0%: top1=1.00 KL=0.00", flush=True)
                continue
            W = int(NL * frac); START = max(1, NL // 2 - W // 2)
            with model.trace(PROMPTS, remote=True):
                entry = model.model.layers[START - 1].output
                for L in range(START, START + W):
                    model.model.layers[L].output = entry
                lg = model.output.logits[:, -1, :].save()
            la = val(lg).float()
            lpb = F.log_softmax(base, -1); lpa = F.log_softmax(la, -1)
            kl = (lpa.exp() * (lpa - lpb)).sum(-1).mean().item()
            top1 = (base.argmax(-1) == la.argmax(-1)).float().mean().item()
            pts.append({"frac": frac, "removed": W, "kl": kl, "top1": top1})
            print(f"  {name} {int(frac*100)}% ({W}/{NL}): top1={top1:.2f} KL={kl:.2f}", flush=True)
        data[name] = {"model": model_id, "layers": NL, "color": color, "points": pts}
        del model
    except Exception as e:
        print(f"  {name} FAILED: {type(e).__name__}: {str(e)[:150]}", flush=True)
        data[name] = {"model": model_id, "error": str(e)[:150]}
    json.dump(data, open(f"{OUT}/metrics_scale.json", "w"), indent=2)

# ---- plot ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 5))
for name, d in data.items():
    if "points" not in d: continue
    xs = [p["frac"] * 100 for p in d["points"]]
    a1.plot(xs, [p["top1"] * 100 for p in d["points"]], "o-", lw=2.2,
            color=d["color"], label=f"{name} ({d['layers']} layers)")
    a2.plot(xs, [max(p["kl"], 1e-3) for p in d["points"]], "o-", lw=2.2,
            color=d["color"], label=f"{name} ({d['layers']} layers)")
a1.set_xlabel("% of middle layers removed"); a1.set_ylabel("next-token top-1 agreement %")
a1.set_ylim(0, 103); a1.set_title("Larger models keep predicting the same next token\nafter more of the middle is removed")
a1.grid(alpha=0.25); a1.legend(title="model")
a2.set_xlabel("% of middle layers removed"); a2.set_ylabel("mean KL(ablated ‖ intact)")
a2.set_yscale("log"); a2.set_title("…and their next-token distribution drifts later"); a2.grid(alpha=0.25); a2.legend(title="model")
fig.suptitle("Scale = redundancy: tolerance to middle-band removal grows with model size (Llama-3.1, base, via NDIF)",
             fontsize=13, y=1.02)
fig.tight_layout()
fig.savefig(f"{OUT}/scale_compare.png", dpi=150, bbox_inches="tight")
print("\n>>> saved scale_compare.png", flush=True)
print(">>> SCALE COMPARE DONE", flush=True)
