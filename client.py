#!/usr/bin/env python3
"""
Mac-side client for the layer-ablation server (runs against the SSH tunnel).

The primary way to explore is ui.html. This CLI is for scripted/plotting use,
which fits this machine's role (code + plotting; heavy compute stays on the VM).

Examples (with the tunnel up on localhost:8000):
    uv run client.py "Explain why the sky is blue." --layer 14
    uv run client.py "Write a haiku about autumn." --sweep 4 18
"""
import argparse
import json
import os

import requests

DEFAULT_API = os.getenv("ABLATION_API", "http://localhost:8000")


def call(api, prompt, layers, max_new_tokens, temperature, system_prompt):
    r = requests.post(
        api.rstrip("/") + "/generate",
        json={
            "prompt": prompt,
            "layers": list(layers),
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "system_prompt": system_prompt,
        },
        timeout=600,
    )
    r.raise_for_status()
    return r.json()


def print_single(d):
    m = d["metrics"]
    print(f"\n=== layers {d['layers']} ({d['plot_labels']}) removed ===")
    print(f"KL={m['kl_mean']}  top1_agreement={m['top1_agreement'] * 100:.1f}%  "
          f"n={m['n_positions']}  "
          f"(base {d['timing']['baseline_s']}s / abl {d['timing']['ablated_s']}s)\n")
    print("--- BASELINE ---\n" + d["baseline"])
    print("\n--- ABLATED ---\n" + d["ablated"])


def sweep(api, prompt, lo, hi, max_new_tokens, temperature, system_prompt):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers, kls, agrees = [], [], []
    for layer in range(lo, hi + 1):
        d = call(api, prompt, [layer], max_new_tokens, temperature, system_prompt)
        m = d["metrics"]
        layers.append(layer)
        kls.append(m["kl_mean"])
        agrees.append(m["top1_agreement"] * 100)
        print(f"layer {layer:2d} (L{layer + 1:<2d})  KL={m['kl_mean']:.4f}  "
              f"agreement={m['top1_agreement'] * 100:.1f}%")

    os.makedirs("results", exist_ok=True)
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(layers, kls, "o-", color="#f0a35e", label="KL divergence")
    ax1.set_xlabel("ablated decoder layer index")
    ax1.set_ylabel("mean KL(ablated ‖ baseline)", color="#f0a35e")
    ax1.tick_params(axis="y", labelcolor="#f0a35e")
    ax1.set_xticks(layers)

    ax2 = ax1.twinx()
    ax2.plot(layers, agrees, "s--", color="#4ec9b0", label="top-1 agreement %")
    ax2.set_ylabel("top-1 agreement (%)", color="#4ec9b0")
    ax2.tick_params(axis="y", labelcolor="#4ec9b0")

    plt.title(f"Effect of removing each layer\nprompt: {prompt[:60]!r}")
    fig.tight_layout()
    out = "results/sweep.png"
    plt.savefig(out, dpi=130)
    print(f"\nsaved {out}")

    with open("results/sweep.json", "w") as f:
        json.dump({"prompt": prompt, "layers": layers, "kl": kls, "agreement": agrees}, f, indent=2)
    print("saved results/sweep.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompt")
    ap.add_argument("--api", default=DEFAULT_API)
    ap.add_argument("--layer", type=int, default=14, help="single decoder layer 0..31 (plot L = layer+1)")
    ap.add_argument("--layers", type=int, nargs="+", metavar="L",
                    help="remove multiple decoder layers at once, e.g. --layers 5 10 14")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--system-prompt", default="You are a helpful assistant.")
    ap.add_argument("--sweep", nargs=2, type=int, metavar=("LO", "HI"),
                    help="sweep single layers LO..HI and plot the effect curve")
    args = ap.parse_args()

    if args.sweep:
        sweep(args.api, args.prompt, args.sweep[0], args.sweep[1],
              args.max_new_tokens, args.temperature, args.system_prompt)
    else:
        layers = args.layers if args.layers else [args.layer]
        d = call(args.api, args.prompt, layers, args.max_new_tokens,
                 args.temperature, args.system_prompt)
        print_single(d)


if __name__ == "__main__":
    main()
