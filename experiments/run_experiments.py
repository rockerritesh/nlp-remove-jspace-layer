#!/usr/bin/env python3
"""
Layer-analysis experiments for the blog. Runs ON THE VM (GPU), detached.

Produces (in ~/layer-ablation/results/):
  1. layer_umap_grid.png  — per-layer 2D projection of hidden states, 4 classes
  2. separability.png     — class decodability (kNN, full-dim) vs visual
                            separation (silhouette, 2D) across layers
  3. ablation_sweep.png   — effect of removing each single layer
                            (KL, top-1 agreement, entropy, perplexity)
  4. cumulative.png        — effect of removing a growing middle band
  5. metrics.json          — all numbers

Reuses the model loader + ablation wrapper from server.py.
"""
import os, sys, json, math, collections, statistics as st

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

# palette shared with the architecture figure
CLASS_COLORS = ["#4ec9b0", "#6ea8fe", "#f0a35e", "#c0392b"]
SURFACE, CONCEPT, OUTPUT = "#eaf0ea", "#fbe7c8", "#e4edf8"
CRIMSON, BLUE, GOOD = "#c0392b", "#2f6fb0", "#3f8f5f"

# ------------------------------------------------------------------ model
log(">>> loading model")
tok, model, term = load_model(load_4bit=True)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "right"
NL = len(model.model.layers)          # 32 decoder layers -> 33 hidden states
model.eval()
log(f">>> model ready: {NL} decoder layers")

# ------------------------------------------------------------------ dataset (4 classes)
from sklearn.datasets import fetch_20newsgroups
CATS = ["sci.space", "rec.sport.baseball", "comp.graphics", "talk.politics.mideast"]
NAMES = ["space", "baseball", "graphics", "mideast"]
PER = 100
log(">>> fetching 20-newsgroups (4 classes)")
ds = fetch_20newsgroups(subset="train", categories=CATS,
                        remove=("headers", "footers", "quotes"))
by = collections.defaultdict(list)
for t, y in zip(ds.data, ds.target):
    t = t.strip()
    if len(t) > 40:
        by[y].append(t[:600])
texts, labels = [], []
for y in range(4):
    for t in by[y][:PER]:
        texts.append(t); labels.append(y)
labels = np.array(labels)
log(f">>> {len(texts)} samples ({[int((labels==c).sum()) for c in range(4)]})")

# ------------------------------------------------------------------ per-layer embeddings
@torch.no_grad()
def embed(texts, bs=8):
    H = [[] for _ in range(NL + 1)]
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                  truncation=True, max_length=128).to(model.device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        mask = enc["attention_mask"].unsqueeze(-1).float()
        for l, hs in enumerate(out.hidden_states):
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
            H[l].append(pooled.float().cpu().numpy())
        del out
    return [np.concatenate(h, 0) for h in H]

log(">>> extracting per-layer hidden states")
Hs = embed(texts)

# ------------------------------------------------------------------ projections
from sklearn.decomposition import PCA
try:
    import umap
    PROJ = "UMAP"
    def project(X):
        X50 = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
        return umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(X50)
except Exception as e:
    from sklearn.manifold import TSNE
    PROJ = "t-SNE"
    log(f">>> umap unavailable ({e!r}); using t-SNE")
    def project(X):
        X50 = PCA(n_components=min(50, X.shape[1])).fit_transform(X)
        return TSNE(n_components=2, perplexity=30, init="pca", random_state=42).fit_transform(X50)

log(f">>> projecting each layer with {PROJ}")
P2d = []
for l in range(NL + 1):
    P2d.append(project(Hs[l]))
    if l % 6 == 0:
        log(f"    projected L{l}")

# grid figure
ncol = 6
nrow = math.ceil((NL + 1) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2.4, nrow * 2.4))
axes = np.atleast_2d(axes)
for l in range(NL + 1):
    ax = axes[l // ncol][l % ncol]
    P = P2d[l]
    for c in range(4):
        m = labels == c
        ax.scatter(P[m, 0], P[m, 1], s=5, c=CLASS_COLORS[c], alpha=0.75, linewidths=0)
    ax.set_title(f"L{l}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
for l in range(NL + 1, nrow * ncol):
    axes[l // ncol][l % ncol].axis("off")
fig.suptitle(f"Per-layer {PROJ} of Llama-3.1-8B hidden states — 4 topic classes "
             f"({', '.join(NAMES)})", fontsize=13, y=1.002)
fig.tight_layout()
fig.savefig(f"{RESULTS}/layer_umap_grid.png", dpi=140, bbox_inches="tight")
plt.close(fig)
log(">>> saved layer_umap_grid.png")

# ------------------------------------------------------------------ separability per layer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

knn_acc, sil = [], []
for l in range(NL + 1):
    Xs = StandardScaler().fit_transform(Hs[l])
    knn_acc.append(float(cross_val_score(KNeighborsClassifier(15), Xs, labels, cv=5).mean()))
    try:
        sil.append(float(silhouette_score(P2d[l], labels)))
    except Exception:
        sil.append(0.0)
log(">>> separability computed")

fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axvspan(-0.5, 5.5, color=SURFACE); ax.axvspan(5.5, 17.5, color=CONCEPT)
ax.axvspan(17.5, NL + 0.5, color=OUTPUT)
xs = list(range(NL + 1))
ax.plot(xs, knn_acc, "o-", color=CRIMSON, lw=2, label="class decodability — kNN acc (full hidden dim)")
ax.axhline(0.25, ls=":", color="#888", lw=1); ax.text(0.2, 0.265, "chance (4 classes)", fontsize=8, color="#888")
ax.set_ylabel("kNN accuracy", color=CRIMSON); ax.set_ylim(0, 1.02)
ax.set_xlabel("hidden-state layer L   (0 = embeddings … 32 = final)")
ax2 = ax.twinx()
ax2.plot(xs, sil, "s--", color=BLUE, lw=1.8, label="visual separation — silhouette (2D projection)")
ax2.set_ylabel("silhouette (2D)", color=BLUE)
ax.set_title("Which layers hold class structure?  Info stays decodable (red) even where "
             "clusters visually merge (blue dip) = the entangled 'workspace'")
lines = ax.get_lines()[:1] + ax2.get_lines()
ax.legend(lines, [l.get_label() for l in lines], loc="lower center", fontsize=8, framealpha=0.9)
fig.tight_layout()
fig.savefig(f"{RESULTS}/separability.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log(">>> saved separability.png")

# ------------------------------------------------------------------ ablation metrics
EVAL = [
    "Explain why the sky is blue.",
    "What is the capital of France, and what language is spoken there?",
    "If a train travels 60 km in 1.5 hours, what is its average speed?",
    "Write one sentence about the moon.",
    "Translate 'good morning' into Spanish.",
    "Summarize what photosynthesis does in one sentence.",
    "A farmer has 3 cows and buys 2 more. How many cows now?",
    "Name a primary color.",
]

def chat_ids(p):
    out = tok.apply_chat_template([{"role": "user", "content": p}],
                                  add_generation_prompt=True, return_tensors="pt")
    ids = out if isinstance(out, torch.Tensor) else out["input_ids"]
    return ids.to(model.device)

log(">>> building baseline continuations")
baselines = []          # (full_ids, prompt_len, baseline_logits_slice)
for p in EVAL:
    ids = chat_ids(p); plen = ids.shape[1]
    with torch.no_grad():
        gen = model.generate(ids, max_new_tokens=48, do_sample=False,
                             eos_token_id=term, pad_token_id=tok.eos_token_id)
        base_logits = model(gen, use_cache=False).logits[0].float()
    sl = slice(plen - 1, gen.shape[1] - 1)
    baselines.append((gen, plen, sl, base_logits[sl].cpu()))

@torch.no_grad()
def metrics_for(layer_idxs):
    kls, tops, ents, ppls = [], [], [], []
    for (full, plen, sl, lb_cpu) in baselines:
        with ablate(model, layer_idxs):
            la = model(full, use_cache=False).logits[0].float()[sl]
        lb = lb_cpu.to(la.device)
        lpb = F.log_softmax(lb, -1); lpa = F.log_softmax(la, -1)
        kls.append((lpa.exp() * (lpa - lpb)).sum(-1).mean().item())
        tops.append((lb.argmax(-1) == la.argmax(-1)).float().mean().item())
        ents.append((-(lpa.exp() * lpa).sum(-1)).mean().item())
        tgt = full[0, plen:plen + la.shape[0]]
        logp = lpa[torch.arange(la.shape[0], device=la.device), tgt]
        ppls.append(torch.exp(-logp.mean()).item())
    return dict(kl=st.mean(kls), top1=st.mean(tops), entropy=st.mean(ents), ppl=st.mean(ppls))

base_m = metrics_for(None)
log(f">>> baseline: entropy={base_m['entropy']:.3f} ppl={base_m['ppl']:.2f}")

log(">>> single-layer ablation sweep")
sweep = []
for L in range(NL):
    m = metrics_for([L]); m["layer"] = L; sweep.append(m)
    log(f"    ablate decoder L{L:2d}: KL={m['kl']:.3f} top1={m['top1']:.2f} "
        f"ent={m['entropy']:.2f} ppl={m['ppl']:.1f}")

# ablation sweep figure
xs = [m["layer"] for m in sweep]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 7.2), sharex=True)
for ax in (a1, a2):
    ax.axvspan(4.5, 16.5, color=CONCEPT); ax.axvspan(-0.5, 4.5, color=SURFACE)
    ax.axvspan(16.5, NL - 0.5, color=OUTPUT)
a1.plot(xs, [m["kl"] for m in sweep], "o-", color=CRIMSON, lw=2, label="KL(ablated‖baseline)")
a1.set_yscale("log"); a1.set_ylabel("mean KL (log)", color=CRIMSON)
a1b = a1.twinx()
a1b.plot(xs, [m["ppl"] for m in sweep], "^--", color="#8a5a00", lw=1.6, label="perplexity")
a1b.axhline(base_m["ppl"], ls=":", color="#8a5a00", lw=1); a1b.set_ylabel("perplexity", color="#8a5a00")
a1.set_title("Effect of removing each single decoder layer (concept band shaded)")
a2.plot(xs, [m["top1"] * 100 for m in sweep], "s-", color=BLUE, lw=2, label="top-1 agreement %")
a2.set_ylabel("top-1 agreement (%)", color=BLUE); a2.set_ylim(0, 102)
a2b = a2.twinx()
a2b.plot(xs, [m["entropy"] for m in sweep], "d--", color=GOOD, lw=1.6, label="entropy H(p_abl)")
a2b.axhline(base_m["entropy"], ls=":", color=GOOD, lw=1); a2b.set_ylabel("entropy (nats)", color=GOOD)
a2.set_xlabel("decoder layer removed  (0..31)")
fig.tight_layout()
fig.savefig(f"{RESULTS}/ablation_sweep.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log(">>> saved ablation_sweep.png")

# ------------------------------------------------------------------ cumulative middle removal
log(">>> cumulative middle-band removal")
cum = []
for k in [0, 1, 2, 4, 6, 8, 10, 12, 14]:
    start = max(0, 11 - k // 2)
    idxs = list(range(start, start + k))
    m = metrics_for(idxs if k else None); m["k"] = k; m["idxs"] = idxs; cum.append(m)
    log(f"    remove {k:2d} mid layers {idxs}: KL={m['kl']:.2f} top1={m['top1']:.2f} ppl={m['ppl']:.1f}")

ks = [m["k"] for m in cum]
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(ks, [m["kl"] for m in cum], "o-", color=CRIMSON, lw=2, label="KL(ablated‖baseline)")
ax.set_yscale("symlog"); ax.set_ylabel("mean KL (symlog)", color=CRIMSON)
ax.set_xlabel("number of contiguous middle layers removed (centered ~L11)")
axb = ax.twinx()
axb.plot(ks, [m["top1"] * 100 for m in cum], "s--", color=BLUE, lw=1.8, label="top-1 agreement %")
axb.set_ylabel("top-1 agreement (%)", color=BLUE); axb.set_ylim(0, 102)
ax.set_title("Removing the workspace: one layer is survivable, the whole band collapses")
fig.tight_layout()
fig.savefig(f"{RESULTS}/cumulative.png", dpi=150, bbox_inches="tight")
plt.close(fig)
log(">>> saved cumulative.png")

# ------------------------------------------------------------------ dump metrics
out = {
    "model": "meta-llama/Llama-3.1-8B-Instruct",
    "projection": PROJ,
    "dataset": {"source": "20newsgroups", "classes": NAMES, "per_class": PER, "n": len(texts)},
    "baseline": base_m,
    "separability": {"knn_acc": knn_acc, "silhouette": sil},
    "ablation_sweep": sweep,
    "cumulative": cum,
}
with open(f"{RESULTS}/metrics.json", "w") as f:
    json.dump(out, f, indent=2)
log(">>> saved metrics.json")
log(">>> EXPERIMENTS COMPLETE")
