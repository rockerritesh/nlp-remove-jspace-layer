#!/usr/bin/env python3
"""
Figures for Experiment 9 (405B long-form, 1/2/3 layers removed).

  longform_metrics.png   2x2 small multiples — the four headline metrics vs k,
                         one line per capability family.
  longform_onset.png     ECDF: fraction of prompts that have already diverged from
                         the baseline by token t, one curve per k (ordinal ramp).
  longform_quality.png   code validity + looped-token share vs k.

Reads results/longform405b/metrics.csv.
"""
import os, sys, csv, statistics as st
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer"
SRC = sys.argv[1] if len(sys.argv) > 1 else f"{ROOT}/results/longform405b/metrics.csv"
FIGDIR = f"{ROOT}/blog/figures"
os.makedirs(FIGDIR, exist_ok=True)

# --- design tokens (validated categorical palette, light mode) ----------------
SURFACE = "#fcfcfb"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8983"
GRID = "#e7e6e2"
# categorical slots 1-5, fixed order — never cycled
CAT = {"narrative": "#2a78d6", "code": "#eb6834", "explain": "#1baf7a",
       "procedure": "#eda100", "reasoning": "#e87ba4"}
ORDER = ["narrative", "code", "explain", "procedure", "reasoning"]
# ordinal blue ramp for k = 1,2,3 (magnitude, not identity)
ORD = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "legend.frameon": False,
})

# ------------------------------------------------------------------- load ----
rows = list(csv.DictReader(open(SRC)))
for r in rows:
    for k, v in r.items():
        if k not in ("prompt_id", "category"):
            r[k] = float(v) if v not in ("", "None") else None
KS = sorted({int(r["removed"]) for r in rows})


def mean_by(field, cat=None):
    """mean of `field` per k, optionally restricted to one category"""
    out = {}
    for k in KS:
        vals = [r[field] for r in rows
                if int(r["removed"]) == k and (cat is None or r["category"] == cat)
                and r[field] is not None]
        out[k] = st.mean(vals) if vals else None
    return out


def style_axis(ax, title, ylabel):
    ax.set_title(title, fontsize=10, weight="bold", pad=8, loc="left")
    ax.set_ylabel(ylabel, fontsize=8.5)
    ax.set_xlabel("layers removed (of 126)", fontsize=8.5)
    ax.set_xticks(KS)
    ax.grid(axis="x", visible=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def series_panel(ax, field, title, ylabel, scale=1.0):
    ends = []
    for cat in ORDER:
        m = mean_by(field, cat)
        xs = [k for k in KS if m[k] is not None]
        ys = [m[k] * scale for k in xs]
        if not xs:
            continue
        ax.plot(xs, ys, color=CAT[cat], lw=2, marker="o", ms=5.5,
                mec=SURFACE, mew=1.5, zorder=3, clip_on=False)
        ends.append([ys[-1], ys[-1], cat, xs[-1]])   # [true_y, label_y, cat, x]
    style_axis(ax, title, ylabel)
    ax.set_xlim(min(KS) - 0.1, max(KS) + 0.9)

    # Direct labels at the right end (mandatory at >=4 series, and the relief for
    # the contrast-WARN slots). Series can converge, so push labels apart to a
    # minimum gap and tie each back to its line with a leader when it moved.
    lo, hi = ax.get_ylim()
    gap = (hi - lo) * 0.062
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        if ends[i][1] - ends[i - 1][1] < gap:
            ends[i][1] = ends[i - 1][1] + gap
    shift = max(0.0, (ends[-1][1] - hi) if ends else 0.0)
    for e in ends:
        e[1] -= shift
    for true_y, label_y, cat, x in ends:
        ax.annotate(cat, xy=(x, true_y), xytext=(x + 0.12, label_y),
                    va="center", fontsize=7.5, color=INK2, annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=GRID, lw=0.8,
                                    shrinkA=0, shrinkB=3)
                    if abs(label_y - true_y) > gap * 0.25 else None)


# ============================================================ FIG 1: metrics ==
fig, axes = plt.subplots(2, 2, figsize=(11, 7.6))
series_panel(axes[0][0], "prefix_frac",
             "A · How long the ablated run tracks the baseline",
             "% of baseline tokens reproduced\nbefore the first difference", scale=100)
series_panel(axes[0][1], "uni_f1",
             "B · Content overlap with the baseline text",
             "unigram F1 vs baseline")
series_panel(axes[1][0], "loop_onset",
             "C · Where the model starts looping",
             "token index of first repeated 20-gram")
series_panel(axes[1][1], "distinct_3",
             "D · Lexical variety of the output",
             "distinct-3 (unique 3-grams / total)")

fig.suptitle("Removing 1–3 of 126 layers from Llama-3.1-405B: effect over a 1024-token generation",
             fontsize=12.5, weight="bold", color=INK, x=0.012, ha="left", y=0.985)
fig.text(0.012, 0.945,
         "Greedy decoding, 100 long-form prompts × 5 capability families. "
         "Layers removed are contiguous and centred on L63 (the network midpoint).",
         fontsize=8.5, color=MUTED, ha="left")
handles = [plt.Line2D([], [], color=CAT[c], lw=2, marker="o", ms=5.5, label=c)
           for c in ORDER]
fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8.5,
           bbox_to_anchor=(0.5, -0.005), labelcolor=INK2)
fig.tight_layout(rect=[0, 0.035, 1, 0.925])
fig.savefig(f"{FIGDIR}/longform_metrics.png", dpi=200)
print("wrote", f"{FIGDIR}/longform_metrics.png")

# ============================================================== FIG 2: ECDF ==
fig2, ax = plt.subplots(figsize=(7.4, 4.6))
for k in [x for x in KS if x > 0]:
    vals = sorted(r["divergence_tok"] for r in rows if int(r["removed"]) == k)
    if not vals:
        continue
    n = len(vals)
    xs, ys = [0], [0]
    for i, v in enumerate(vals):
        xs.append(v); ys.append((i + 1) / n * 100)
    med = st.median(vals)
    ax.step(xs, ys, where="post", color=ORD[k], lw=2, zorder=3,
            label=f"{k} layer{'s' if k > 1 else ''} removed — median token {med:.0f}")
ax.set_title("How soon a tiny ablation shows up in the text", fontsize=11,
             weight="bold", loc="left", pad=8)
ax.set_xlabel("token index of first difference from the baseline", fontsize=8.5)
ax.set_ylabel("% of the 100 prompts that have diverged", fontsize=8.5)
ax.set_ylim(0, 101)
ax.grid(axis="x", visible=False)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.legend(fontsize=8.5, loc="lower right", labelcolor=INK2)
fig2.tight_layout()
fig2.savefig(f"{FIGDIR}/longform_onset.png", dpi=200)
print("wrote", f"{FIGDIR}/longform_onset.png")

# =========================================================== FIG 3: quality ==
fig3, (axl, axr) = plt.subplots(1, 2, figsize=(10, 4.2))

# code validity — one series, so emphasis (no legend needed; the title names it)
m = mean_by("py_parse_frac", "code")
xs = [k for k in KS if m[k] is not None]
ys = [m[k] * 100 for k in xs]
axl.plot(xs, ys, color=CAT["code"], lw=2, marker="o", ms=6, mec=SURFACE, mew=1.5,
         zorder=3, clip_on=False)
for x, y in zip(xs, ys):
    axl.annotate(f"{y:.0f}%", (x, y), textcoords="offset points", xytext=(0, 9),
                 ha="center", fontsize=8, color=INK2)
style_axis(axl, "E · Does the generated Python still parse?",
           "longest valid-Python prefix (% of lines)")
axl.set_ylim(0, 108)

# looped token share — all families
series_panel(axr, "looped_frac", "F · Share of tokens inside a repeated 10-gram",
             "% of generated tokens", scale=100)

fig3.tight_layout()
fig3.savefig(f"{FIGDIR}/longform_quality.png", dpi=200)
print("wrote", f"{FIGDIR}/longform_quality.png")
print(">>> PLOTS DONE")
