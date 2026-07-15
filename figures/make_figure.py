#!/usr/bin/env python3
"""
Research-grade figure for the layer-ablation experiment.

Panel A: token x layer residual-stream grid of Llama-3.1-8B, with the three
         representation regimes from the layer-wise t-SNE (surface / concept-
         forming / output-oriented) and the ablated middle layer marked.
Panel B: the discard-delta ablation mechanism (normal vs ablated decoder block).
Panel C: the MEASURED effect on generation (real numbers from the T4 run).

Outputs: figures/architecture.png (300 dpi) and figures/architecture.pdf (vector).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# ----------------------------------------------------------------- style
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "mathtext.fontset": "cm",
    "axes.linewidth": 0.8,
})
INK      = "#23272e"
MUTED    = "#6b7480"
BLOCK_FC = "#f3f5f8"
BLOCK_EC = "#b6bfca"
RESID    = "#3b414a"
ATTN     = "#98a2b0"
SURFACE  = "#eaf0ea"; SURFACE_E = "#b9cbb9"
CONCEPT  = "#fbe7c8"; CONCEPT_E = "#d79a3c"
OUTPUT   = "#e4edf8"; OUTPUT_E  = "#6f93c4"
CRIMSON  = "#c0392b"
CALLOUT  = "#2f6fb0"
GOOD     = "#3f8f5f"


def rbox(ax, cx, cy, w, h, fc=BLOCK_FC, ec=BLOCK_EC, lw=1.1, ls="-", z=3, alpha=1.0):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.004,rounding_size=0.012",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls, zorder=z, alpha=alpha))


def arrow(ax, p0, p1, color=RESID, lw=1.6, rad=0.0, ms=10, z=4, alpha=1.0, ls="-"):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=z, alpha=alpha, linestyle=ls,
        shrinkA=0, shrinkB=0))


# ================================================================= figure
fig = plt.figure(figsize=(13.6, 9.4))
gs = fig.add_gridspec(2, 2, height_ratios=[1.62, 1.0],
                      width_ratios=[1.32, 1.0], hspace=0.20, wspace=0.14,
                      left=0.035, right=0.975, top=0.90, bottom=0.055)
axA = fig.add_subplot(gs[0, :])
axB = fig.add_subplot(gs[1, 0])
axC = fig.add_subplot(gs[1, 1])
for ax in (axA, axB, axC):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

fig.suptitle("Removing a meaning-forming layer from Llama-3.1-8B-Instruct",
             x=0.037, y=0.965, ha="left", fontsize=18, fontweight="bold", color=INK)
fig.text(0.037, 0.925,
         "Ablate one middle decoder layer (residual stream passes straight through), regenerate, and measure the shift.",
         ha="left", fontsize=11.5, color=MUTED)

# ----------------------------------------------------------------- Panel A
ABL = 14                                   # ablated decoder index
cols = [0, 3, 6, 8, 10, 12, 14, 16, 18, 21, 25, 28, 31]
tokens = ["<bos>", "The", "sky", "is", "blue"]
x0, x1 = 0.075, 0.80
def X(d): return x0 + (d / 31.0) * (x1 - x0)
ys = [0.78, 0.665, 0.55, 0.435, 0.32]
BW, BH = 0.026, 0.052

# regime bands
def band(d_lo, d_hi, fc, ec):
    xa, xb = X(d_lo), X(d_hi)
    axA.add_patch(Rectangle((xa, 0.275), xb - xa, 0.565, facecolor=fc,
                            edgecolor=ec, lw=1.1, zorder=0, alpha=0.85))
band(-0.55, 5.5, SURFACE, SURFACE_E)
band(5.5, 16.5, CONCEPT, CONCEPT_E)
band(16.5, 31.6, OUTPUT, OUTPUT_E)

# regime labels
axA.text(X(2.4), 0.945, "surface / token\nfeatures", ha="center", va="center",
         fontsize=10.2, color="#4d6b4d", linespacing=1.05)
axA.text(X(11), 0.967, "concept-forming layers", ha="center", va="center",
         fontsize=11.8, color="#9a6a1e", fontweight="bold")
axA.text(X(11), 0.929, "t-SNE clusters intermix — semantic processing",
         ha="center", va="center", fontsize=9.6, color="#9a6a1e", style="italic")
axA.text(X(24), 0.945, "output-oriented\nrepresentations", ha="center", va="center",
         fontsize=10.2, color="#3f5f8c", linespacing=1.05)

# token labels
for y, t in zip(ys, tokens):
    axA.text(0.052, y, t, ha="right", va="center", fontsize=10.5, color=INK)

# residual streams + blocks
for y in ys:
    arrow(axA, (0.058, y), (X(cols[0]) - BW / 2, y), color=RESID, lw=1.8, ms=9)
    for i, d in enumerate(cols):
        if i < len(cols) - 1:
            bold = (d == ABL or cols[i + 1] == ABL)
            arrow(axA, (X(d) + BW / 2, y), (X(cols[i + 1]) - BW / 2, y),
                  color=(CRIMSON if bold else RESID), lw=(2.4 if bold else 1.6), ms=9,
                  z=(5 if bold else 4))
        if d == ABL:
            rbox(axA, X(d), y, BW, BH, fc="#e9ecf0", ec=CRIMSON, lw=1.4, ls=(0, (3, 2)), z=3)
            axA.plot([X(d) - BW * 0.34, X(d) + BW * 0.34], [y - BH * 0.32, y + BH * 0.32],
                     color=CRIMSON, lw=1.3, zorder=6)
            axA.plot([X(d) - BW * 0.34, X(d) + BW * 0.34], [y + BH * 0.32, y - BH * 0.32],
                     color=CRIMSON, lw=1.3, zorder=6)
        else:
            rbox(axA, X(d), y, BW, BH)

# attention flow arrows converging on the generation (bottom) lane
flows = [(0, 6, 8), (1, 8, 10), (2, 10, 14), (0, 14, 16), (1, 16, 18), (3, 18, 21)]
for lane, da, db in flows:
    arrow(axA, (X(da), ys[lane] - BH / 2), (X(db), ys[-1] + BH / 2),
          color=ATTN, lw=1.5, rad=-0.18, ms=9, z=2, alpha=0.9)

# output head
arrow(axA, (X(cols[-1]) + BW / 2, ys[-1]), (0.895, ys[-1]), color=RESID, lw=1.8, ms=10)
rbox(axA, 0.93, ys[-1], 0.055, 0.075, fc="#dfe7f2", ec=OUTPUT_E, lw=1.3)
axA.text(0.93, ys[-1], "$p(\\cdot)$", ha="center", va="center", fontsize=12, color=INK)
axA.text(0.93, ys[-1] - 0.06, "next\ntoken", ha="center", va="top", fontsize=8.6,
         color=MUTED, linespacing=1.0)

# ablated-layer annotation (small, inside the band strip above the top lane)
axA.text(X(ABL), 0.822, "removed", ha="center", va="center", fontsize=9.2,
         color=CRIMSON, fontweight="bold")

# blue callout (matches the reference style)
axA.add_patch(FancyArrowPatch((0.905, 0.68), (X(ABL) + BW / 2, ys[2]),
              arrowstyle="-|>", mutation_scale=14, lw=2.0, color=CALLOUT,
              connectionstyle="arc3,rad=0.28", zorder=7))
axA.text(0.998, 0.76, "ablate one meaning-\nforming layer\n$\\ell = 14$  (t-SNE L15)\n$x_\\ell = x_{\\ell-1}$",
         ha="right", va="top", fontsize=9.8, color=CALLOUT, linespacing=1.35)

# depth axis
for d in [0, 8, 16, 24, 31]:
    axA.text(X(d), 0.252, str(d), ha="center", va="top", fontsize=8.6, color=MUTED)
axA.text(X(15.5), 0.222, "decoder layer index  $\\ell$  (0–31)   —   t-SNE plot $L_k$ = output of layer $\\ell=k-1$",
         ha="center", va="top", fontsize=9.2, color=MUTED)
axA.text(0.037, 0.985, "A", transform=axA.transAxes, fontsize=14, fontweight="bold", color=INK)

# ----------------------------------------------------------------- Panel B
axB.text(0.0, 0.96, "B   Ablation mechanism: discard-delta wrapper", ha="left",
         va="top", fontsize=12.5, fontweight="bold", color=INK)

def block_diagram(ax, x_left, title, ablated):
    yc = 0.44
    xin, xproc, xsum, xout = x_left + 0.03, x_left + 0.16, x_left + 0.30, x_left + 0.42
    ax.text(x_left + 0.22, 0.80, title, ha="center", fontsize=10.6,
            color=(CRIMSON if ablated else INK), fontweight="bold")
    ax.text(xin, yc, "$x_{\\ell-1}$", ha="center", va="center", fontsize=12, color=INK)
    # processing block
    rbox(ax, xproc, yc, 0.11, 0.15,
         fc=("#efeff1" if ablated else "#eaf1ff"),
         ec=(CRIMSON if ablated else OUTPUT_E),
         lw=1.3, ls=((0, (3, 2)) if ablated else "-"))
    ax.text(xproc, yc + 0.005, "Attn\n+ MLP", ha="center", va="center",
            fontsize=8.8, color=(MUTED if ablated else INK), linespacing=1.0)
    # sum node
    ax.add_patch(plt.Circle((xsum, yc), 0.028, fc="white", ec=INK, lw=1.2, zorder=4))
    ax.text(xsum, yc, "+", ha="center", va="center", fontsize=12, color=INK, zorder=5)
    # input -> block -> sum -> out
    arrow(ax, (xin + 0.03, yc), (xproc - 0.058, yc), lw=1.6, ms=9)
    if ablated:
        arrow(ax, (xproc + 0.058, yc), (xsum - 0.03, yc), color=MUTED, lw=1.3,
              ms=8, ls=(0, (2, 2)))
        ax.text(xproc + 0.075, yc + 0.13, "$\\Delta_\\ell$ computed,\ndiscarded",
                ha="center", va="bottom", fontsize=8.2, color=MUTED, linespacing=1.0)
        ax.text(xproc, yc - 0.135, "(still runs -> KV-cache\nstays consistent)",
                ha="center", va="top", fontsize=7.8, color=MUTED, linespacing=1.0)
    else:
        arrow(ax, (xproc + 0.058, yc), (xsum - 0.03, yc), lw=1.6, ms=9)
    arrow(ax, (xsum + 0.03, yc), (xout - 0.02, yc),
          color=(CRIMSON if ablated else RESID), lw=(2.4 if ablated else 1.6), ms=9)
    ax.text(xout + 0.02, yc, "$x_\\ell$", ha="center", va="center", fontsize=12, color=INK)
    # residual skip arc
    arrow(ax, (xin, yc - 0.02), (xsum, yc - 0.02),
          color=(CRIMSON if ablated else RESID), lw=(2.4 if ablated else 1.6),
          rad=-0.55, ms=9)
    # equation
    eq = "$x_\\ell = x_{\\ell-1}$" if ablated else "$x_\\ell = x_{\\ell-1} + \\Delta_\\ell$"
    ax.text(x_left + 0.22, 0.10, eq, ha="center", va="center", fontsize=11.5,
            color=(CRIMSON if ablated else INK))

block_diagram(axB, 0.02, "normal layer", ablated=False)
axB.plot([0.5, 0.5], [0.12, 0.74], color="#d5dae1", lw=1.0)
block_diagram(axB, 0.52, "ablated layer", ablated=True)

# ----------------------------------------------------------------- Panel C
axC.text(0.0, 0.96, "C   Measured effect on generation", ha="left", va="top",
         fontsize=12.5, fontweight="bold", color=INK)
axC.text(0.0, 0.86, "Ablating $\\ell=14$ (t-SNE L15) · Llama-3.1-8B-Instruct, 4-bit on a T4",
         ha="left", va="top", fontsize=9.4, color=MUTED)

# top-1 agreement bars (real measurements)
data = [("“why is the sky blue?”", 0.88, 0.15),
        ("“name three primary colors”", 1.00, 0.04)]
bx0, bw = 0.05, 0.62
for i, (label, agree, kl) in enumerate(data):
    y = 0.62 - i * 0.22
    axC.text(bx0, y + 0.075, label, ha="left", va="bottom", fontsize=9.6, color=INK)
    axC.add_patch(Rectangle((bx0, y), bw, 0.05, facecolor="#eceff3",
                            edgecolor=BLOCK_EC, lw=0.8))
    axC.add_patch(Rectangle((bx0, y), bw * agree, 0.05, facecolor=GOOD,
                            edgecolor="none"))
    axC.text(bx0 + bw + 0.02, y + 0.025, f"{agree*100:.0f}%  top-1",
             ha="left", va="center", fontsize=9.2, color=INK)
    axC.text(bx0 + bw * agree - 0.01, y + 0.025, f"KL {kl:.2f}",
             ha="right", va="center", fontsize=8.0, color="white", fontweight="bold")

axC.text(0.0, 0.15,
         "Output stays coherent and on-topic — a single middle\n"
         "layer is individually robust (the residual stream routes\n"
         "around it). Early / late layers break generation instead.",
         ha="left", va="top", fontsize=9.2, color=INK, linespacing=1.25)

# footer: system
fig.text(0.037, 0.02,
         "System:  Mac (ui.html)  --SSH tunnel :8000-->  GCloud T4 VM  ·  FastAPI  ·  Llama-3.1-8B-Instruct (4-bit)   "
         "|   metric = mean KL(ablated || baseline) & top-1 agreement over the baseline continuation",
         ha="left", fontsize=8.2, color=MUTED)

os.makedirs(os.path.dirname(__file__) or ".", exist_ok=True)
out = os.path.join(os.path.dirname(__file__), "architecture")
fig.savefig(out + ".png", dpi=300, facecolor="white", bbox_inches="tight")
fig.savefig(out + ".pdf", facecolor="white", bbox_inches="tight")
print("wrote", out + ".png", "and", out + ".pdf")
