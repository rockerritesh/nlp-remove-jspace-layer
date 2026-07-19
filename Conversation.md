# Conversation History — nlp-remove-jspace-layer

## 2026-07-15 — Project kickoff: layer-ablation inference experiment

**Goal:** Given layer-wise t-SNE plots of Llama-3.1-8B hidden states, identify the
"meaning-forming" middle layers (plots L6–L17, where category clusters blur), remove
one such decoder layer, and observe the effect on generation.

**Decisions (brainstormed):**
- Compute: model runs on the **GCloud VM (embedding project, GPU)**; Mac is for UI + plotting only.
- Model: `meta-llama/Llama-3.1-8B-Instruct`.
- Ablation scope: **one specific layer** at a time (UI lets you pick any of 0–31 live), default `decoder[14]` = plot L15.
- Metrics: **qualitative** (side-by-side text) + **quantitative** (KL divergence, top-1 agreement).
- Mechanism (A): discard-delta wrapper — layer still runs (KV cache stays consistent) but returns its input, so the residual stream skips it. Reversible.
- UI: single `ui.html` on the Mac (no build), fetches the VM API through an SSH tunnel; side-by-side baseline vs ablated.
- Env: `uv` on the Mac (`client.py` deps); VM env already exists.

**Built:** `server.py` (VM), `ui.html` (Mac), `client.py` (Mac CLI + sweep plot),
`pyproject.toml`, `requirements-vm.txt`, `README.md` (+ Mermaid architecture diagrams),
design spec in `docs/superpowers/specs/`.

**Indexing note:** t-SNE `Lk` = `output_hidden_states[k]`; L0 = embeddings, so plot Lk = `model.model.layers[k-1]`.

**Open / next:** run `server.py` on the VM, verify end-to-end through the tunnel with a
few prompts; then explore which middle layer's removal changes generation most.

## 2026-07-15 (later) — Deployed & verified end-to-end on the VM

- **VM:** `embed-instance-20260311-044232-t4-01` (project `embeddingserver-489904`, zone
  `us-central1-b`), **NVIDIA T4 16 GB**, `n1-highmem-4`. Was TERMINATED; started it.
- Env: system Python had nothing; the real env is **`~/.venv`** (uv project) with torch
  2.10.0+cu128. Installed transformers(5.13.1)/accelerate/fastapi/uvicorn/python-dotenv/
  bitsandbytes(0.49.2) into it. Code lives in `~/layer-ablation/` on the VM.
- **Disk was 100% full** (99 G). User chose to delete re-downloadable HF caches
  (IndicBERT/LaBSE/glotlid ≈ 5.5 G) to free space; Llama download then resumed & finished.
- Runs **4-bit** (`--load-4bit`) because T4 can't hold bf16 8B. Compute dtype auto-selected.
- **transformers 5.x fix:** `apply_chat_template` returns a `BatchEncoding` (not a dict);
  `build_inputs` now keys off `torch.Tensor`. Committed as `a342c23`.
- **Verified end-to-end:** "why is the sky blue?" ablating L15 → coherent, KL 0.15,
  top-1 88%. "three primary colors" → identical, KL 0.04, top-1 100%. Both via the tunnel.
- **SSH to the VM was intermittently flaky** from this machine (long sessions dropped;
  orphaned ssh procs tripped sshd MaxStartups). Workaround: short SSH calls only + a
  detached VM script (`run_smoke.sh`) writing a result file. `run_smoke.sh` on VM only.
- Tunnel: `gcloud compute ssh <vm> --project embeddingserver-489904 --zone us-central1-b -- -N -L 8000:localhost:8000`.
- **Reminder:** the T4 VM is billing — stop it when done:
  `gcloud compute instances stop embed-instance-20260311-044232-t4-01 --project embeddingserver-489904 --zone us-central1-b`.

## 2026-07-15 (cont.) — Multi-layer UI, figures, blog, 100-query analysis, PDF→Telegram

- **Multi-layer ablation** (`0c971b4`): `ablate()` takes a set of layers; `/generate` accepts
  `layers[]`; UI replaced the single dropdown with a 0–31 toggle grid (regime-coloured) +
  presets (Concept L6–L17 / Early / Late). Verified: 1 layer (L15) → KL 0.15, top-1 88%;
  whole concept band (12 layers) → KL 15.6, top-1 20%, output collapses to repetition.
- **Architecture figure** (`3575392`): `figures/make_figure.py` → `figures/architecture.png`
  (+ PDF), publication-grade token×layer diagram; embedded in the README.
- **Experiments + blog** (`75db88a`): `experiments/run_experiments.py` on a 4-class
  20-Newsgroups set → per-layer UMAP grid, decodability(kNN) vs silhouette, single-layer
  ablation sweep, cumulative removal. `blog/README.md` written with KL-vs-entropy explainer,
  step-by-step walkthrough, and the link to Anthropic's 2026 J-space / global-workspace paper
  (Gurnee, Sofroniew, … Lindsey) + citations. Finding: info decodable from L2 (~0.9 kNN) but
  visually entangled mid-network → re-separates late; single-layer ablation is an **early
  cliff** (only L0/L1 catastrophic).
- **100-query analysis** (Experiment 5, `9d9598b`): `experiments/run_queries100.py`, 100
  prompts × 5 capability types. Exports `metrics_100.json`, `per_query_results.json/.csv`, and
  3 plots. Per-category finding: **single-step recall most robust, creative/abstract breaks
  first** — the direction the workspace paper predicts. (Run took ~60 min on the T4.)
- **KEY TOOLING LESSON:** the "flaky SSH" was mostly a shell bug — calling `gcloud ssh` via a
  shell variable (`$G "cmd"`) silently fails here; **inline** gcloud calls work reliably.
  Fetch VM files with inline `gcloud … --command "base64 ~/path"` piped to local `base64 -D`.
- **PDF → Telegram:** rendered `blog/README.md` → `blog/layer-ablation-blog.pdf`
  (pandoc + weasyprint, images embedded, 1.88 MB) and sent via the bot in
  `research/agents/control-claude-with-telegram/.env` (sendDocument, msg 1319).
- **VM STOPPED** (status TERMINATED) — GPU billing halted. Boot disk keeps a small storage
  charge; all code/results persist on the VM at `~/layer-ablation/`.
- **Status: all deliverables complete and committed** (latest `9d9598b`). Untracked/local:
  `blog/layer-ablation-blog.pdf` (build artifact), `Conversation.md` (now un-ignored),
  `.claude/`, `results/manual/`.

## 2026-07-16 — Private repo, nnsight/NDIF remote, larger models (405B), comparative figure

- **Private GitHub repo:** pushed everything to `github.com/rockerritesh/nlp-remove-jspace-layer`
  (PRIVATE, personal `nl-mode`/rockerritesh). `.env` never tracked (keys safe). Latest `c890ff9`.
- **nnsight + NDIF remote (Experiment 6):** re-ran the ablation on FULL-PRECISION base
  `Meta-Llama-3.1-8B` via `remote=True` (model on NDIF servers, laptop only orchestrates — no
  local GPU/VM). Reproduces the early cliff + band collapse → findings aren't a 4-bit/Instruct
  artifact. Local uv `.venv` (py3.12) with nnsight 0.7.
- **Tooling gotchas (durable):** nnsight 0.7 needs `transformers` in **[4.48, 5.0)** (5.x →
  MissedProviderError; use 4.57). Llama layer `.output` is a **tensor** (not tuple) here.
  `.input`/`.inputs` fail remotely — ablate by **rebind**: `layers[L].output = layers[L-1].output`
  (embeddings for L0). In-place `[:]=` is NOT captured remotely — must rebind. nnsight traces
  **parse block source**, so run from a real `.py` file (heredoc/`-c` → "could not get source").
  Generation interventions sit directly inside `with model.generate(...)` and apply to all
  tokens; read output via `model.generator.output.save()`. NDIF **free tier runs only PINNED
  base models** (8B/70B/405B), not Instruct.
- **Large-model generation (Experiment 7):** real text from **70B** with a middle band removed.
  70B is robust — 25% removal still correct/coherent; only ~50% starts to wobble (fluent).
- **Cross-model gallery (§6e, Experiment 8):** same relativity prompt through 8B/70B/405B at
  0/25/50%. All stay fluent even at 50% (artifacts/drift, not gibberish). Reconciled: the
  dramatic §6 collapse was Instruct + concept band; KL/top-1 measure divergence from the
  original distribution, not loss of fluency.
- **Comparative scale figure (§6f):** `scale_compare.png` — next-token top-1 & KL vs %-removed,
  one curve per model. HONEST finding: on the strict next-token metric all sizes collapse by
  ~30-40% removal; larger models only hold modestly better at ≤20%, and 405B diverges MOST at
  40-50%. Free-generation fluency (gallery) is far more forgiving than this metric implies.
- **Blog now spans 8 experiments**; PDF recompiled (2.32 MB, all figures embedded) and sent to
  Telegram twice (msg 1319, 1323) via the control-claude-with-telegram bot. Everything pushed;
  working tree clean.
- Scripts added: `experiments/nnsight_verify.py`, `nnsight_remote_smoke.py`,
  `nnsight_remote_sweep.py`, `nnsight_remote_generate.py`, `nnsight_gallery.py`,
  `nnsight_scale_compare.py`. Reusable: `nnsight_remote_generate.py <model> <max_tokens> <band_frac>`.
