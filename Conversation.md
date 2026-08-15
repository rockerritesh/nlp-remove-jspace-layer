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

## 2026-08-14/15 — Experiment 9: long-form generation under a 1/2/3-layer cut (405B)

**Question:** a 1–3 layer cut is <2.5% of a 126-layer model and is invisible on next-token
metrics. Does it stay invisible over a LONG roll-out, or does the per-step perturbation
compound? (Earlier smoke: at 128 tokens a 3-layer cut was byte-identical to baseline; at
256 tokens the story diverged — so the effect only exists at length.)

**Design:** `meta-llama/Meta-Llama-3.1-405B` (base, NDIF remote via nnsight), 126 layers.
100 long-form prompts × 5 families (narrative 25 / code 20 / explain 20 / procedure 15 /
reasoning 20), completion-style cues because NDIF pins base models. Conditions k = 0/1/2/3
contiguous layers centred on L63. **Greedy** decoding so every difference is attributable to
the ablation. 1024 tokens forced (see below) ≈ 830 words mean. 400 generations total.

**HEADLINE FINDING — divergence without degradation.** The cut reroutes the trajectory
but does not damage capability:
- 94–98% of ablated runs diverge from baseline; median first difference at token **101 (k=1)
  / 58 (k=2) / 53 (k=3)** — i.e. ~5–10% of the way in. They never reconverge: only ~0.60
  unigram F1 with baseline over the remaining ~900 tokens. One layer of 126 rewrites most
  of a 900-word answer.
- More removal diverges earlier (178→144→124 mean tokens), significant only k1 vs k3
  (paired permutation p=0.042).
- **BUT no quality metric differs from baseline** (paired sign-flip permutation, n=100,
  20k iters): distinct-3, looped-token share, TTR, mean sentence length, loop onset — all
  p>0.05. Python parse validity 100/100/100/96.4% for k=0/1/2/3.
- So divergence ≠ damage. KL/top-1 from the earlier experiments measure *departure from the
  original distribution*, not loss of capability — this run separates the two directly.
- Damage barely scales with k: k=1 and k=3 give nearly the same F1 (~0.60). Dominant effect
  is *any* perturbation of the middle band, not its size.

**Two methodology bugs caught on the first real batch (both would have produced wrong
findings — the partial numbers reported before the fix were artifacts):**
1. **Pad tokens scored as model output.** Batched generation left-pads every sequence to the
   batch max with `128001`; one baseline row was 909 pads of 1024. Repetition metrics were
   measuring pad runs ("baseline loops at token 302, 75% repeated" was ~all artifact).
   Analysis now trims each sequence at its first EOS.
2. **The prompts didn't produce long output on a BASE model.** Real tokens before EOS:
   reasoning ~51, procedure ~200, explain ~385. Only narrative ran long. Fixed with
   `min_new_tokens == max_new_tokens`, which blocks the EOS logit for the whole roll-out —
   verified remotely (51 → full 200, zero EOS). Natural-length pilot kept as
   `generations_naturallen_pilot.jsonl`.
3. Minor: `prefix_frac` divided by the *ablated* length, so an early-stopping run that matched
   the baseline scored "100% identical". Now divided by baseline length; early stop is its
   own metric.

**NDIF/tooling lessons (durable):**
- **Batching is free.** Wall-clock scales with token count, NOT batch size (4 prompts @128tok
  == 1 prompt @128tok == 33s). Batch of 10 @1024tok = ~305s. This turned a ~30h serial study
  into ~3.4h. Length-sort prompts before batching to minimise left-padding.
- `min_new_tokens` DOES pass through nnsight to the remote model.
- **NDIF can wedge a job in RUNNING forever** — it never raises, so a plain retry loop blocks
  indefinitely (observed one job stuck 6h against a 304s norm). Added a `SIGALRM` per-job
  wall-clock budget. **Caveat: the alarm only rescues a CLEAN process** — after the first
  alarm tore an exception out of nnsight's generate context, the in-process retry wedged again
  and the alarm did NOT fire (blocked in a C-level wait that never yields to the handler).
  Use `--retries 1` and re-run the script; resume by (prompt_id, k) makes that cheap.
- Transient `socketio ConnectionError` bursts happen; a minimal probe confirms NDIF health.
- Batch-outer / condition-inner loop order so partial results are analysable immediately.

**Artifacts:** `experiments/{longform_prompts,nnsight_405b_longform,analyze_longform,
plot_longform}.py`, `results/longform405b/{generations.jsonl,metrics.csv,summary.json}`,
`blog/figures/longform_{metrics,onset,quality}.png`. run.log gitignored (14 MB of spinners).
