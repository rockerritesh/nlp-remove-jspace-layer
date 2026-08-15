#!/usr/bin/env python3
"""
Experiment 9 — Long-form generation under a 1/2/3-layer removal on Llama-3.1-405B.

Question: a 1-3 layer cut is <2.5% of a 126-layer model and is invisible on the
next-token metric. Does it stay invisible over a LONG autoregressive roll-out, or
does the tiny per-step perturbation compound into visible drift, repetition or
broken structure?

Design
  model      meta-llama/Meta-Llama-3.1-405B  (base; NDIF pins base 8B/70B/405B)
  conditions removed k in {0, 1, 2, 3} contiguous layers centred on the midpoint (L63)
  prompts    100 long-form prompts x 5 capability families (longform_prompts.py)
  decoding   GREEDY (do_sample=False) so every difference is caused by the ablation
  length     --max-tokens (default 1024 ~ 750 words)

Efficiency: NDIF wall-clock scales with token count, NOT batch size (measured:
4 prompts @128 tok == 1 prompt @128 tok == ~33s), so prompts are batched. Prompts
are length-sorted before batching to minimise left-padding.

Output: results/longform405b/generations.jsonl (one row per prompt x condition,
written incrementally). Re-running resumes: rows already present are skipped.

Usage:
  python experiments/nnsight_405b_longform.py                 # full run
  python experiments/nnsight_405b_longform.py --limit 4 --max-tokens 128   # dry run
"""
import os, sys, json, time, signal, argparse, traceback
from dotenv import load_dotenv

ROOT = "/Users/sumityadav/Documents/research/nlp-remove-jspace-layer"
sys.path.insert(0, f"{ROOT}/experiments")
from longform_prompts import PROMPTS

load_dotenv(f"{ROOT}/.env")
from nnsight import LanguageModel, CONFIG
CONFIG.set_default_api_key(os.environ["NDIF_API_KEY"])

def val(p): return p.value if hasattr(p, "value") else p

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="meta-llama/Meta-Llama-3.1-405B")
ap.add_argument("--max-tokens", type=int, default=1024)
ap.add_argument("--min-tokens", type=int, default=-1,
                help="floor on generated tokens (default: == max-tokens, which "
                     "suppresses EOS entirely and forces a full-length roll-out). "
                     "Set 0 to allow natural early stopping.")
ap.add_argument("--batch-size", type=int, default=10)
ap.add_argument("--conditions", default="0,1,2,3")
ap.add_argument("--limit", type=int, default=0, help="use only the first N prompts (dry run)")
ap.add_argument("--out", default=f"{ROOT}/results/longform405b/generations.jsonl")
ap.add_argument("--retries", type=int, default=3)
ap.add_argument("--job-timeout", type=int, default=900,
                help="seconds before a remote job is abandoned and retried. NDIF "
                     "can wedge a job in RUNNING forever; it never raises, so "
                     "without this the run blocks indefinitely (observed: one job "
                     "stuck 49min against a 304s norm).")
args = ap.parse_args()

CONDITIONS = [int(c) for c in args.conditions.split(",")]
# Base Llama emits EOS early on most of these prompts (reasoning stopped at ~51
# tokens), which is the opposite of the long-roll-out regime under test. Holding
# min == max blocks the EOS logit for the whole roll-out, so every prompt in every
# condition produces exactly max_tokens of real output and lengths are comparable.
MIN_TOKENS = args.max_tokens if args.min_tokens < 0 else args.min_tokens
os.makedirs(os.path.dirname(args.out), exist_ok=True)

# ---- resume: which (prompt_id, removed) pairs do we already have? -------------
done = set()
if os.path.exists(args.out):
    for line in open(args.out):
        try:
            r = json.loads(line)
            done.add((r["prompt_id"], r["removed"]))
        except Exception:
            pass
print(f"resume: {len(done)} rows already present in {args.out}", flush=True)

# ---- model -------------------------------------------------------------------
model = LanguageModel(args.model)
tok = model.tokenizer
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
NL = model.config.num_hidden_layers
MID = NL // 2
print(f"model={args.model} layers={NL} midpoint={MID} "
      f"conditions={CONDITIONS} max_tokens={args.max_tokens} batch={args.batch_size}",
      flush=True)

class JobTimeout(Exception):
    pass


def _on_alarm(signum, frame):
    raise JobTimeout(f"remote job exceeded {args.job_timeout}s")


signal.signal(signal.SIGALRM, _on_alarm)


def band_for(k):
    """k contiguous layers centred on the network midpoint."""
    if k == 0:
        return []
    start = MID - k // 2
    return list(range(start, start + k))

# ---- batching: length-sorted so left-padding is minimal -----------------------
prompts = PROMPTS[:args.limit] if args.limit else PROMPTS
for p in prompts:
    p["_ntok"] = len(tok(p["text"])["input_ids"])
ordered = sorted(prompts, key=lambda p: p["_ntok"])
BATCHES = [ordered[i:i + args.batch_size] for i in range(0, len(ordered), args.batch_size)]
print(f"{len(prompts)} prompts -> {len(BATCHES)} batches "
      f"x {len(CONDITIONS)} conditions = {len(BATCHES)*len(CONDITIONS)} remote jobs", flush=True)

fout = open(args.out, "a")
t_start = time.time()
job = 0
total_jobs = len(BATCHES) * len(CONDITIONS)

# Batch-outer / condition-inner: every batch finishes all 4 conditions before the
# next batch starts, so partial results are already analysable (a prompt is only
# useful once its k=0 baseline AND its ablated runs exist).
for bi, batch in enumerate(BATCHES):
    for k in CONDITIONS:
        BAND = band_for(k)
        pending = [p for p in batch if (p["id"], k) not in done]
        job += 1
        if not pending:
            print(f"[{job}/{total_jobs}] k={k} batch {bi} — all present, skip", flush=True)
            continue

        texts = [p["text"] for p in pending]
        in_len = tok(texts, return_tensors="pt", padding=True)["input_ids"].shape[1]

        for attempt in range(1, args.retries + 1):
            try:
                t0 = time.time()
                signal.alarm(args.job_timeout)
                try:
                    with model.generate(texts, max_new_tokens=args.max_tokens,
                                        min_new_tokens=MIN_TOKENS,
                                        do_sample=False, remote=True) as g:
                        if BAND:
                            entry = model.model.layers[BAND[0] - 1].output
                            for L in BAND:
                                model.model.layers[L].output = entry
                        o = model.generator.output.save()
                    out = val(o)
                finally:
                    signal.alarm(0)
                dt = time.time() - t0

                for row, p in zip(out, pending):
                    gen_ids = [int(t) for t in row[in_len:]]
                    text = tok.decode(gen_ids, skip_special_tokens=True)
                    fout.write(json.dumps({
                        "prompt_id": p["id"], "category": p["category"],
                        "prompt": p["text"], "removed": k, "band": BAND,
                        "model": args.model, "layers": NL,
                        "max_new_tokens": args.max_tokens,
                        "min_new_tokens": MIN_TOKENS,
                        "gen_ids": gen_ids, "text": text,
                    }) + "\n")
                    done.add((p["id"], k))
                fout.flush()

                el = time.time() - t_start
                eta = el / job * (total_jobs - job)
                print(f"[{job}/{total_jobs}] k={k} band={BAND} batch {bi} "
                      f"n={len(pending)} in_len={in_len} — {dt:.0f}s "
                      f"(elapsed {el/60:.0f}m, eta {eta/60:.0f}m)", flush=True)
                break
            except Exception as e:
                print(f"  !! k={k} batch {bi} attempt {attempt}/{args.retries} failed: "
                      f"{type(e).__name__}: {str(e)[:200]}", flush=True)
                if attempt == args.retries:
                    traceback.print_exc()
                else:
                    time.sleep(20 * attempt)

fout.close()
print(f"\n>>> wrote {args.out}  ({time.time()-t_start:.0f}s total)", flush=True)
print(">>> LONGFORM DONE", flush=True)
