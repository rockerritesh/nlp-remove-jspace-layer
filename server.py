#!/usr/bin/env python3
"""
Layer-ablation inference server for Llama-3.1-8B-Instruct.

Runs on a GPU VM. Loads the model once and exposes a /generate endpoint that,
for each prompt, produces a BASELINE generation and an ABLATED generation (one
decoder layer's residual contribution removed) plus quantitative metrics
comparing the two next-token distributions.

Ablation mechanism (A) -- discard-delta wrapper:
    Wrap the target LlamaDecoderLayer so it STILL runs (keeping the KV cache
    consistent during generate()) but returns its INPUT hidden states as output.
    Since the block is residual (x_out = x_in + attn + mlp), returning x_in makes
    the residual stream skip the layer -> the layer is effectively removed.
    Fully reversible (unwrap to restore).

Layer indexing:
    The t-SNE grid is output_hidden_states (33 tensors): L0 = embedding output,
    plot Lk = output of decoder layer index (k-1). The API uses the 0-based
    decoder index (0..31); the response echoes the plot label.

Usage on the VM (env already provisioned):
    python server.py                 # bf16 on CUDA, binds 127.0.0.1:8000
    python server.py --load-4bit     # 4-bit (bitsandbytes) for small GPUs
    python server.py --self-test     # verify ablation logic (only needs torch)
"""
import argparse
import os
import time
from contextlib import contextmanager

import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"

# Globals populated by load_model() in __main__ (read by the endpoints).
tokenizer = None
model = None
terminators = None


# --------------------------------------------------------------------------- #
# Ablation
# --------------------------------------------------------------------------- #
class AblatedDecoderLayer(nn.Module):
    """Wraps a decoder layer: runs it (keeps the KV cache consistent) but returns
    the *input* hidden states, so the residual stream skips this layer."""

    def __init__(self, orig: nn.Module):
        super().__init__()
        self.orig = orig

    def forward(self, hidden_states, *args, **kwargs):
        out = self.orig(hidden_states, *args, **kwargs)
        # Decoder layers return a tuple whose first element is the new hidden
        # states; preserve any trailing elements (attn weights, etc.).
        if isinstance(out, tuple):
            return (hidden_states,) + tuple(out[1:])
        return hidden_states


@contextmanager
def ablate(mdl, layer_idx):
    """Temporarily ablate decoder layer `layer_idx`. `None` = baseline (no-op)."""
    if layer_idx is None:
        yield
        return
    layers = mdl.model.layers
    if not (0 <= layer_idx < len(layers)):
        raise ValueError(f"layer {layer_idx} out of range 0..{len(layers) - 1}")
    orig = layers[layer_idx]
    layers[layer_idx] = AblatedDecoderLayer(orig)
    try:
        yield
    finally:
        layers[layer_idx] = orig


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(load_4bit: bool):
    from dotenv import load_dotenv
    from transformers import AutoModelForCausalLM, AutoTokenizer

    load_dotenv()
    token = os.getenv("HF_TOKEN")

    tok = AutoTokenizer.from_pretrained(MODEL_ID, token=token)

    kwargs = dict(token=token, device_map="cuda")
    if load_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    mdl.eval()

    term = [tok.eos_token_id, tok.convert_tokens_to_ids("<|eot_id|>")]
    term = [t for t in term if t is not None]
    return tok, mdl, term


# --------------------------------------------------------------------------- #
# Generation + metrics
# --------------------------------------------------------------------------- #
def build_inputs(system_prompt: str, prompt: str):
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    return ids.to(model.device)


@torch.no_grad()
def run_generate(input_ids, max_new_tokens: int, temperature: float, layer_idx):
    do_sample = bool(temperature and temperature > 0)
    with ablate(model, layer_idx):
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=0.9 if do_sample else None,
            eos_token_id=terminators,
            pad_token_id=(tokenizer.pad_token_id or tokenizer.eos_token_id),
        )
    continuation = out[0, input_ids.shape[1]:]
    text = tokenizer.decode(continuation, skip_special_tokens=True)
    return text, out


@torch.no_grad()
def compare_metrics(full_ids, prompt_len: int, layer_idx: int):
    """Teacher-forced comparison over the baseline continuation: same context for
    both models, so differences are purely the ablation's effect."""

    def logits_for(li):
        with ablate(model, li):
            return model(full_ids, use_cache=False).logits[0].float()  # [seq, vocab]

    base = logits_for(None)
    abl = logits_for(layer_idx)

    # Positions prompt_len-1 .. seq-2 predict the continuation tokens.
    sl = slice(prompt_len - 1, full_ids.shape[1] - 1)
    lb, la = base[sl], abl[sl]
    if lb.shape[0] == 0:
        return {"kl_mean": 0.0, "top1_agreement": 1.0, "n_positions": 0}

    logp_b = F.log_softmax(lb, dim=-1)
    logp_a = F.log_softmax(la, dim=-1)
    kl = (logp_a.exp() * (logp_a - logp_b)).sum(-1)  # KL(ablated || baseline)
    top1 = (lb.argmax(-1) == la.argmax(-1)).float().mean().item()
    return {
        "kl_mean": round(kl.mean().item(), 4),
        "top1_agreement": round(top1, 4),
        "n_positions": int(lb.shape[0]),
    }


# --------------------------------------------------------------------------- #
# API (web deps imported lazily so --self-test needs only torch)
# --------------------------------------------------------------------------- #
def serve(host: str, port: int):
    import uvicorn
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel

    app = FastAPI(title="Layer Ablation Server")
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
    )

    class GenRequest(BaseModel):
        prompt: str
        layer: int = 14
        max_new_tokens: int = 256
        temperature: float = 0.0
        system_prompt: str = "You are a helpful assistant."

    @app.get("/health")
    def health():
        return {"status": "ok", "loaded": model is not None}

    @app.get("/info")
    def info():
        return {
            "model": MODEL_ID,
            "num_layers": len(model.model.layers),
            "device": str(model.device),
            "dtype": str(model.dtype),
        }

    @app.post("/generate")
    def generate_endpoint(req: GenRequest):
        input_ids = build_inputs(req.system_prompt, req.prompt)
        prompt_len = input_ids.shape[1]

        t0 = time.time()
        base_text, base_full = run_generate(input_ids, req.max_new_tokens, req.temperature, None)
        t1 = time.time()
        abl_text, _ = run_generate(input_ids, req.max_new_tokens, req.temperature, req.layer)
        t2 = time.time()
        metrics = compare_metrics(base_full, prompt_len, req.layer)
        t3 = time.time()

        return {
            "baseline": base_text,
            "ablated": abl_text,
            "layer": req.layer,
            "plot_label": f"L{req.layer + 1}",
            "metrics": metrics,
            "timing": {
                "baseline_s": round(t1 - t0, 2),
                "ablated_s": round(t2 - t1, 2),
                "metrics_s": round(t3 - t2, 2),
            },
        }

    uvicorn.run(app, host=host, port=port)


# --------------------------------------------------------------------------- #
# Self-test (no model load, no GPU, no web deps)
# --------------------------------------------------------------------------- #
def self_test():
    """Verify the ablation wrapper and swap/restore logic with a dummy layer."""

    class DummyLayer(nn.Module):
        def forward(self, hidden_states, *a, **k):
            return (hidden_states + 100.0, "attn_weights")

    x = torch.zeros(1, 3, 4)

    normal = DummyLayer()(x)
    assert torch.allclose(normal[0], x + 100.0), "dummy layer sanity"

    wrapped = AblatedDecoderLayer(DummyLayer())(x)
    assert torch.allclose(wrapped[0], x), "ablated layer must return input unchanged"
    assert wrapped[1] == "attn_weights", "must preserve trailing tuple elements"

    layers = nn.ModuleList([DummyLayer(), DummyLayer(), DummyLayer()])
    fake = type("M", (), {})()
    fake.model = type("M", (), {})()
    fake.model.layers = layers
    with ablate(fake, 1):
        assert isinstance(layers[1], AblatedDecoderLayer), "should be wrapped inside"
        assert isinstance(layers[0], DummyLayer), "other layers untouched"
    assert isinstance(layers[1], DummyLayer), "layer must be restored after context"

    # Out-of-range guard.
    try:
        with ablate(fake, 99):
            pass
        raise AssertionError("expected ValueError for out-of-range layer")
    except ValueError:
        pass

    print("self-test passed ✓")


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--load-4bit", action="store_true", help="4-bit quantization")
    ap.add_argument("--self-test", action="store_true", help="verify ablation logic and exit")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        raise SystemExit(0)

    tokenizer, model, terminators = load_model(args.load_4bit)
    print(f"loaded {MODEL_ID}: {len(model.model.layers)} decoder layers on {model.device}")
    serve(args.host, args.port)
