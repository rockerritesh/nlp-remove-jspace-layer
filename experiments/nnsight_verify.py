#!/usr/bin/env python3
"""Lock down nnsight 0.7 (transformers 4.57) ablation syntax on local GPT-2."""
import torch
import torch.nn.functional as F
from nnsight import LanguageModel

m = LanguageModel("openai-community/gpt2", device_map="cpu", dispatch=True)
def val(p): return p.value if hasattr(p, "value") else p
prompt = "The Eiffel Tower is in the city of"

with m.trace(prompt):
    out5 = m.transformer.h[5].output.save()
    logits = m.output.logits.save()
out5, logits = val(out5), val(logits)
is_tuple = isinstance(out5, (tuple, list))
print("h5.output tuple:", is_tuple, "|", (tuple(out5[0].shape) if is_tuple else tuple(out5.shape)))
base = logits[0, -1, :]
print("baseline top:", repr(m.tokenizer.decode(base.argmax())))

def kl(a, b):
    la = F.log_softmax(a.float(), -1); lb = F.log_softmax(b.float(), -1)
    return (la.exp() * (la - lb)).sum().item()

ok = None
for name in ["out[0][:]=input", "out[:]=input"]:
    try:
        with m.trace(prompt):
            L = m.transformer.h[5]
            if name == "out[0][:]=input":
                L.output[0][:] = L.input
            else:
                L.output[:] = L.input
            ab = m.output.logits.save()
        a = val(ab)[0, -1, :]
        print(f"[{name}] OK  KL(abl||base)={kl(a, base):.3f}  top={m.tokenizer.decode(a.argmax())!r}")
        ok = name
        break
    except Exception as e:
        print(f"[{name}] FAIL  {type(e).__name__}: {str(e)[:130]}")
print(">>> WORKING_ACCESSOR =", ok)
print(">>> VERIFY DONE")
