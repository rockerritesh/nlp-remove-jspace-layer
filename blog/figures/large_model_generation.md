# Large-model generation under middle-band ablation (Llama-3.1-70B, base, via NDIF remote)

Real generated text (greedy, 40 new tokens) with a contiguous middle band of decoder
layers made identity via nnsight, on the **full 70B** model running on NDIF (no local GPU).

## Prompt: "The theory of relativity states that"

**Baseline (nothing removed):**
> the speed of light is a constant for all observers, regardless of their relative speeds. This means that if you were traveling at a speed close to the speed of light, you would still measure the speed…

**Remove 25% (layers 30–49, 20 of 80):** *still correct & coherent*
> space and time are relative concepts rather than absolute concepts. The relativity of time is shown by the fact that time passes at different rates for different observers. The faster you move, the slower time passes…

**Remove 50% (layers 20–59, 40 of 80):** *degrades — repetitive, loses the physics*
> The universe is made up of space and time. The universe is a vast expanse of space and time. The universe is made up of matter and energy. The universe is made up of stars,

## Prompt: "The capital of France is"

**Baseline:** Paris. The capital of Italy is Rome. The capital of Germany is Berlin…
**Remove 25%:** the city of Paris, which is also the largest city in France… situated on the Seine River. *(correct)*
**Remove 50%:** a city of light, love, and luxury… Paris is one of the most popular tourist destinations… *(drifts to brochure prose but keeps "Paris")*

## Prompt: "Step by step, to bake bread you first"

**Baseline:** need to make a dough… flour, water, yeast, and salt… mix… until smooth, elastic…
**Remove 25%:** prepare the dough. Then wait for it to rise. Finally, bake… *(fine)*
**Remove 50%:** A. mix the ingredients. B. bake the dough. C. knead… Answer: A *(collapses into a stray multiple-choice artifact)*

## Takeaway

8B collapsed when ~37% of its layers were removed; **70B is untouched at 25% and only
starts to wobble at 50%.** Larger models carry far more redundant middle layers — the
"workspace" is more distributed and robust at scale. (Base model, greedy decoding, 4 prompts,
band centered at the network's middle — directional, not a benchmark.)
