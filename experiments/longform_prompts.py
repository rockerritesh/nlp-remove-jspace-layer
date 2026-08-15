#!/usr/bin/env python3
"""
100 long-form prompts for the 405B layer-removal study.

These target BASE (non-Instruct) Llama-3.1-405B, so each prompt is written as a
completion lead-in (instruction + an explicit opening cue like "The story:") —
base models continue text, they do not follow chat instructions.

Five capability families, chosen because the workspace/J-space hypothesis predicts
they degrade at different rates:
  narrative  — long-range plot & entity coherence
  code       — hard syntactic + semantic constraints, machine-checkable
  explain    — factual/technical exposition
  procedure  — ordered multi-step planning
  reasoning  — argumentative / analytical structure
"""

NARRATIVE = [
    "Write a long, detailed story about a lighthouse keeper who discovers a message in a bottle.\n\nThe story:\n",
    "Write a long, detailed story about a cartographer who maps a city that keeps rearranging itself.\n\nThe story:\n",
    "Write a long, detailed story about the last bookshop on a colony orbiting Jupiter.\n\nThe story:\n",
    "Write a long, detailed story about two sisters who inherit a farm and disagree about its future.\n\nThe story:\n",
    "Write a long, detailed story about a watchmaker who repairs a clock that runs backwards.\n\nThe story:\n",
    "Write a long, detailed story about a translator hired to interpret for a machine.\n\nThe story:\n",
    "Write a long, detailed story about a village that shares a single collective memory.\n\nThe story:\n",
    "Write a long, detailed story about a deep-sea welder who hears singing in the pipes.\n\nThe story:\n",
    "Write a long, detailed story about a chess prodigy who loses the ability to plan ahead.\n\nThe story:\n",
    "Write a long, detailed story about a night-shift radio host and a caller who never hangs up.\n\nThe story:\n",
    "Write a long, detailed story about a botanist cataloguing plants that grow only in ruins.\n\nThe story:\n",
    "Write a long, detailed story about a courier carrying a package that must never be opened.\n\nThe story:\n",
    "Write a long, detailed story about a family reunion held in a house that is being demolished.\n\nThe story:\n",
    "Write a long, detailed story about an archivist who finds a photograph of herself as a stranger.\n\nThe story:\n",
    "Write a long, detailed story about a mountain rescue team searching for a climber who left a note.\n\nThe story:\n",
    "Write a long, detailed story about a baker whose bread makes people remember their childhood.\n\nThe story:\n",
    "Write a long, detailed story about a retired detective asked to solve one final case.\n\nThe story:\n",
    "Write a long, detailed story about a generation ship where the crew has forgotten the destination.\n\nThe story:\n",
    "Write a long, detailed story about a river that changes course and splits a town in two.\n\nThe story:\n",
    "Write a long, detailed story about a puppeteer whose puppets begin improvising.\n\nThe story:\n",
    "Write a long, detailed story about an apprentice glassblower and a commission that cannot be made.\n\nThe story:\n",
    "Write a long, detailed story about a lighthouse of the desert: a radio tower and its keeper.\n\nThe story:\n",
    "Write a long, detailed story about a teacher who returns to the village she once fled.\n\nThe story:\n",
    "Write a long, detailed story about a beekeeper negotiating with a neighbouring orchard.\n\nThe story:\n",
    "Write a long, detailed story about a stage magician who performs one honest trick.\n\nThe story:\n",
]

CODE = [
    "Write a complete, well-commented Python module that implements a LRU cache with a fixed capacity.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a binary search tree with insert, delete and in-order traversal.\n\n```python\n",
    "Write a complete, well-commented Python module that parses a CSV file and computes per-column summary statistics.\n\n```python\n",
    "Write a complete, well-commented Python module that implements Dijkstra's shortest path algorithm on a weighted graph.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a simple recursive-descent parser for arithmetic expressions.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a thread-safe bounded queue using locks.\n\n```python\n",
    "Write a complete, well-commented Python module that implements run-length encoding and decoding for byte strings.\n\n```python\n",
    "Write a complete, well-commented Python module that implements k-means clustering from scratch using only the standard library.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a retry decorator with exponential backoff.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a minimal JSON serialiser without using the json library.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a priority queue using a binary heap.\n\n```python\n",
    "Write a complete, well-commented Python module that implements matrix multiplication and transposition for nested lists.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a simple event emitter with subscribe and publish.\n\n```python\n",
    "Write a complete, well-commented Python module that implements the Levenshtein edit distance with a traceback.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a rolling-window rate limiter.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a topological sort and detects cycles.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a tiny key-value store persisted to disk.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a state machine for a vending machine.\n\n```python\n",
    "Write a complete, well-commented Python module that implements binary search plus insertion-point lookup.\n\n```python\n",
    "Write a complete, well-commented Python module that implements a simple Markov-chain text generator.\n\n```python\n",
]

EXPLAIN = [
    "Write a detailed technical explanation of how a refrigerator moves heat against a temperature gradient.\n\nExplanation:\n",
    "Write a detailed technical explanation of how the TCP three-way handshake establishes a reliable connection.\n\nExplanation:\n",
    "Write a detailed technical explanation of how photosynthesis converts light into chemical energy.\n\nExplanation:\n",
    "Write a detailed technical explanation of how a public-key cryptosystem allows strangers to exchange secrets.\n\nExplanation:\n",
    "Write a detailed technical explanation of how the human cochlea separates sound into frequencies.\n\nExplanation:\n",
    "Write a detailed technical explanation of how a jet engine produces thrust at cruising altitude.\n\nExplanation:\n",
    "Write a detailed technical explanation of how vaccines train the adaptive immune system.\n\nExplanation:\n",
    "Write a detailed technical explanation of how plate tectonics produces earthquakes and mountain ranges.\n\nExplanation:\n",
    "Write a detailed technical explanation of how a transformer steps voltage up and down.\n\nExplanation:\n",
    "Write a detailed technical explanation of how garbage collection reclaims unreachable memory.\n\nExplanation:\n",
    "Write a detailed technical explanation of how GPS receivers determine position from satellite signals.\n\nExplanation:\n",
    "Write a detailed technical explanation of how the greenhouse effect regulates planetary temperature.\n\nExplanation:\n",
    "Write a detailed technical explanation of how CRISPR-Cas9 edits a targeted DNA sequence.\n\nExplanation:\n",
    "Write a detailed technical explanation of how a hash table achieves average constant-time lookup.\n\nExplanation:\n",
    "Write a detailed technical explanation of how noise-cancelling headphones suppress ambient sound.\n\nExplanation:\n",
    "Write a detailed technical explanation of how blood clotting cascades stop a wound from bleeding.\n\nExplanation:\n",
    "Write a detailed technical explanation of how a nuclear reactor sustains and controls a chain reaction.\n\nExplanation:\n",
    "Write a detailed technical explanation of how compilers translate source code into machine instructions.\n\nExplanation:\n",
    "Write a detailed technical explanation of how ocean currents redistribute heat around the globe.\n\nExplanation:\n",
    "Write a detailed technical explanation of how an MRI scanner produces images of soft tissue.\n\nExplanation:\n",
]

PROCEDURE = [
    "Write a detailed step-by-step plan for organising a small academic conference from scratch.\n\nPlan:\n",
    "Write a detailed step-by-step plan for migrating a production database with zero downtime.\n\nPlan:\n",
    "Write a detailed step-by-step plan for restoring a neglected fruit orchard over one growing season.\n\nPlan:\n",
    "Write a detailed step-by-step plan for onboarding a new engineer in their first month.\n\nPlan:\n",
    "Write a detailed step-by-step plan for conducting a building energy audit.\n\nPlan:\n",
    "Write a detailed step-by-step plan for running a randomised controlled trial of a teaching method.\n\nPlan:\n",
    "Write a detailed step-by-step plan for setting up a community lending library.\n\nPlan:\n",
    "Write a detailed step-by-step plan for preparing a research vessel for a six-week survey.\n\nPlan:\n",
    "Write a detailed step-by-step plan for incident response after a suspected data breach.\n\nPlan:\n",
    "Write a detailed step-by-step plan for building and calibrating a home weather station.\n\nPlan:\n",
    "Write a detailed step-by-step plan for translating and publishing a book in a second language.\n\nPlan:\n",
    "Write a detailed step-by-step plan for winterising a small sailing boat.\n\nPlan:\n",
    "Write a detailed step-by-step plan for launching a municipal composting programme.\n\nPlan:\n",
    "Write a detailed step-by-step plan for auditing and reducing a company's cloud spending.\n\nPlan:\n",
    "Write a detailed step-by-step plan for staging a school play with a cast of thirty children.\n\nPlan:\n",
]

REASONING = [
    "Write a detailed, balanced analysis of whether cities should replace parking minimums with congestion pricing.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether scientific peer review should be conducted openly.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether a four-day working week improves total productivity.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether nuclear power belongs in a decarbonisation strategy.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether universities should weight standardised tests in admissions.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether remote work weakens the transfer of tacit knowledge.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether copyright terms should be shortened.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether central banks should target nominal GDP instead of inflation.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether autonomous vehicles will reduce total road deaths.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether large-scale desalination is a sustainable answer to water scarcity.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether open-source models help or harm AI safety.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether historical monuments should be relocated to museums.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether high-speed rail is justified in low-density countries.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether antibiotic use in livestock should be banned outright.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether a universal basic income would reduce poverty traps.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether social media platforms should be liable for recommendations.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether gene drives should be deployed against malaria vectors.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether space mining should be governed by a global treaty.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether cash transfers outperform in-kind aid in disaster relief.\n\nAnalysis:\n",
    "Write a detailed, balanced analysis of whether teaching handwriting still matters in a digital era.\n\nAnalysis:\n",
]

_FAMILIES = [
    ("narrative", NARRATIVE),
    ("code", CODE),
    ("explain", EXPLAIN),
    ("procedure", PROCEDURE),
    ("reasoning", REASONING),
]

PROMPTS = [
    {"id": f"{cat}_{i:02d}", "category": cat, "text": text}
    for cat, items in _FAMILIES
    for i, text in enumerate(items)
]

if __name__ == "__main__":
    from collections import Counter
    print(len(PROMPTS), "prompts:", dict(Counter(p["category"] for p in PROMPTS)))
