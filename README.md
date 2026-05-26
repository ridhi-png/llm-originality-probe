# Do LLMs Know When They're Being Unoriginal?

> Probing internal model signals as proxies for human-rated originality in LLM-generated text.

## Research Question

Do internal model signals — token entropy, self-perplexity, n-gram novelty, semantic centroid distance — correlate with how humans rate the originality of generated text?

This directly addresses an open question in AI creativity research: **are LLMs structurally blind to their own unoriginality?**

## Hypothesis

Low self-perplexity outputs ("safe" generations) will be rated less original by humans — meaning a model's internal confidence signal is inversely related to perceived creativity.

## Experiment Design

- **Task:** Metaphor generation — "Explain [CONCEPT] using a metaphor"
- **Concepts:** 20 domain concepts (gravity, democracy, memory, rust, loneliness...)
- **Models:** GPT-2-XL (local, HuggingFace) + Mistral-7B-Instruct (Together.ai API)
- **Outputs:** 10 per concept per model = 400 total generated texts
- **Signals extracted per output:**
  1. Mean token entropy
  2. Self-perplexity
  3. 3-gram novelty vs Wikipedia corpus
  4. Type-token ratio (TTR)
  5. Semantic centroid distance
- **Ground truth:** Human originality ratings (1–5 scale, 5 raters, Krippendorff's α)

## Structure

```
llm-originality-probe/
├── scripts/
│   ├── generate_outputs.py       # Generate metaphors from both models
│   ├── extract_signals.py        # Compute all 5 internal signals
│   └── annotate_prep.py          # Prepare Google Form annotation sheet
├── notebooks/
│   └── analysis.ipynb            # Correlation analysis + plots
├── data/
│   ├── concepts.txt              # 20 prompt concepts
│   ├── generated_outputs.csv     # All 400 generated texts (populated by script)
│   └── human_ratings.csv         # Annotator ratings (populated after annotation)
├── signals/
│   └── signal_matrix.csv         # All 5 signals per output (populated by script)
├── requirements.txt
└── README.md
```

## Status

- [x] Repo setup + experimental design
- [ ] Output generation (Week 1)
- [ ] Signal extraction (Week 2)
- [ ] Human annotation (Week 3)
- [ ] Analysis + write-up (Week 4-5)

## Stack

Python · HuggingFace Transformers · GPT-2-XL · Mistral-7B-Instruct · Together.ai API · Sentence-Transformers · Krippendorff's α

## Author

Ridhi Arora — IIT Jodhpur (BS AI/DS, 2025–2029)
