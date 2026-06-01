# Do LLMs Know When They're Being Unoriginal?

> Probing internal model signals as proxies for human-rated originality in LLM-generated text.

## Research Question
Do internal model signals — token entropy, self-perplexity, 
n-gram novelty, semantic centroid distance — correlate with 
human-rated originality of generated text?

## Hypothesis
Low self-perplexity outputs ("safe" generations) are rated 
less original by humans — suggesting LLMs are structurally 
blind to their own unoriginality.

## Experimental Design
- **Models:** GPT-2-XL, Mistral-7B-Instruct
- **Task:** Metaphor generation across 20 domain concepts
- **Outputs:** 400 total (20 concepts × 10 outputs × 2 models)
- **Signals measured:** mean token entropy, self-perplexity, 
  3-gram novelty (vs Wikipedia), type-token ratio, semantic centroid distance
- **Human annotation:** 5 raters, Krippendorff's α for agreement scoring

## Current Status
- [x] Experimental design finalized
- [x] Generation pipeline built (GPT-2)
- [x] Pilot outputs generated — 10 outputs across 5 concepts
- [x] Signal extraction complete — entropy + perplexity on pilot set
- [x] Pilot human annotation complete — originality ratings collected
- [x] Pilot correlation analysis run — negative correlation observed
- [ ] Scaling to full dataset — 400 outputs, 20 concepts, 2 models
- [ ] N-gram novelty + centroid distance signals — in progress
- [ ] Full human annotation — 5 raters pending
- [ ] Krippendorff's α computation — pending full annotation

## Pilot Results (10 outputs, single rater)
| Signal | Correlation with Human Originality |
|--------|-----------------------------------|
| mean_token_entropy | -0.24 |
| self_perplexity | -0.14 |

> Early pilot suggests a weak negative correlation — 
> higher entropy/perplexity does not strongly predict 
> human-rated originality at this scale. 
> Full dataset needed before drawing conclusions.

## View Analysis
📊 [View pilot analysis notebook on nbviewer](https://nbviewer.org/github/ridhi-png/llm-originality-probe/blob/main/notebooks/analysis.ipynb)

## Stack
Python, HuggingFace Transformers, Sentence-Transformers, 
Mistral-7B-Instruct, GPT-2, NLTK, NumPy, Pandas, Matplotlib

## Repository Structure
