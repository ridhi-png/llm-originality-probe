"""
extract_signals.py
------------------
Computes 5 internal novelty signals for every output in data/generated_outputs.csv.
Saves results to signals/signal_matrix.csv

Signals computed:
    1. mean_token_entropy     — average uncertainty across generation steps
    2. self_perplexity        — how surprised the model is by its own output
    3. ngram_novelty          — fraction of 3-grams absent from Wikipedia sample
    4. type_token_ratio       — lexical diversity (unique tokens / total tokens)
    5. centroid_distance      — semantic distance from the mean output per concept+model

Usage:
    python scripts/extract_signals.py

Requires data/generated_outputs.csv to exist (run generate_outputs.py first).
"""

import pandas as pd
import numpy as np
import torch
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from collections import Counter

INPUT_PATH  = "data/generated_outputs.csv"
OUTPUT_PATH = "signals/signal_matrix.csv"
WIKI_NGRAM_PATH = "data/wiki_ngrams.txt"   # auto-built on first run


# ── 1 & 2: Token entropy + self-perplexity via GPT-2-XL ──────────────────────

def load_gpt2():
    print("Loading GPT-2-XL for entropy/perplexity scoring...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def compute_entropy_and_perplexity(text, model, tokenizer, device):
    """Returns (mean_token_entropy, self_perplexity) for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
    input_ids = inputs["input_ids"]
    if input_ids.shape[1] < 2:
        return 0.0, 0.0
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        logits = outputs.logits  # (1, seq_len, vocab_size)

    # Token entropy: H = -sum(p * log p) per position
    probs = torch.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)
    log_probs = torch.log(probs + 1e-10)
    entropy_per_token = -torch.sum(probs * log_probs, dim=-1)  # (seq_len,)
    mean_entropy = entropy_per_token.mean().item()

    # Self-perplexity: exp(mean NLL)
    loss = outputs.loss.item()  # mean NLL already computed
    self_perplexity = float(np.exp(loss))

    return round(mean_entropy, 4), round(self_perplexity, 4)


# ── 3: N-gram novelty vs Wikipedia ───────────────────────────────────────────

def build_wiki_ngrams(n=3, sample_size=5000):
    """Build a set of n-grams from a Wikipedia sample. Cached to disk."""
    if os.path.exists(WIKI_NGRAM_PATH):
        print("Loading cached Wikipedia n-grams...")
        with open(WIKI_NGRAM_PATH) as f:
            return set(line.strip() for line in f)

    print("Building Wikipedia n-gram reference corpus (first run only)...")
    try:
        from datasets import load_dataset
        wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
        ngrams = set()
        for i, row in enumerate(wiki):
            if i >= sample_size:
                break
            words = row["text"].lower().split()
            for j in range(len(words) - n + 1):
                ngrams.add(" ".join(words[j:j+n]))
        os.makedirs("data", exist_ok=True)
        with open(WIKI_NGRAM_PATH, "w") as f:
            for ng in ngrams:
                f.write(ng + "\n")
        print(f"Saved {len(ngrams)} Wikipedia {n}-grams.")
        return ngrams
    except Exception as e:
        print(f"Warning: Could not load Wikipedia dataset ({e}). Using empty reference.")
        return set()


def ngram_novelty(text, wiki_ngrams, n=3):
    """Fraction of n-grams in text NOT found in Wikipedia."""
    words = text.lower().split()
    if len(words) < n:
        return 0.0
    grams = [" ".join(words[i:i+n]) for i in range(len(words)-n+1)]
    if not grams:
        return 0.0
    novel = sum(1 for g in grams if g not in wiki_ngrams)
    return round(novel / len(grams), 4)


# ── 4: Type-token ratio ───────────────────────────────────────────────────────

def type_token_ratio(text):
    words = text.lower().split()
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


# ── 5: Semantic centroid distance ─────────────────────────────────────────────

def compute_centroid_distances(df, embedder):
    """
    For each (concept, model) group, embed all outputs,
    compute the mean embedding (centroid), and return
    cosine distance from centroid for each output.
    """
    print("Computing semantic centroid distances...")
    distances = []
    for (concept, model), group in df.groupby(["concept", "model"]):
        texts = group["output"].tolist()
        if not any(texts):
            distances.extend([0.0] * len(texts))
            continue
        embeddings = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        centroid = embeddings.mean(axis=0)
        # Cosine distance = 1 - cosine_similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        centroid_norm = np.linalg.norm(centroid)
        sims = (embeddings @ centroid) / (norms.flatten() * centroid_norm + 1e-10)
        dists = [round(float(1 - s), 4) for s in sims]
        distances.extend(dists)

    # re-align with original df order
    df = df.copy()
    df["centroid_distance"] = 0.0
    idx = 0
    for (concept, model), group in df.groupby(["concept", "model"]):
        size = len(group)
        df.loc[group.index, "centroid_distance"] = distances[idx:idx+size]
        idx += size
    return df["centroid_distance"].tolist()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} outputs from {INPUT_PATH}")

    # Filter empty outputs
    df = df[df["output"].notna() & (df["output"].str.strip() != "")].reset_index(drop=True)
    print(f"{len(df)} non-empty outputs to process.")

    # Signal 1 & 2: entropy + perplexity
    model, tokenizer, device = load_gpt2()
    entropies, perplexities = [], []
    for text in tqdm(df["output"], desc="Entropy + Perplexity"):
        e, p = compute_entropy_and_perplexity(text, model, tokenizer, device)
        entropies.append(e)
        perplexities.append(p)
    df["mean_token_entropy"] = entropies
    df["self_perplexity"]    = perplexities

    # Signal 3: n-gram novelty
    wiki_ngrams = build_wiki_ngrams()
    df["ngram_novelty"] = [ngram_novelty(t, wiki_ngrams) for t in tqdm(df["output"], desc="N-gram Novelty")]

    # Signal 4: TTR
    df["type_token_ratio"] = [type_token_ratio(t) for t in df["output"]]

    # Signal 5: centroid distance
    print("Loading sentence-transformers model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    df["centroid_distance"] = compute_centroid_distances(df, embedder)

    # Save
    os.makedirs("signals", exist_ok=True)
    signal_cols = ["concept", "model", "run_id", "output",
                   "mean_token_entropy", "self_perplexity",
                   "ngram_novelty", "type_token_ratio", "centroid_distance"]
    df[signal_cols].to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved signal matrix to {OUTPUT_PATH}")
    print(df[signal_cols].describe())


if __name__ == "__main__":
    main()
