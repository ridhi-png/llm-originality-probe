"""
generate_outputs.py
-------------------
Generates metaphor outputs from GPT-2-XL (local) and Mistral-7B (Together.ai API).
Run this first. Saves results to data/generated_outputs.csv

Usage:
    # GPT-2-XL only (no API key needed):
    python scripts/generate_outputs.py --model gpt2

    # Mistral via Together.ai (needs API key):
    python scripts/generate_outputs.py --model mistral --api_key YOUR_KEY

    # Both:
    python scripts/generate_outputs.py --model both --api_key YOUR_KEY
"""

import argparse
import pandas as pd
import torch
import requests
import json
import os
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

CONCEPTS_PATH = "data/concepts.txt"
OUTPUT_PATH = "data/generated_outputs.csv"
N_OUTPUTS_PER_CONCEPT = 10
TOGETHER_API_URL = "https://api.together.xyz/v1/chat/completions"

PROMPT_TEMPLATE = "Explain {concept} using a single, creative metaphor. Be as original as possible. Metaphor:"


def load_concepts():
    with open(CONCEPTS_PATH, "r") as f:
        return [line.strip() for line in f if line.strip()]


def generate_gpt2(concepts, n=N_OUTPUTS_PER_CONCEPT):
    print("\nLoading GPT-2-XL...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
    model = GPT2LMHeadModel.from_pretrained("gpt2-xl")
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    tokenizer.pad_token = tokenizer.eos_token

    rows = []
    for concept in tqdm(concepts, desc="GPT-2-XL"):
        prompt = PROMPT_TEMPLATE.format(concept=concept)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        for i in range(n):
            with torch.no_grad():
                out = model.generate(
                    **inputs,
                    max_new_tokens=60,
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.92,
                    pad_token_id=tokenizer.eos_token_id,
                )
            generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            generated = generated.split("\n")[0].strip()
            rows.append({
                "concept": concept,
                "model": "gpt2-xl",
                "run_id": i,
                "prompt": prompt,
                "output": generated,
            })
    return rows


def generate_mistral(concepts, api_key, n=N_OUTPUTS_PER_CONCEPT):
    print("\nCalling Mistral-7B via Together.ai...")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    rows = []
    for concept in tqdm(concepts, desc="Mistral-7B"):
        prompt = PROMPT_TEMPLATE.format(concept=concept)
        for i in range(n):
            payload = {
                "model": "mistralai/Mistral-7B-Instruct-v0.2",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 80,
                "temperature": 0.9,
                "top_p": 0.92,
            }
            try:
                resp = requests.post(TOGETHER_API_URL, headers=headers, json=payload, timeout=30)
                resp.raise_for_status()
                content = resp.json()["choices"][0]["message"]["content"].strip()
                content = content.split("\n")[0].strip()
            except Exception as e:
                print(f"  Error for {concept} run {i}: {e}")
                content = ""
            rows.append({
                "concept": concept,
                "model": "mistral-7b",
                "run_id": i,
                "prompt": prompt,
                "output": content,
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["gpt2", "mistral", "both"], default="gpt2")
    parser.add_argument("--api_key", default=os.environ.get("TOGETHER_API_KEY", ""))
    args = parser.parse_args()

    concepts = load_concepts()
    print(f"Loaded {len(concepts)} concepts.")

    all_rows = []

    if args.model in ("gpt2", "both"):
        all_rows += generate_gpt2(concepts)

    if args.model in ("mistral", "both"):
        if not args.api_key:
            print("ERROR: --api_key required for Mistral. Get a free key at https://api.together.ai")
        else:
            all_rows += generate_mistral(concepts, args.api_key)

    df = pd.DataFrame(all_rows)
    os.makedirs("data", exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(df)} outputs to {OUTPUT_PATH}")
    print(df.head())


if __name__ == "__main__":
    main()
