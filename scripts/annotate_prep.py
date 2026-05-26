"""
annotate_prep.py
----------------
Prepares a randomised annotation sheet from generated_outputs.csv.
Creates data/annotation_sheet.csv — paste this into Google Forms or share directly.

Each rater sees 80 outputs in random order (blind to model and concept).
They rate each on: "How original / surprising is this metaphor?" (1–5)

Usage:
    python scripts/annotate_prep.py --n_per_rater 80 --n_raters 5
"""

import pandas as pd
import numpy as np
import argparse
import os

INPUT_PATH  = "data/generated_outputs.csv"
OUTPUT_PATH = "data/annotation_sheet.csv"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_per_rater", type=int, default=80)
    parser.add_argument("--n_raters",    type=int, default=5)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)
    df = pd.read_csv(INPUT_PATH)
    df = df[df["output"].notna() & (df["output"].str.strip() != "")].reset_index(drop=True)

    # Assign a unique output_id
    df["output_id"] = df.index

    total_needed = args.n_per_rater * args.n_raters
    if total_needed > len(df):
        # Allow overlap — sample with replacement if needed
        sample = df.sample(n=total_needed, replace=True, random_state=args.seed).reset_index(drop=True)
    else:
        sample = df.sample(n=total_needed, replace=False, random_state=args.seed).reset_index(drop=True)

    # Split into rater batches
    rows = []
    for rater_id in range(1, args.n_raters + 1):
        batch = sample.iloc[(rater_id-1)*args.n_per_rater : rater_id*args.n_per_rater].copy()
        batch["rater_id"] = rater_id
        # Shuffle within rater batch
        batch = batch.sample(frac=1, random_state=args.seed + rater_id).reset_index(drop=True)
        batch["position"] = range(1, len(batch)+1)
        rows.append(batch)

    annotation_df = pd.concat(rows, ignore_index=True)

    # Only show rater what they need — hide model name (blind evaluation)
    out_cols = ["rater_id", "position", "output_id", "output"]
    annotation_df[out_cols].to_csv(OUTPUT_PATH, index=False)

    print(f"Annotation sheet saved to {OUTPUT_PATH}")
    print(f"  {args.n_raters} raters × {args.n_per_rater} outputs = {total_needed} ratings total")
    print(f"\nInstructions for raters:")
    print("  Rate each metaphor 1–5 for originality/surprise:")
    print("  1 = very predictable / cliché")
    print("  3 = somewhat original")
    print("  5 = genuinely surprising / fresh")
    print("\nPaste the 'output' column into a Google Form, one row per question.")


if __name__ == "__main__":
    main()
