import json
import random
from pathlib import Path
import pandas as pd
from datasets import load_dataset

RANDOM_SEED = 42

def extract_category(question: str) -> str:
    # question format: '...related to "Category Name" that...'
    start = question.find('"')
    end = question.find('"', start + 1)
    return question[start + 1:end] if start != -1 and end != -1 else "Unknown"

def flatten_cuad(dataset) -> pd.DataFrame:
    rows = []
    for example in dataset:
        category = extract_category(example["question"])
        for answer_text in example["answers"]["text"]:
            if len(answer_text.strip()) > 20:
                rows.append({"text": answer_text.strip(), "label": category})
    return pd.DataFrame(rows).drop_duplicates(subset=["text"])

def add_negatives(df: pd.DataFrame, dataset, n_negatives: int) -> pd.DataFrame:
    random.seed(RANDOM_SEED)
    all_contexts = list({ex["context"] for ex in dataset})
    negatives = []
    existing_texts = set(df["text"])

    while len(negatives) < n_negatives and all_contexts:
        context = random.choice(all_contexts)
        sentences = [s.strip() for s in context.split(".") if len(s.strip()) > 40]
        if not sentences:
            continue
        candidate = random.choice(sentences)
        if candidate not in existing_texts:
            negatives.append({"text": candidate, "label": "No Clause"})
            existing_texts.add(candidate)

    return pd.concat([df, pd.DataFrame(negatives)], ignore_index=True)

print("Downloading CUAD (this pulls real contract data, may take a few minutes)...")
dataset = load_dataset("theatticusproject/cuad-qa", split="train", trust_remote_code=True)

df = flatten_cuad(dataset)
print(f"\nTotal labeled clauses extracted: {len(df)}")
print(f"Number of distinct categories: {df['label'].nunique()}")
print("\nSample rows:")
print(df.sample(5, random_state=RANDOM_SEED)[["label", "text"]].to_string())
print("\nTop 10 most common categories:")
print(df["label"].value_counts().head(10))

# --- Step 8: add negative ("No Clause") examples ---
n_negatives = int(len(df) * 0.15)
df = add_negatives(df, dataset, n_negatives)
print(f"\nAfter adding negatives: {len(df)} total examples")

# --- Step 9: build label map, split, save ---
labels = sorted(df["label"].unique())
label2id = {label: i for i, label in enumerate(labels)}

df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
n = len(df)
train_df = df.iloc[: int(n * 0.8)]
val_df = df.iloc[int(n * 0.8): int(n * 0.9)]
test_df = df.iloc[int(n * 0.9):]

Path("data/processed").mkdir(parents=True, exist_ok=True)
train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/val.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

with open("data/processed/label_map.json", "w") as f:
    json.dump({v: k for k, v in label2id.items()}, f, indent=2)

print(f"\nSplit: {len(train_df)} train / {len(val_df)} val / {len(test_df)} test")
print(f"Saved label_map.json with {len(label2id)} labels")