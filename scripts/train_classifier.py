import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer,
    DataCollatorWithPadding, Trainer, TrainingArguments,
)

DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("models/roberta-clause-classifier")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

with open(DATA_DIR / "label_map.json") as f:
    id2label = {int(k): v for k, v in json.load(f).items()}
label2id = {v: k for k, v in id2label.items()}

def load_split(name):
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df["label"] = df["label"].map(label2id)
    return Dataset.from_pandas(df[["text", "label"]], preserve_index=False)

train_ds, val_ds = load_split("train"), load_split("val")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(
    "roberta-base", num_labels=len(label2id), id2label=id2label, label2id=label2id
)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, max_length=256)

train_ds = train_ds.map(tokenize, batched=True)
val_ds = val_ds.map(tokenize, batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1, "precision": p, "recall": r}

args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "checkpoints"),
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_steps=50,
    fp16=torch.cuda.is_available(),
    report_to="none",
)

trainer = Trainer(
    model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("Training...")
trainer.train()
print("Final metrics:", trainer.evaluate())

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
with open(OUTPUT_DIR / "label_map.json", "w") as f:
    json.dump(id2label, f, indent=2)