import json
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from app.model_loader import ensure_model_downloaded
ensure_model_downloaded()
MODEL_PATH = Path("models/roberta-clause-classifier")

RISK_WEIGHTS = {
    "Uncapped Liability": 0.95, "Cap On Liability": 0.55, "Exclusivity": 0.75,
    "Audit Rights": 0.5, "Anti-Assignment": 0.4, "Governing Law": 0.3,
    "Post-Termination Services": 0.6,
}
DEFAULT_RISK = 0.5

_tokenizer = None
_model = None
_id2label = None

def _load():
    global _tokenizer, _model, _id2label
    if _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        _model.eval()
        with open(MODEL_PATH / "label_map.json") as f:
            _id2label = {int(k): v for k, v in json.load(f).items()}

@torch.no_grad()
def classify(texts: list[str]) -> list[dict]:
    _load()
    inputs = _tokenizer(texts, padding=True, truncation=True, max_length=256, return_tensors="pt")
    probs = torch.softmax(_model(**inputs).logits, dim=-1)
    confidences, pred_ids = torch.max(probs, dim=-1)
    results = []
    for pid, conf in zip(pred_ids.tolist(), confidences.tolist()):
        label = _id2label[pid]
        risk = RISK_WEIGHTS.get(label, DEFAULT_RISK) * conf
        results.append({"label": label, "confidence": round(conf, 3), "risk_score": round(risk, 3)})
    return results