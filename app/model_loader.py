from pathlib import Path
from huggingface_hub import snapshot_download

MODEL_DIR = Path("models/roberta-clause-classifier")
HF_MODEL_REPO = "ankursingh1506/norisk-clause-classifier"

def ensure_model_downloaded():
    if MODEL_DIR.exists() and (MODEL_DIR / "model.safetensors").exists():
        return
    print(f"Downloading model from {HF_MODEL_REPO}...")
    snapshot_download(repo_id=HF_MODEL_REPO, local_dir=MODEL_DIR)
    print("Model download complete.")