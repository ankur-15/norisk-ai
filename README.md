---
title: NoRisk AI
emoji: ⚖️
colorFrom: blue
colorTo: gray
sdk: docker
app_port: 7860
---

# NoRisk AI

An NLP system that classifies and analyzes legal contract clauses using a fine-tuned transformer model, with semantic search over contract text.

**Live demo:** https://ankursingh1506-norisk-ai.hf.space/docs
**Trained model:** https://huggingface.co/ankursingh1506/norisk-clause-classifier

---

## What it does

Upload a contract, and NoRisk AI will:

1. **Split it into clauses** — chunks the raw text on structural boundaries (numbered sections, sub-clauses, paragraph breaks)
2. **Classify each clause** — a RoBERTa model fine-tuned on 8,000+ labeled examples from the CUAD (Contract Understanding Atticus Dataset) predicts which of 41 legal categories each clause belongs to (e.g. Indemnification, Termination For Convenience, Cap On Liability, Auto-Renewal)
3. **Embed each clause** — a sentence-transformer model converts each clause into a vector capturing its meaning
4. **Enable semantic search** — query clauses by meaning, not just keywords (e.g. searching "who pays if something goes wrong" correctly surfaces indemnification clauses, even with zero keyword overlap)
5. **Persist everything** — contracts and their classified clauses are stored in PostgreSQL, so results survive restarts

## Why "pseudo-RAG"

This isn't a traditional RAG (Retrieval-Augmented Generation) system built for open-ended question answering. It's retrieval (FAISS semantic search) + classification (fine-tuned RoBERTa) working together — the "generation" layer is optional and narrow (explaining one clause at a time, grounded strictly in its text), not a general-purpose chatbot over your contracts.

## Architecture

Contract text
│
▼
┌─────────────┐     regex-based structural split +
│  Chunking   │──── token-window fallback for long
└──────┬──────┘     sections (RoBERTa tokenizer)
│
▼
┌─────────────┐     sentence-transformers/all-MiniLM-L6-v2
│  Embedding  │────  → 384-dim vector per clause
└──────┬──────┘
│
▼
┌─────────────┐     fine-tuned RoBERTa, 41 CUAD categories
│Classification│───  + heuristic risk scoring
└──────┬──────┘
│
▼
┌─────────────┐
│ PostgreSQL  │──── clause text, label, confidence,
└──────┬──────┘     risk score, embedding
│
▼
┌─────────────┐     FAISS-backed semantic search,
│   FastAPI   │──── query clauses by meaning
└─────────────┘

## Tech stack

| Layer | Tech |
|---|---|
| Classification model | RoBERTa (fine-tuned), Hugging Face Transformers |
| Embeddings | Sentence-Transformers (`all-MiniLM-L6-v2`) |
| Semantic search | FAISS |
| API | FastAPI |
| Database | PostgreSQL, SQLAlchemy |
| Training data | CUAD (Contract Understanding Atticus Dataset) |
| Deployment | Docker, Hugging Face Spaces |
| Model hosting | Hugging Face Hub (auto-downloaded at runtime) |

## Project structure

norisk-ai/
├── app/
│   ├── main.py              # FastAPI app, endpoint definitions
│   ├── db.py                 # SQLAlchemy models + session
│   ├── model_loader.py       # auto-downloads trained model from HF Hub
│   └── services/
│       ├── chunking.py       # splits contracts into clause-sized chunks
│       ├── embedding.py      # sentence-transformer embedding
│       ├── classifier.py     # loads + runs the fine-tuned RoBERTa model
│       ├── retrieval.py      # FAISS semantic search
│       └── pipeline.py       # ties chunking + embedding together
├── scripts/
│   ├── download_cuad.py      # downloads + preprocesses CUAD dataset
│   └── train_classifier.py   # fine-tunes RoBERTa on CUAD
├── benchmark.py               # measures batched vs sequential inference speed
├── Dockerfile
└── requirements.txt

## Results

- **41 clause categories** classified, trained on 8,000+ real labeled clauses from CUAD
- **Semantic search works on meaning, not keywords** — e.g. "who pays if something goes wrong" correctly retrieves indemnification clauses
- **2.5x inference speedup** from request batching vs. one-clause-at-a-time processing (see `benchmark.py`)

## Running it locally

**1. Clone and set up environment**
```bash
git clone https://github.com/ankur-15/norisk-ai.git
cd norisk-ai
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

**2. Set up PostgreSQL**

Create a database, then create a `.env` file:DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/norisk

**3. Run the API**
```bash
uvicorn app.main:app --reload
```

Visit `http://localhost:8000/docs` for interactive API testing. The trained model auto-downloads from Hugging Face Hub on first run.

**4. Try it**

Upload a contract via `POST /contracts/upload`:
```json
{
  "contract_id": "test1",
  "text": "1.1 The vendor shall deliver all goods within thirty days.\n\n1.2 Either party may terminate this Agreement upon thirty days written notice."
}
```

Then search it via `POST /search`:
```json
{
  "contract_id": "test1",
  "query": "termination rights",
  "top_k": 2
}
```

## Retraining the model

The trained model is hosted on Hugging Face Hub, not in this repo (too large for git). To retrain from scratch:

```bash
python scripts/download_cuad.py       # downloads + preprocesses CUAD
python scripts/train_classifier.py    # fine-tunes RoBERTa (needs GPU — Colab works)
```

## Known limitations

- Short, generic clauses (e.g. one-line delivery terms) are sometimes misclassified — the model performs best on clauses that closely resemble CUAD's typical clause length and phrasing
- Free-tier Hugging Face Spaces hosting means the app may need a few seconds to "wake up" after inactivity
- Risk scoring is a heuristic layered on top of classification confidence, not a separately trained risk model

## Links

- **Live app:** https://ankursingh1506-norisk-ai.hf.space/docs
- **Model card:** https://huggingface.co/ankursingh1506/norisk-clause-classifier
- **Code:** https://github.com/ankur-15/norisk-ai