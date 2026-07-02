# NoRisk AI

NLP system for classifying and analyzing legal contract clauses using a fine-tuned RoBERTa model, with FAISS-based semantic search over contract text.

## Stack
Python, FastAPI, Hugging Face Transformers, Sentence-Transformers, FAISS, PostgreSQL, SQLAlchemy

## How it works
1. Upload a contract → split into clause-level chunks
2. Each chunk is embedded (sentence-transformers) and classified (fine-tuned RoBERTa on CUAD dataset, 41 legal clause categories)
3. Semantic search lets you query clauses by meaning, not just keywords
4. Results persist in PostgreSQL

Trained model hosted separately on [Hugging Face Hub](https://huggingface.co/ankursingh1506/norisk-clause-classifier) and downloaded automatically at startup.

## Run locally
\`\`\`
pip install -r requirements.txt
cp .env.example .env   # fill in your DATABASE_URL
uvicorn app.main:app --reload
\`\`\`
Visit `/docs` for interactive API testing.