import json
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db import SessionLocal, Clause
from app.services.pipeline import build_index_from_contract
from app.services.classifier import classify
from app.services.embedding import embed_texts
from app.services.retrieval import ClauseIndex

app = FastAPI(title="NoRisk AI")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class ClassifyRequest(BaseModel):
    text: str

class UploadRequest(BaseModel):
    contract_id: str
    text: str

class SearchRequest(BaseModel):
    contract_id: str
    query: str
    top_k: int = 3

@app.post("/classify")
def classify_endpoint(req: ClassifyRequest):
    return classify([req.text])[0]

@app.post("/contracts/upload")
def upload_contract(req: UploadRequest, db: Session = Depends(get_db)):
    index = build_index_from_contract(req.text)
    labels = classify(index.texts)
    vectors = embed_texts(index.texts)  # real embeddings, not a placeholder

    try:
        # delete any previous rows for this contract_id so re-uploads don't duplicate
        db.query(Clause).filter(Clause.contract_id == req.contract_id).delete()

        for text, label_info, vector in zip(index.texts, labels, vectors):
            db.add(Clause(
                contract_id=req.contract_id,
                text=text,
                label=label_info["label"],
                confidence=label_info["confidence"],
                risk_score=label_info["risk_score"],
                embedding_json=json.dumps(vector.tolist()),
            ))
        db.commit()
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Database write failure: {str(e)}")

    return {"num_clauses": len(index.texts), "clauses": [
        {"text": t, **l} for t, l in zip(index.texts, labels)
    ]}

@app.post("/search")
def search_contract(req: SearchRequest, db: Session = Depends(get_db)):
    clauses = db.query(Clause).filter(Clause.contract_id == req.contract_id).all()
    if not clauses:
        return {"error": "contract not found, upload it first"}

    # rebuild a fresh FAISS index from the stored real embeddings
    index = ClauseIndex()
    vectors = np.array([json.loads(c.embedding_json) for c in clauses], dtype="float32")
    index.index.add(vectors)
    index.texts = [c.text for c in clauses]

    results = index.search(req.query, top_k=req.top_k)
    return {"results": [{"text": t, "similarity": s} for t, s in results]}

@app.get("/health")
def health():
    return {"status": "ok"}