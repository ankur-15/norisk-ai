import json
import numpy as np
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from fastapi import UploadFile, File
from app.services.pdf_extraction import extract_text_from_pdf
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
@app.post("/contracts/upload-pdf")
async def upload_pdf_contract(
    contract_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    file_bytes = await file.read()
    raw_text = extract_text_from_pdf(file_bytes)

    if not raw_text.strip():
        raise HTTPException(422, "Could not extract any text from this PDF — it may be a scanned image without a text layer")

    index = build_index_from_contract(raw_text)
    labels = classify(index.texts)
    vectors = embed_texts(index.texts)

    try:
        db.query(Clause).filter(Clause.contract_id == contract_id).delete()
        for text, label_info, vector in zip(index.texts, labels, vectors):
            db.add(Clause(
                contract_id=contract_id, text=text,
                label=label_info["label"], confidence=label_info["confidence"],
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