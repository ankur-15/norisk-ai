from app.db import SessionLocal, Clause

db = SessionLocal()
rows = db.query(Clause).all()
print(f"Total rows in clauses table: {len(rows)}")
for r in rows:
    print(r.contract_id, "-", r.label, "-", r.text[:50])
db.close()