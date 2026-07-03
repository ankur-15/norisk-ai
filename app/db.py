import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, String, Float, Integer, Text
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()  # reads .env file into environment variables

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:ankur15@localhost:5432/norisk")
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class Clause(Base):
    __tablename__ = "clauses"
    id = Column(Integer, primary_key=True)
    contract_id = Column(String)
    text = Column(Text)
    label = Column(String)
    confidence = Column(Float)
    risk_score = Column(Float)
    embedding_json = Column(Text)

Base.metadata.create_all(engine)