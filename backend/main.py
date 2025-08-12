
import os
from datetime import datetime, timedelta
from typing import Optional, List

from fastapi import FastAPI, Depends, HTTPException, status, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session, relationship
from passlib.context import CryptContext
import jwt
from jwt import PyJWTError

# Optional Transformers with sklearn fallback
USE_TRANSFORMERS = os.environ.get("USE_TRANSFORMERS", "true").lower() == "true"

# ---- Config ----
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./app.db")
SECRET_KEY = os.environ.get("SECRET_KEY", "CHANGE_ME_IN_PROD")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ---- App & CORS ----
app = FastAPI(title="FastAPI + Streamlit (Roles + Postgres + Real ML + RAG)", version="4.0.0")
origins = ["http://localhost:8501", "http://127.0.0.1:8501"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# ---- DB ----
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="user")
    created_at = Column(DateTime, default=datetime.utcnow)

    predictions = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    text = Column(String, nullable=False)
    length = Column(Integer)
    words = Column(Integer)
    uppercase = Column(String)
    sentiment = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="predictions")

Base.metadata.create_all(bind=engine)

# ---- Auth ----
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except PyJWTError:
        raise credentials_exception
    user = db.query(User).filter(User.username == username).first()
    if user is None:
        raise credentials_exception
    return user

def require_admin(user: User = Depends(get_current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user

# ---- Schemas ----
class PredictIn(BaseModel):
    text: str

class PredictOut(BaseModel):
    length: int
    words: int
    uppercase: str
    sentiment: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class RegisterIn(BaseModel):
    username: str
    password: str

class HistoryItem(BaseModel):
    id: int
    text: str
    length: int
    words: int
    uppercase: str
    sentiment: str
    created_at: datetime

class MeOut(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

class RoleUpdate(BaseModel):
    role: str

class UserInfo(BaseModel):
    id: int
    username: str
    role: str
    created_at: datetime

# ---- Sentiment model(s) ----
MODEL_PATH = "model.joblib"
transformers_pipe = None
def setup_transformers():
    global transformers_pipe
    try:
        from transformers import pipeline
        transformers_pipe = pipeline("sentiment-analysis")
    except Exception:
        transformers_pipe = None

def predict_sentiment_transformers(text: str) -> Optional[str]:
    if transformers_pipe is None:
        return None
    res = transformers_pipe(text)[0]
    label = res.get("label", "").lower()
    if "pos" in label or "positive" in label:
        return "positive"
    if "neg" in label or "negative" in label:
        return "negative"
    return "positive" if float(res.get("score", 0.5)) >= 0.5 else "negative"

def train_or_load_sklearn():
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    texts = [
        "I love this product, it is amazing and works great",
        "This is the best day ever, absolutely wonderful",
        "I hate it, terrible and disappointing",
        "Worst experience of my life, very bad",
        "So happy and satisfied with the results",
        "Awful, broken, waste of money",
        "Pretty good overall, I'm pleased",
        "Not good at all, very upset",
        "Excellent quality and great value",
        "Bad quality and not worth it",
    ]
    labels = ["pos","pos","neg","neg","pos","neg","pos","neg","pos","neg"]
    pipe = Pipeline([("tfidf", TfidfVectorizer()), ("clf", LogisticRegression(max_iter=1000))])
    pipe.fit(texts, labels)
    joblib.dump(pipe, MODEL_PATH)
    return pipe

sk_model = None
if USE_TRANSFORMERS:
    setup_transformers()

def predict_sentiment(text: str) -> str:
    if USE_TRANSFORMERS and transformers_pipe is not None:
        out = predict_sentiment_transformers(text)
        if out is not None:
            return out
    global sk_model
    if sk_model is None:
        sk_model = train_or_load_sklearn()
    label = sk_model.predict([text])[0]
    return "positive" if label == "pos" else "negative"

# ---- Health ----
@app.get("/health")
def health():
    return {"status": "ok"}

# ---- Auth routes ----
@app.post("/auth/register", response_model=Token)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.username == payload.username).first()
    if existing:
        raise HTTPException(status_code=400, detail="Username already taken")
    user = User(username=payload.username, password_hash=get_password_hash(payload.password))
    if db.query(User).count() == 0:
        user.role = "admin"
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.password_hash):
        raise HTTPException(status_code=400, detail="Incorrect username or password")
    token = create_access_token({"sub": user.username})
    return {"access_token": token, "token_type": "bearer"}

@app.get("/me", response_model=MeOut)
def me(current_user: User = Depends(get_current_user)):
    return MeOut(id=current_user.id, username=current_user.username, role=current_user.role, created_at=current_user.created_at)

# ---- Predict & History ----
@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    t = payload.text
    result = {"length": len(t), "words": len(t.split()), "uppercase": t.upper(), "sentiment": predict_sentiment(t)}
    row = Prediction(user_id=current_user.id, text=t, length=result["length"], words=result["words"], uppercase=result["uppercase"], sentiment=result["sentiment"])
    db.add(row); db.commit()
    return result

@app.get("/history", response_model=List[HistoryItem])
def history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = (db.query(Prediction).filter(Prediction.user_id == current_user.id).order_by(Prediction.created_at.desc()).limit(20).all())
    return [HistoryItem(id=r.id, text=r.text, length=r.length, words=r.words, uppercase=r.uppercase, sentiment=r.sentiment, created_at=r.created_at) for r in rows]

# ---- Admin ----
@app.get("/admin/users", response_model=List[UserInfo])
def list_users(admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [UserInfo(id=u.id, username=u.username, role=u.role, created_at=u.created_at) for u in users]

@app.post("/admin/users/{username}/role")
def set_role(username: str, payload: RoleUpdate, admin: User = Depends(require_admin), db: Session = Depends(get_db)):
    if payload.role not in {"user","admin"}:
        raise HTTPException(status_code=400, detail="role must be 'user' or 'admin'")
    user = db.query(User).filter(User.username == username).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = payload.role; db.commit()
    return {"ok": True, "username": user.username, "role": user.role}

# ---- RAG: Chroma + sentence-transformers ----
import chromadb
from sentence_transformers import SentenceTransformer

CHROMA_PATH = os.environ.get("CHROMA_PATH", "chroma_db")
client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = client.get_or_create_collection(name="logs", metadata={"hnsw:space": "cosine"})

_EMBED_MODEL_NAME = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(_EMBED_MODEL_NAME)
    return _embedder

def embed_texts(texts):
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=True).tolist()

def chunk_lines(text: str, lines_per_chunk: int = 50, overlap: int = 5):
    lines = text.splitlines()
    chunks = []
    i = 0
    while i < len(lines):
        start = i
        end = min(len(lines), i + lines_per_chunk)
        chunk = "\n".join(lines[start:end])
        chunks.append((start, end, chunk))
        if end == len(lines):
            break
        i = max(0, end - overlap)
    return chunks

class RAGIngestFile(BaseModel):
    filename: str = Field(..., description="Plain-text log filename")
    text: str = Field(..., description="Full contents of the log file")

class RAGIngestRequest(BaseModel):
    files: List[RAGIngestFile]
    lines_per_chunk: int = 50
    overlap: int = 5

class RAGIngestResponse(BaseModel):
    added_chunks: int

class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 5

class RAGQueryHit(BaseModel):
    filename: str
    start_line: int
    end_line: int
    snippet: str
    score: float

class RAGQueryResponse(BaseModel):
    answer: str
    hits: List[RAGQueryHit]

@app.post("/rag/ingest", response_model=RAGIngestResponse)
def rag_ingest(payload: RAGIngestRequest, current_user: User = Depends(get_current_user)):
    import uuid
    added = 0
    ids, docs, metadatas = [], [], []
    for f in payload.files:
        chunks = chunk_lines(f.text, payload.lines_per_chunk, payload.overlap)
        for (start, end, chunk) in chunks:
            ids.append(f"{current_user.id}:{uuid.uuid4()}")
            docs.append(chunk)
            metadatas.append({"user_id": str(current_user.id), "username": current_user.username, "filename": f.filename, "start_line": start, "end_line": end})
            added += 1
    if docs:
        # batch embeddings
        batch = 64
        for i in range(0, len(docs), batch):
            embs = embed_texts(docs[i:i+batch])
            collection.add(ids=ids[i:i+batch], documents=docs[i:i+batch], metadatas=metadatas[i:i+batch], embeddings=embs)
    return RAGIngestResponse(added_chunks=added)

def extractive_answer(question: str, docs: List[str]) -> str:
    q_terms = [w.lower() for w in question.split() if len(w) > 2]
    answers = []
    for doc in docs:
        for line in doc.splitlines():
            low = line.lower()
            if any(t in low for t in q_terms):
                answers.append(line.strip())
    if not answers and docs:
        first_lines = "\n".join(docs[0].splitlines()[:5])
        return f"Top match snippet:\n{first_lines}"
    return "\n".join(answers[:10]) if answers else "No relevant lines found."

@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(payload: RAGQueryRequest, current_user: User = Depends(get_current_user)):
    q_emb = embed_texts([payload.question])[0]
    res = collection.query(query_embeddings=[q_emb], n_results=max(1, min(payload.top_k, 20)), where={"user_id": str(current_user.id)}, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    hits = []
    for doc, meta, dist in zip(docs, metas, dists):
        hits.append(RAGQueryHit(filename=meta.get("filename","unknown"), start_line=int(meta.get("start_line",0)), end_line=int(meta.get("end_line",0)), snippet=doc[:1000], score=float(1.0 - dist) if dist is not None else 0.0))
    answer = extractive_answer(payload.question, docs)
    return RAGQueryResponse(answer=answer, hits=hits)

# Extra: ingest folder & stats
from glob import glob

class RAGIngestFolderRequest(BaseModel):
    folder_path: str
    pattern: str = "*.txt"
    lines_per_chunk: int = 50
    overlap: int = 5

@app.post("/rag/ingest_folder", response_model=RAGIngestResponse)
def rag_ingest_folder(payload: RAGIngestFolderRequest, current_user: User = Depends(get_current_user)):
    folder = payload.folder_path
    if not os.path.isdir(folder):
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    files = []
    for path in glob(os.path.join(folder, "**", payload.pattern), recursive=True):
        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as fh:
                txt = fh.read()
            files.append({"filename": os.path.relpath(path, folder), "text": txt})
        except Exception:
            continue
    req = RAGIngestRequest(files=[RAGIngestFile(**f) for f in files],
                           lines_per_chunk=payload.lines_per_chunk,
                           overlap=payload.overlap)
    return rag_ingest(req, current_user)

@app.get("/rag/stats")
def rag_stats(current_user: User = Depends(get_current_user)):
    res = collection.get(where={"user_id": str(current_user.id)}, include=["metadatas"])
    by_file = {}
    for m in res.get("metadatas", []):
        fname = m.get("filename","unknown")
        by_file[fname] = by_file.get(fname, 0) + 1
    total = sum(by_file.values())
    top = sorted(by_file.items(), key=lambda x: x[1], reverse=True)[:50]
    return {"total_chunks": total, "files": [{"filename": k, "chunks": v} for k,v in top]}
