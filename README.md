# FastAPI + Streamlit (Auth, Roles, Postgres, Real ML, RAG over Logs)

This repo gives you a batteries‑included starter that runs end‑to‑end:

- FastAPI backend with JWT auth and **role‑based access** (user/admin)
- SQLite or **Postgres** (via Docker Compose)
- Sentiment analysis: **Transformers** pipeline if available, **scikit‑learn** fallback
- **RAG over plain‑text logs** using **ChromaDB** + **sentence‑transformers**
- Streamlit UI: login/register, predict, history, RAG upload/folder ingest, and **stats**

## Quick start (local)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export SECRET_KEY="a_very_long_random_string"
# optional:
export USE_TRANSFORMERS=true
export CHROMA_PATH="chroma_db"
export EMBED_MODEL="sentence-transformers/all-MiniLM-L6-v2"
uvicorn backend.main:app --reload --port 8000
# new terminal
streamlit run frontend/app.py --server.port 8501
```

Open http://localhost:8501

## Docker Compose (with Postgres)

```bash
docker compose up
```

Services:
- `db` (Postgres 16)
- `api` (FastAPI) with `DATABASE_URL=postgresql+psycopg2://app:app@db:5432/appdb`
- `ui` (Streamlit)

## RAG over logs

- `POST /rag/ingest` — ingest uploaded `.txt` logs (per‑user)
- `POST /rag/ingest_folder` — ingest all matching files from a folder on the **API server**
  - body:
    ```json
    { "folder_path": "/var/log/app", "pattern": "*.txt", "lines_per_chunk": 50, "overlap": 5 }
    ```
- `POST /rag/query` — semantic search
- `GET /rag/stats` — totals + chunks per file

### How it works
- Chunks by **lines** with **overlap**
- Embeds with `sentence-transformers` (configurable via `EMBED_MODEL`)
- Persists vectors in `./chroma_db`
- Metadata per chunk: `{user_id, username, filename, start_line, end_line}` and filtered per‑user at query time

## Auth & roles

- First registered user is **admin**
- Admin panel in Streamlit lets you change roles
- Admin API: `GET /admin/users`, `POST /admin/users/{username}/role`

## Files

```
backend/main.py
frontend/app.py
requirements.txt
docker-compose.yml
run_all.sh
README.md
```
