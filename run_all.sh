#!/usr/bin/env bash
set -euo pipefail
trap "kill 0" EXIT
uvicorn backend.main:app --reload --port 8000 &
streamlit run frontend/app.py --server.port 8501
wait
