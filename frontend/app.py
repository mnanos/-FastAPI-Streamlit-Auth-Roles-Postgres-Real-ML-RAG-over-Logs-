
import streamlit as st
import httpx

st.set_page_config(page_title="FastAPI + Streamlit (Auth, Roles, Postgres, Real ML, RAG)")

st.title("FastAPI + Streamlit (Auth, Roles, Postgres, Real ML, RAG)")
st.write("Login or register, analyze text, upload or index log files, and semantically search them. Admins can manage roles.")

if "token" not in st.session_state:
    st.session_state.token = None
if "username" not in st.session_state:
    st.session_state.username = None
if "role" not in st.session_state:
    st.session_state.role = None

backend_url = st.sidebar.text_input("Backend URL", "http://localhost:8000")

def get_headers():
    if st.session_state.token:
        return {"Authorization": f"Bearer {st.session_state.token}"}
    return {}

def check_health():
    try:
        r = httpx.get(f"{backend_url}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False

healthy = check_health()
st.sidebar.write("Backend:", "✅ up" if healthy else "⚠️ down")

st.sidebar.subheader("Auth")
tab_login, tab_register = st.sidebar.tabs(["Login", "Register"])
with tab_login:
    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")
    if st.button("Login"):
        try:
            res = httpx.post(f"{backend_url}/auth/login", data={"username": username, "password": password})
            res.raise_for_status()
            data = res.json()
            st.session_state.token = data["access_token"]
            st.session_state.username = username
            me = httpx.get(f"{backend_url}/me", headers={"Authorization": f"Bearer {st.session_state.token}"})
            if me.status_code == 200:
                st.session_state.role = me.json().get("role")
            st.success("Logged in!")
        except Exception as e:
            st.error(f"Login failed: {e}")

with tab_register:
    r_user = st.text_input("New username", key="reg_user")
    r_pass = st.text_input("New password", type="password", key="reg_pass")
    if st.button("Register"):
        try:
            res = httpx.post(f"{backend_url}/auth/register", json={"username": r_user, "password": r_pass})
            res.raise_for_status()
            data = res.json()
            st.session_state.token = data["access_token"]
            st.session_state.username = r_user
            me = httpx.get(f"{backend_url}/me", headers={"Authorization": f"Bearer {st.session_state.token}"})
            if me.status_code == 200:
                st.session_state.role = me.json().get("role")
            st.success("Registered and logged in!")
        except Exception as e:
            st.error(f"Registration failed: {e}")

if st.session_state.token:
    st.success(f"Authenticated as **{st.session_state.username}** (role: **{st.session_state.role}**)")

st.header("Analyze text")
text = st.text_area("Enter text", "I absolutely love this!")
if st.button("Analyze"):
    if not st.session_state.token:
        st.warning("Please login first.")
    else:
        try:
            res = httpx.post(f"{backend_url}/predict", json={"text": text}, headers=get_headers(), timeout=20)
            res.raise_for_status()
            st.subheader("Result")
            st.json(res.json())
        except Exception as e:
            st.error(f"Request failed: {e}")

st.header("History (latest 20)")
if st.session_state.token:
    try:
        res = httpx.get(f"{backend_url}/history", headers=get_headers(), timeout=20)
        res.raise_for_status()
        items = res.json()
        if items:
            for it in items:
                with st.expander(f"#{it['id']} • {it['sentiment']} • {it['created_at']}"):
                    st.write(f"**Text:** {it['text']}")
                    st.write({k: it[k] for k in ['length','words','uppercase','sentiment']})
        else:
            st.info("No history yet. Analyze something!")
    except Exception as e:
        st.error(f"Failed to load history: {e}")
else:
    st.info("Login to see your history.")

st.header("Admin panel")
if st.session_state.token and st.session_state.role == "admin":
    st.success("Admin access granted.")
    if st.button("Refresh users"):
        st.session_state["_users"] = None
    try:
        res = httpx.get(f"{backend_url}/admin/users", headers=get_headers(), timeout=20)
        res.raise_for_status()
        users = res.json()
        for u in users:
            cols = st.columns([3,1,1])
            cols[0].write(f"**{u['username']}** (role: {u['role']}) • created: {u['created_at']}")
            new_role = cols[1].selectbox("Set role", ["user","admin"], index=0 if u["role"]=="user" else 1, key=f"role_{u['id']}")
            if cols[2].button("Apply", key=f"apply_{u['id']}"):
                try:
                    r = httpx.post(f"{backend_url}/admin/users/{u['username']}/role", json={"role": new_role}, headers=get_headers())
                    if r.status_code == 200:
                        st.success(f"Updated {u['username']} to {new_role}")
                    else:
                        st.error(r.text)
                except Exception as e:
                    st.error(f"Update failed: {e}")
    except Exception as e:
        st.error(f"Could not load users: {e}")
else:
    st.info("Admin-only area. (First registered user becomes admin.)")

# --- RAG over logs ---
st.header("RAG: Search your log files")
st.caption("Upload plain-text .txt logs, index them into a local vector store, then ask questions semantically.")

if st.session_state.token:
    with st.expander("Ingest logs (.txt)"):
        uploaded = st.file_uploader("Select one or more .txt files", type=["txt"], accept_multiple_files=True)
        lines_per_chunk = st.number_input("Lines per chunk", min_value=10, max_value=500, value=50, step=10)
        overlap = st.number_input("Overlap (lines)", min_value=0, max_value=100, value=5, step=1)
        if st.button("Ingest selected files"):
            files_payload = []
            for f in uploaded or []:
                try:
                    text = f.read().decode("utf-8", errors="ignore")
                    files_payload.append({"filename": f.name, "text": text})
                except Exception as e:
                    st.error(f"Failed to read {f.name}: {e}")
            if files_payload:
                try:
                    res = httpx.post(f"{backend_url}/rag/ingest",
                                     json={"files": files_payload, "lines_per_chunk": int(lines_per_chunk), "overlap": int(overlap)},
                                     headers=get_headers(), timeout=120)
                    res.raise_for_status()
                    st.success(f"Ingested {res.json()['added_chunks']} chunks.")
                except Exception as e:
                    st.error(f"Ingest failed: {e}")
            else:
                st.info("No files to ingest.")

    st.subheader("Query logs")
    q = st.text_input("Ask a question about your logs", "Where are the errors happening?")
    k = st.slider("Top K", 1, 20, 5)
    if st.button("Search"):
        try:
            res = httpx.post(f"{backend_url}/rag/query", json={"question": q, "top_k": int(k)}, headers=get_headers(), timeout=60)
            res.raise_for_status()
            data = res.json()
            st.write("**Answer (extractive):**")
            st.code(data["answer"])
            st.write("**Top hits:**")
            for hit in data["hits"]:
                with st.expander(f"{hit['filename']} • lines {hit['start_line']}-{hit['end_line']} • score {hit['score']:.3f}"):
                    st.code(hit["snippet"])
        except Exception as e:
            st.error(f"Search failed: {e}")

    # Folder ingestion & stats
    st.subheader("Ingest from a folder")
    col1, col2 = st.columns([3,1])
    folder_path = col1.text_input("Folder path on the API server", value="")
    pattern = col2.text_input("Glob pattern", value="*.txt")
    lpc = st.number_input("Lines per chunk (folder)", min_value=10, max_value=500, value=50, step=10)
    ovl = st.number_input("Overlap (folder)", min_value=0, max_value=100, value=5, step=1)
    if st.button("Ingest folder"):
        try:
            res = httpx.post(f"{backend_url}/rag/ingest_folder",
                             json={"folder_path": folder_path, "pattern": pattern,
                                   "lines_per_chunk": int(lpc), "overlap": int(ovl)},
                             headers=get_headers(), timeout=600)
            res.raise_for_status()
            st.success(f"Ingested {res.json()['added_chunks']} chunks from folder.")
        except Exception as e:
            st.error(f"Folder ingest failed: {e}")

    st.subheader("RAG stats")
    try:
        res = httpx.get(f"{backend_url}/rag/stats", headers=get_headers(), timeout=60)
        res.raise_for_status()
        stats = res.json()
        st.write(f"Total chunks: {stats.get('total_chunks',0)}")
        files = stats.get("files", [])
        if files:
            import pandas as pd
            df = pd.DataFrame(files)
            st.dataframe(df)
            st.bar_chart(df.set_index("filename")["chunks"])
        else:
            st.info("No RAG data yet.")
    except Exception as e:
        st.error(f"Failed to load stats: {e}")
else:
    st.info("Login to use RAG over logs.")
