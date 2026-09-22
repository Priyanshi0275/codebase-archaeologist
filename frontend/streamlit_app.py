import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Legacy Codebase Archaeologist", layout="centered")
st.title("🪲 Legacy Codebase Archaeologist")
st.caption("Ask why old code exists — three agents dig through commit history to answer.")

with st.expander("Step 1: Ingest a repo", expanded=True):
    repo_input = st.text_input(
        "Local path or public GitHub URL",
        placeholder="https://github.com/owner/repo or /path/to/local/repo",
    )
    max_commits = st.slider("Max commits to ingest", 20, 500, 150)
    if st.button("Ingest repo"):
        with st.spinner("Cloning + indexing commit history..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ingest",
                    json={"repo_path_or_url": repo_input, "max_commits": max_commits},
                    timeout=300,
                )
                resp.raise_for_status()
                st.success(resp.json())
            except Exception as e:
                st.error(f"Ingest failed: {e}")

st.divider()

st.subheader("Step 2: Ask about the code")
question = st.text_input(
    "e.g. Why does the retry logic in payment.py exist?",
)
if st.button("Investigate") and question:
    with st.spinner("Agents are digging through commit history..."):
        try:
            resp = requests.post(f"{API_URL}/query", json={"question": question}, timeout=180)
            resp.raise_for_status()
            data = resp.json()

            st.markdown("### 🕵️ Historian")
            st.write(data["historian"])

            st.markdown("### ⚠️ Dependency Risk Analyst")
            st.write(data["dependency_mapper"])

            st.markdown("### 🛠️ Refactor Advisor")
            st.write(data["refactor_drafter"])
        except Exception as e:
            st.error(f"Query failed: {e}")
