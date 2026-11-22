# Frontend

The frontend is a Streamlit application that communicates with the FastAPI
backend. It provides chat, graph exploration, and graph build interfaces.

## Run locally

From the repository root:

```powershell
$env:BACKEND_URL = "http://localhost:8000"
streamlit run frontend\streamlit_app.py
```

Open http://localhost:8501 unless Streamlit prints a different URL.

When running through Docker Compose, `BACKEND_URL` is
`http://backend:8000` and the UI is published on host port `8502` by default.

## Key files

- `streamlit_app.py`: Page navigation and backend status in the sidebar.
- `api.py`: Thin HTTP client for FastAPI endpoints.
- `app_pages/chat.py`: Graph RAG chat interface.
- `app_pages/explorer.py`: Graph statistics, entity lookup, and shortest paths.
- `app_pages/build.py`: Document graph build and build status.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image and Streamlit startup command.
