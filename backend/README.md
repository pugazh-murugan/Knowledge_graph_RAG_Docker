# Backend

The backend is a FastAPI service that builds and queries a NetworkX knowledge
graph. It exposes graph status and exploration endpoints, plus a Graph RAG
question-answering endpoint.

## Run locally

From the repository root:

```powershell
uvicorn main:app --app-dir backend --reload
```

The API is then available at http://localhost:8000. Interactive documentation
is available at http://localhost:8000/docs.

Set `OPENAI_API_KEY` in the repository `.env` file before graph builds or
`POST /ask` requests.

## Key files

- `main.py`: FastAPI routes, application state, health checks, and background builds.
- `kg_rag_full.py`: PDF/text loading, chunking, entity extraction, relationship
  extraction, graph storage, semantic indexing, and graph-grounded answers.
- `requirements.txt`: Python dependencies.
- `Dockerfile`: Container image and `uvicorn` startup command.

## Main endpoints

- `GET /health`
- `GET /graph/status`
- `GET /graph/stats`
- `GET /graph`
- `GET /entities`
- `GET /entity?name=<name>`
- `GET /path?source=<name>&target=<name>`
- `POST /ask`
- `POST /graph/build`

Graph builds run as background tasks. Poll `GET /graph/status` until
`building` becomes `false`. A failed build reports `last_error`.
