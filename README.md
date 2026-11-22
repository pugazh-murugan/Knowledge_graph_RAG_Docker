# Knowledge Graph RAG with Docker

A Dockerized knowledge graph retrieval-augmented generation (RAG) application.
It loads PDF or text documents, extracts entities and relationships with OpenAI,
stores the result in NetworkX, and provides a FastAPI backend and Streamlit UI.

## Architecture

```text
PDF/TXT document
      |
      v
Document chunks
      |
      v
OpenAI entity + relationship extraction
      |
      v
NetworkX MultiDiGraph --> graph_data/kg.json
      |
      +--> FastAPI backend --> Graph queries, paths, and RAG answers
                              |
                              v
                         Streamlit UI
```

The application is divided into two containers:

- `backend/`: FastAPI API and the document-to-graph RAG pipeline.
- `frontend/`: Streamlit pages for chat, graph exploration, and graph builds.
- `data/`: Source PDF or text files.
- `graph_data/`: Generated `kg.json` graph file.

## Prerequisites

- Docker with Docker Compose
- An OpenAI API key

## Quick start

1. Create a `.env` file in the repository root:

```dotenv
OPENAI_API_KEY=your_openai_api_key
BACKEND_PORT=8001
FRONTEND_PORT=8502
```

2. Place a PDF or text document in `data/`, or keep the included
   `spotify_web_app_architecture.pdf`.

3. Start the containers:

```powershell
docker compose up --build
```

4. Open the application:

- Streamlit UI: http://localhost:8502
- FastAPI API: http://localhost:8001
- Interactive API docs: http://localhost:8001/docs

5. Open the **Build** page and select the source document to create the graph.

## Docker services

| Service | Container | Host port | Purpose |
| --- | --- | --- | --- |
| `backend` | `kg-rag-backend` | `8001` by default | Graph building and RAG API |
| `frontend` | `kg-rag-frontend` | `8502` by default | Streamlit user interface |

The backend mounts `./data` and `./graph_data` into the container. This preserves
source documents and the generated graph when containers are recreated.

## API overview

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Container health check |
| `GET` | `/graph/status` | Graph load/build status |
| `GET` | `/graph/stats` | Node, edge, and connection statistics |
| `GET` | `/graph` | Complete graph for visualization |
| `GET` | `/entities` | List all graph entities |
| `GET` | `/entity?name=...` | Entity details and relationships |
| `GET` | `/path?source=...&target=...` | Shortest path between entities |
| `POST` | `/ask` | Answer a question using graph facts |
| `POST` | `/graph/build` | Build or rebuild the graph in the background |

Example request:

```powershell
$body = @{
    question = "How does the catalog service use Redis?"
    depth = 2
} | ConvertTo-Json

Invoke-RestMethod -Method Post `
  -Uri "http://localhost:8001/ask" `
  -ContentType "application/json" `
  -Body $body
```

## Local development

Docker is the recommended setup. For local processes, run the two services
from the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
uvicorn main:app --app-dir backend --reload
```

Start the Streamlit frontend in another terminal:

```powershell
$env:BACKEND_URL = "http://localhost:8000"
streamlit run frontend\streamlit_app.py
```

Set `OPENAI_API_KEY` in `.env` before building or querying the graph.

## Standalone graph scripts

The repository also contains earlier standalone pipeline modules:

- `kg_rag_full.py`: Complete pipeline and command-line chat.
- `kg_rag.py`: Document loading and chunking.
- `entities.py`: Entity extraction and merging.
- `relationships.py`: Relationship extraction and merging.
- `graph_store.py`: NetworkX graph construction and JSON persistence.
- `graph_query.py`: Local graph queries and graph-grounded answers.

Example:

```powershell
python kg_rag_full.py --rebuild --limit 3
python kg_rag_full.py --pdf data\other.pdf --rebuild
```

## Project structure

```text
.
|-- backend/
|   |-- main.py          # FastAPI application and API routes
|   |-- kg_rag_full.py   # Build, index, and RAG implementation
|   `-- requirements.txt
|-- frontend/
|   |-- streamlit_app.py # Streamlit navigation shell
|   |-- api.py           # FastAPI HTTP client
|   |-- app_pages/       # Chat, explorer, and build pages
|   `-- requirements.txt
|-- data/
|   `-- spotify_web_app_architecture.pdf
|-- graph_data/
|   `-- kg.json
|-- docker-compose.yml
`-- README.md
```

## Configuration

| Variable | Default | Used by |
| --- | --- | --- |
| `OPENAI_API_KEY` | Required | Backend graph building and RAG |
| `PDF_PATH` | `data/spotify_web_app_architecture.pdf` | Backend pipeline |
| `GRAPH_PATH` | `graph_data/kg.json` | Backend graph persistence |
| `BACKEND_PORT` | `8001` | Docker host port |
| `FRONTEND_PORT` | `8502` | Docker host port |
| `BACKEND_URL` | `http://localhost:8000` locally | Streamlit frontend |

Keep `.env` private and do not commit API keys.

