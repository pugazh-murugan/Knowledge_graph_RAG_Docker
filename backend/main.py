"""
FastAPI backend for the Knowledge Graph RAG pipeline.

Run locally (from the repo root, so data/ and graph_data/ resolve):
    uvicorn main:app --app-dir backend --reload

Docs:  http://localhost:8000/docs
"""

import os
import threading
from contextlib import asynccontextmanager

import networkx as nx
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

import kg_rag_full as kg


# ----------------------------------------------------------------------
# App state (one graph per process, shared by all requests)
# ----------------------------------------------------------------------
class State:
    G: nx.MultiDiGraph | None = None
    index = None                       # (node ids, embedding matrix) for semantic search
    building: bool = False
    last_error: str | None = None
    index_lock = threading.Lock()
    build_lock = threading.Lock()


state = State()
DATA_DIR = os.path.dirname(kg.PDF_PATH) or "."


def load_graph_into_state() -> bool:
    """Load the saved graph (offline, no API calls). The embedding index is built lazily."""
    if not os.path.exists(kg.GRAPH_PATH):
        return False
    state.G = kg.load_graph()
    state.index = None
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_graph_into_state()
    yield


app = FastAPI(title="Knowledge Graph RAG API", version="1.0.0", lifespan=lifespan)


def require_graph() -> nx.MultiDiGraph:
    if state.G is None:
        raise HTTPException(503, "Knowledge graph not built yet. Call POST /graph/build first.")
    return state.G


def require_index():
    """Embed all nodes on first use (one OpenAI embeddings call), then reuse."""
    G = require_graph()
    with state.index_lock:
        if state.index is None:
            if G.number_of_nodes() == 0:
                raise HTTPException(503, "The graph is empty.")
            state.index = kg.build_node_index(G)
        return state.index


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------
class AskRequest(BaseModel):
    question: str = Field(min_length=1, description="Question about the documents")
    depth: int = Field(1, ge=1, le=3, description="Hops of graph context around matched nodes")


class AskResponse(BaseModel):
    answer: str
    linked: list[str]      # entities found by name in the question
    semantic: list[str]    # entities found by embedding similarity
    facts: list[str]       # graph facts the LLM was allowed to use


class BuildRequest(BaseModel):
    pdf: str | None = Field(None, description="File name inside the data folder (.pdf or .txt)")
    limit: int | None = Field(None, ge=1, description="Only process the first N chunks (testing)")


# ----------------------------------------------------------------------
# Health / status
# ----------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph/status")
def graph_status():
    G = state.G
    return {
        "loaded": G is not None,
        "building": state.building,
        "last_error": state.last_error,
        "nodes": G.number_of_nodes() if G else 0,
        "edges": G.number_of_edges() if G else 0,
    }


# ----------------------------------------------------------------------
# Graph exploration (no LLM calls)
# ----------------------------------------------------------------------
@app.get("/graph/stats")
def graph_stats():
    G = require_graph()
    top = sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "isolated_nodes": len(list(nx.isolates(G))),
        "top_connected": [
            {"name": G.nodes[n]["name"], "type": G.nodes[n]["type"], "connections": d}
            for n, d in top
        ],
    }


@app.get("/graph")
def graph_full():
    """Whole graph as nodes + edges (for visualisation)."""
    G = require_graph()
    return {
        "nodes": [
            {"id": n, "name": d["name"], "type": d["type"],
             "description": d["description"], "degree": G.degree(n)}
            for n, d in G.nodes(data=True)
        ],
        "edges": [
            {"source": u, "target": v, "relation": d["relation"], "evidence": d["evidence"]}
            for u, v, d in G.edges(data=True)
        ],
    }


@app.get("/entities")
def list_entities():
    G = require_graph()
    items = [{"name": d["name"], "type": d["type"]} for _, d in G.nodes(data=True)]
    return sorted(items, key=lambda x: x["name"].lower())


@app.get("/entity")
def get_entity(name: str = Query(..., min_length=1)):
    G = require_graph()
    nid = kg.find_node(G, name)
    if nid is None:
        raise HTTPException(404, f"Entity '{name}' not found.")
    out, inc = kg.get_neighbors(G, nid)
    return {
        "name": G.nodes[nid]["name"],
        "type": G.nodes[nid]["type"],
        "description": G.nodes[nid]["description"],
        "outgoing": [{"relation": r, "target": t} for r, t in out],
        "incoming": [{"source": s, "relation": r} for s, r in inc],
    }


@app.get("/path")
def get_path(source: str = Query(..., min_length=1), target: str = Query(..., min_length=1)):
    G = require_graph()
    path = kg.find_path(G, source, target)
    if path is None:
        raise HTTPException(404, f"No path between '{source}' and '{target}'.")
    return {"path": path, "length": len(path) - 1}


# ----------------------------------------------------------------------
# Graph RAG question answering (uses OpenAI)
# ----------------------------------------------------------------------
@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    G = require_graph()
    index = require_index()
    return kg.ask_detailed(G, req.question.strip(), index, req.depth)


# ----------------------------------------------------------------------
# Build / rebuild the graph (uses OpenAI, runs in the background)
# ----------------------------------------------------------------------
def run_build(pdf_path: str, limit: int | None):
    try:
        kg.build_and_save(pdf_path, limit)
        load_graph_into_state()
        state.last_error = None
    except Exception as exc:  # surfaced via GET /graph/status
        state.last_error = f"{type(exc).__name__}: {exc}"
    finally:
        state.building = False
        state.build_lock.release()


@app.post("/graph/build", status_code=202)
def build_graph(req: BuildRequest, background: BackgroundTasks):
    if req.pdf:
        if os.path.basename(req.pdf) != req.pdf:
            raise HTTPException(400, "Give a file name only, not a path.")
        if not req.pdf.lower().endswith((".pdf", ".txt")):
            raise HTTPException(400, "Only .pdf and .txt files are supported.")
        pdf_path = os.path.join(DATA_DIR, req.pdf)
    else:
        pdf_path = kg.PDF_PATH
    if not os.path.isfile(pdf_path):
        raise HTTPException(404, f"File not found: {pdf_path}")

    if not state.build_lock.acquire(blocking=False):
        raise HTTPException(409, "A build is already running.")
    state.building = True
    state.last_error = None
    background.add_task(run_build, pdf_path, req.limit)
    return {"status": "started", "file": pdf_path, "limit": req.limit}
