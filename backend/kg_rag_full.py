"""
Knowledge Graph RAG - LangChain + NetworkX (single file)

Pipeline:
  Step 1  Document loader        (PDF / TXT -> chunks)
  Step 2  Entity extraction      (LLM, structured output)
  Step 3  Relationship mapping   (LLM, structured output)
  Step 4  Graph storage          (NetworkX MultiDiGraph, saved as JSON)
  Step 5  Graph query + RAG      (entity linking + semantic fallback + LLM answer)

Install:
  pip install langchain langchain-community langchain-text-splitters langchain-openai
  pip install networkx pypdf python-dotenv numpy pydantic

.env file:
  OPENAI_API_KEY=your_key_here

Usage:
  python kg_rag_full.py                  # build graph if missing, then chat
  python kg_rag_full.py --rebuild        # force rebuild the graph, then chat
  python kg_rag_full.py --rebuild --limit 3   # quick test on first 3 chunks
  python kg_rag_full.py --pdf data/other.pdf --rebuild
"""

import os
import json
import argparse
import difflib
from typing import Literal

import numpy as np
import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
PDF_PATH = os.getenv("PDF_PATH", "data/spotify_web_app_architecture.pdf")
GRAPH_PATH = os.getenv("GRAPH_PATH", "graph_data/kg.json")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
LLM_MODEL = "gpt-4o-mini"
EMBED_MODEL = "text-embedding-3-small"

llm = ChatOpenAI(model=LLM_MODEL, temperature=0)
embedder = OpenAIEmbeddings(model=EMBED_MODEL)


# ======================================================================
# STEP 1: Document loader
# ======================================================================
def load_documents(file_path: str):
    """Load a .txt or .pdf file into LangChain Document objects."""
    if file_path.lower().endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(documents)


# ======================================================================
# STEP 2: Entity extraction
# ======================================================================
class Entity(BaseModel):
    name: str = Field(description="Name of the entity, e.g. 'PostgreSQL'")
    type: Literal["Service", "Technology", "Database", "Feature", "Page", "Concept"]
    description: str = Field(description="One short sentence about this entity")


class EntityList(BaseModel):
    entities: list[Entity]


entity_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract entities for a knowledge graph from technical documents. "
     "Extract only meaningful entities (services, technologies, databases, features, pages, concepts). "
     "Use the exact name as written in the text. Do not invent entities. "
     "Do NOT extract section headings, database table names, or simple attributes "
     "(like 'language' or 'theme') as entities."),
    ("human", "Text:\n{text}"),
])
entity_chain = entity_prompt | llm.with_structured_output(EntityList)


def extract_entities(chunks):
    results = []
    for i, chunk in enumerate(chunks):
        print(f"  [entities] chunk {i + 1}/{len(chunks)}")
        output = entity_chain.invoke({"text": chunk.page_content})
        results.append({
            "chunk_id": i,
            "text": chunk.page_content,
            "entities": output.entities,
        })
    return results


def merge_entities(results):
    """Same name (case-insensitive) = same entity."""
    merged = {}
    for r in results:
        for e in r["entities"]:
            key = e.name.strip().lower()
            if key not in merged:
                merged[key] = e
    return list(merged.values())


# ======================================================================
# STEP 3: Relationship mapping
# ======================================================================
class Relationship(BaseModel):
    source: str = Field(description="Source entity name, exactly from the entity list")
    target: str = Field(description="Target entity name, exactly from the entity list")
    relation: Literal[
        "USES", "STORES_DATA_IN", "CACHES_IN", "PUBLISHES_TO", "DEPENDS_ON",
        "PART_OF", "PROVIDES", "HANDLES", "INTEGRATES_WITH", "DEPLOYED_ON",
        "HAS_OPTION", "ALTERNATIVE_TO", "RELATED_TO",
    ]
    evidence: str = Field(description="Short phrase from the text that supports this relationship")


class RelationshipList(BaseModel):
    relationships: list[Relationship]


relationship_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract relationships between entities for a knowledge graph. "
     "Use ONLY entity names from the provided entity list, spelled exactly the same. "
     "Only extract relationships that are stated or clearly implied by the text. "
     "Pick the most specific relation type; use RELATED_TO only as a last resort. "
     "When the text lists options like 'CDN (CloudFront/Fastly)', use HAS_OPTION from the "
     "category to each option (CDN HAS_OPTION CloudFront). Use ALTERNATIVE_TO only between "
     "two options."),
    ("human", "Entities:\n{entity_names}\n\nText:\n{text}"),
])
relationship_chain = relationship_prompt | llm.with_structured_output(RelationshipList)


def extract_relationships(entity_results):
    all_rels = []
    for r in entity_results:
        names = [e.name for e in r["entities"]]
        if len(names) < 2:
            continue  # need at least 2 entities for a relationship
        print(f"  [relations] chunk {r['chunk_id'] + 1}/{len(entity_results)}")
        output = relationship_chain.invoke({
            "entity_names": "\n".join(f"- {n}" for n in names),
            "text": r["text"],
        })
        valid = {n.lower() for n in names}
        for rel in output.relationships:
            # drop anything the LLM invented that isn't in our entity list
            if rel.source.lower() in valid and rel.target.lower() in valid:
                all_rels.append(rel)
    return all_rels


def merge_relationships(rels):
    seen, unique = set(), []
    for rel in rels:
        key = (rel.source.lower(), rel.relation, rel.target.lower())
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique


# ======================================================================
# STEP 4: Graph storage (NetworkX)
# ======================================================================
def node_id(name: str) -> str:
    """Normalize names so 'Redis' and 'redis' become the same node."""
    return name.strip().lower()


def build_graph(entities, relationships) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()  # directed + allows several edges between the same nodes

    for e in entities:
        G.add_node(node_id(e.name), name=e.name, type=e.type, description=e.description)

    for r in relationships:
        s, t = node_id(r.source), node_id(r.target)
        for nid, label in ((s, r.source), (t, r.target)):
            if nid not in G:
                G.add_node(nid, name=label, type="Unknown", description="")
        G.add_edge(s, t, relation=r.relation, evidence=r.evidence)

    return G


def save_graph(G, path=GRAPH_PATH):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = nx.node_link_data(G, edges="edges")  # needs networkx >= 3.4
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Graph saved to {path}")


def load_graph(path=GRAPH_PATH) -> nx.MultiDiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


def build_and_save(pdf_path=PDF_PATH, limit=None):
    """Run Steps 1-4 once and save the graph."""
    print("Step 1: loading and splitting document...")
    docs = load_documents(pdf_path)
    chunks = split_documents(docs)
    if limit:
        chunks = chunks[:limit]
    print(f"  {len(chunks)} chunk(s)")

    print("Step 2: extracting entities...")
    entity_results = extract_entities(chunks)
    entities = merge_entities(entity_results)
    print(f"  {len(entities)} unique entities")

    print("Step 3: extracting relationships...")
    relationships = merge_relationships(extract_relationships(entity_results))
    print(f"  {len(relationships)} unique relationships")

    print("Step 4: building and saving graph...")
    G = build_graph(entities, relationships)
    save_graph(G)
    return G


def print_graph_stats(G):
    print(f"\nNodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}   "
          f"Isolated nodes: {len(list(nx.isolates(G)))}")
    print("Top 5 most connected nodes:")
    for nid, degree in sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {G.nodes[nid]['name']} ({G.nodes[nid]['type']}) - {degree} connections")


# ======================================================================
# STEP 5a: Plain NetworkX queries (no LLM)
# ======================================================================
def find_node(G, name, cutoff=0.85):
    """Exact (case-insensitive) match, then strict fuzzy match. Returns node id or None."""
    lookup = {data["name"].lower(): nid for nid, data in G.nodes(data=True)}
    key = name.strip().lower()
    if key in lookup:
        return lookup[key]
    close = difflib.get_close_matches(key, lookup.keys(), n=1, cutoff=cutoff)
    return lookup[close[0]] if close else None


def get_neighbors(G, nid):
    out = [(d["relation"], G.nodes[v]["name"]) for _, v, d in G.out_edges(nid, data=True)]
    inc = [(G.nodes[u]["name"], d["relation"]) for u, _, d in G.in_edges(nid, data=True)]
    return out, inc


def find_path(G, a, b):
    """Shortest connection between two entities (edge direction ignored)."""
    na, nb = find_node(G, a), find_node(G, b)
    if not na or not nb:
        return None
    try:
        path = nx.shortest_path(G.to_undirected(), na, nb)
        return [G.nodes[n]["name"] for n in path]
    except nx.NetworkXNoPath:
        return None


# ======================================================================
# STEP 5b: Graph RAG
# ======================================================================
class QuestionEntities(BaseModel):
    entities: list[str]


question_prompt = ChatPromptTemplate.from_messages([
    ("system", "List the technologies, services, databases or features mentioned "
               "in the user's question. Return only names, as written."),
    ("human", "{question}"),
])
question_chain = question_prompt | llm.with_structured_output(QuestionEntities)

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the knowledge graph facts below. "
               "If the facts are not enough, say you don't know. Be concise."),
    ("human", "Facts:\n{context}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | llm


def build_node_index(G):
    """Embed every node as 'name (type): description' for semantic search."""
    ids = list(G.nodes)
    texts = [
        f"{G.nodes[n]['name']} ({G.nodes[n]['type']}): {G.nodes[n]['description']}"
        for n in ids
    ]
    vecs = np.array(embedder.embed_documents(texts))
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return ids, vecs


def semantic_nodes(question, index, k=3):
    ids, vecs = index
    q = np.array(embedder.embed_query(question))
    q /= np.linalg.norm(q)
    top = np.argsort(vecs @ q)[::-1][:k]
    return [ids[i] for i in top]


def get_facts(G, nid, depth=1, max_facts=30):
    """Collect facts around a node (within `depth` hops) as readable lines."""
    sub = nx.ego_graph(G, nid, radius=depth, undirected=True)
    facts = [f"Entity: {G.nodes[nid]['name']} ({G.nodes[nid]['type']}) - {G.nodes[nid]['description']}"]
    for u, v, d in sub.edges(data=True):
        facts.append(
            f"({G.nodes[u]['name']}) --{d['relation']}--> ({G.nodes[v]['name']}) "
            f"| source text: {d['evidence']}"
        )
    return facts[:max_facts]


def ask_detailed(G, question, index, depth=1) -> dict:
    """Full Graph RAG run. Returns the answer plus what was retrieved (used by the API)."""
    # 1. exact entity linking
    names = question_chain.invoke({"question": question}).entities
    linked = [n for n in (find_node(G, x) for x in names) if n]

    # 2. semantic search (takes more seeds when linking found nothing)
    semantic = semantic_nodes(question, index, k=2 if linked else 3)
    seeds = list(dict.fromkeys(linked + semantic))

    # 3. gather facts around the seed nodes
    facts = []
    for nid in seeds:
        facts.extend(get_facts(G, nid, depth))
    facts = list(dict.fromkeys(facts))[:60]

    # 4. LLM answers from the facts only
    answer = answer_chain.invoke(
        {"context": "\n".join(facts), "question": question}
    ).content

    return {
        "answer": answer,
        "linked": [G.nodes[n]["name"] for n in linked],
        "semantic": [G.nodes[n]["name"] for n in semantic],
        "facts": facts,
    }


def ask(G, question, index, depth=1):
    """CLI helper: same as ask_detailed, but prints the retrieval info and returns the answer."""
    result = ask_detailed(G, question, index, depth)
    print(f"  linked  : {result['linked']}")
    print(f"  semantic: {result['semantic']}")
    return result["answer"]


# ======================================================================
# Main
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Knowledge Graph RAG (LangChain + NetworkX)")
    parser.add_argument("--pdf", default=PDF_PATH, help="path to .pdf or .txt file")
    parser.add_argument("--rebuild", action="store_true", help="rebuild the graph even if it exists")
    parser.add_argument("--limit", type=int, default=None, help="only process the first N chunks (testing)")
    args = parser.parse_args()

    if args.rebuild or not os.path.exists(GRAPH_PATH):
        G = build_and_save(args.pdf, args.limit)
    else:
        print(f"Loading existing graph from {GRAPH_PATH} (use --rebuild to rebuild)")

    G = load_graph()
    print_graph_stats(G)

    print("\n=== Plain graph queries ===")
    nid = find_node(G, "Catalog Service")
    if nid:
        out, _ = get_neighbors(G, nid)
        print("Catalog Service outgoing:", out[:4], "...")
    print("Path Catalog Service -> Redis:", find_path(G, "Catalog Service", "Redis"))
    print("Path Kubernetes -> AKS:", find_path(G, "Kubernetes", "AKS"))

    print("\n=== Graph RAG chat (type 'exit' to stop) ===")
    index = build_node_index(G)
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        print("Answer:", ask(G, q, index))


if __name__ == "__main__":
    main()
