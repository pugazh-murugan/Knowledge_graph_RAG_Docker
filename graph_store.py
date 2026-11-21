import os
import json
import networkx as nx

from kg_rag import load_documents, split_documents
from entities import extract_entities, merge_entities
from relationships import extract_relationships, merge_relationships

GRAPH_PATH = "graph_data/kg.json"


def node_id(name: str) -> str:
    """Normalize names so 'Redis' and 'redis' become the same node."""
    return name.strip().lower()


# 1. Build the graph
def build_graph(entities, relationships) -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()

    for e in entities:
        G.add_node(
            node_id(e.name),
            name=e.name,
            type=e.type,
            description=e.description,
        )

    for r in relationships:
        s, t = node_id(r.source), node_id(r.target)

        # safety: make sure both endpoints exist as nodes
        for nid, label in ((s, r.source), (t, r.target)):
            if nid not in G:
                G.add_node(nid, name=label, type="Unknown", description="")

        G.add_edge(s, t, relation=r.relation, evidence=r.evidence)

    return G


# 2. Save / load (JSON file)
def save_graph(G, path=GRAPH_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Graph saved to {path}")


def load_graph(path=GRAPH_PATH) -> nx.MultiDiGraph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return nx.node_link_graph(data, edges="edges")


# 3. Build the whole pipeline once
def build_and_save(pdf_path="data/spotify_web_app_architecture.pdf"):
    docs = load_documents(pdf_path)
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)

    entity_results = extract_entities(chunks)          # all 15 chunks
    entities = merge_entities(entity_results)
    relationships = merge_relationships(extract_relationships(entity_results))

    G = build_graph(entities, relationships)
    save_graph(G)
    return G


if __name__ == "__main__":
    G = build_and_save()

    # Reload from disk to prove saving works
    G = load_graph()
    print(f"\nNodes: {G.number_of_nodes()}   Edges: {G.number_of_edges()}")

    print("\nTop 5 most connected nodes:")
    for nid, degree in sorted(G.degree, key=lambda x: x[1], reverse=True)[:5]:
        print(f"  {G.nodes[nid]['name']} ({G.nodes[nid]['type']}) - {degree} connections")

    print("\nOutgoing edges from 'Catalog Service':")
    for _, target, data in G.out_edges("catalog service", data=True):
        print(f"  --{data['relation']}--> {G.nodes[target]['name']}")