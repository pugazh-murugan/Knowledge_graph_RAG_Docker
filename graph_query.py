import difflib
import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from graph_store import load_graph

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


# ---------- PART A: plain NetworkX queries (no LLM) ----------

def find_node(G, name, cutoff=0.7):
    """Fuzzy-match a name to a node id. Returns node id or None."""
    lookup = {data["name"].lower(): nid for nid, data in G.nodes(data=True)}
    key = name.strip().lower()
    if key in lookup:
        return lookup[key]
    # substring match, then fuzzy match
    for k, nid in lookup.items():
        if key in k or k in key:
            return nid
    close = difflib.get_close_matches(key, lookup.keys(), n=1, cutoff=cutoff)
    return lookup[close[0]] if close else None


def get_neighbors(G, nid):
    out = [(d["relation"], G.nodes[v]["name"]) for _, v, d in G.out_edges(nid, data=True)]
    inc = [(G.nodes[u]["name"], d["relation"]) for u, _, d in G.in_edges(nid, data=True)]
    return out, inc


def find_path(G, a, b):
    """Shortest connection between two entities (ignoring edge direction)."""
    na, nb = find_node(G, a), find_node(G, b)
    if not na or not nb:
        return None
    try:
        path = nx.shortest_path(G.to_undirected(), na, nb)
        return [G.nodes[n]["name"] for n in path]
    except nx.NetworkXNoPath:
        return None


# ---------- PART B: Graph RAG ----------

class QuestionEntities(BaseModel):
    entities: list[str]

extract_prompt = ChatPromptTemplate.from_messages([
    ("system", "List the technologies, services, databases or features mentioned "
               "in the user's question. Return only names, as written."),
    ("human", "{question}"),
])
extract_chain = extract_prompt | llm.with_structured_output(QuestionEntities)


def get_facts(G, nid, depth=1, max_facts=30):
    """Collect facts around a node as readable lines."""
    sub = nx.ego_graph(G, nid, radius=depth, undirected=True)
    facts = [f"Entity: {G.nodes[nid]['name']} ({G.nodes[nid]['type']}) - {G.nodes[nid]['description']}"]
    for u, v, d in sub.edges(data=True):
        facts.append(
            f"({G.nodes[u]['name']}) --{d['relation']}--> ({G.nodes[v]['name']}) "
            f"| source text: {d['evidence']}"
        )
    return facts[:max_facts]


answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using ONLY the knowledge graph facts below. "
               "If the facts are not enough, say you don't know. Be concise."),
    ("human", "Facts:\n{context}\n\nQuestion: {question}"),
])
answer_chain = answer_prompt | llm


def ask(G, question, depth=1):
    # 1. which entities is the question about?
    names = extract_chain.invoke({"question": question}).entities

    # 2. link them to graph nodes
    node_ids = [n for n in (find_node(G, x) for x in names) if n]
    print(f"  entities in question: {names}")
    print(f"  matched graph nodes : {[G.nodes[n]['name'] for n in node_ids]}")
    if not node_ids:
        return "I couldn't find those entities in the graph."

    # 3. gather facts (remove duplicates, keep order)
    facts = []
    for nid in node_ids:
        facts.extend(get_facts(G, nid, depth))
    facts = list(dict.fromkeys(facts))
    context = "\n".join(facts)

    # 4. LLM answers from the facts
    return answer_chain.invoke({"context": context, "question": question}).content


if __name__ == "__main__":
    G = load_graph()

    print("=== Part A: plain graph queries ===")
    nid = find_node(G, "Catalog Service")
    out, inc = get_neighbors(G, nid)
    print("Catalog Service outgoing:", out[:4], "...")

    print("Path Catalog Service -> Redis:", find_path(G, "Catalog Service", "Redis"))
    print("Path Kubernetes -> AKS:", find_path(G, "Kubernetes", "AKS"))

    print("\n=== Part B: Graph RAG (type 'exit' to stop) ===")
    while True:
        q = input("\nQuestion: ").strip()
        if q.lower() in ("exit", "quit", ""):
            break
        print("Answer:", ask(G, q))