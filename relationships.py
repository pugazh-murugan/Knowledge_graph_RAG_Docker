from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from kg_rag import load_documents, split_documents
from entities import extract_entities, merge_entities

load_dotenv()


# 1. Schema
class Relationship(BaseModel):
    source: str = Field(description="Source entity name, exactly from the entity list")
    target: str = Field(description="Target entity name, exactly from the entity list")
    relation: Literal[
        "USES", "STORES_DATA_IN", "CACHES_IN", "PUBLISHES_TO", "DEPENDS_ON",
        "PART_OF", "PROVIDES", "HANDLES", "INTEGRATES_WITH", "DEPLOYED_ON",
        "RELATED_TO",
    ]
    evidence: str = Field(description="Short phrase from the text that supports this relationship")


class RelationshipList(BaseModel):
    relationships: list[Relationship]


# 2. LLM + prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(RelationshipList)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract relationships between entities for a knowledge graph. "
     "Use ONLY entity names from the provided entity list, spelled exactly the same. "
     "Only extract relationships that are stated or clearly implied by the text. "
     "Pick the most specific relation type; use RELATED_TO only as a last resort."),
    ("human", "Entities:\n{entity_names}\n\nText:\n{text}"),
])

chain = prompt | structured_llm


# 3. Extract relationships for each chunk
def extract_relationships(entity_results):
    all_rels = []
    for r in entity_results:
        names = [e.name for e in r["entities"]]
        if len(names) < 2:
            continue  # need at least 2 entities to make a relationship

        output = chain.invoke({
            "entity_names": "\n".join(f"- {n}" for n in names),
            "text": r["text"],
        })

        valid = {n.lower() for n in names}
        for rel in output.relationships:
            # drop anything the LLM invented that isn't in our entity list
            if rel.source.lower() in valid and rel.target.lower() in valid:
                all_rels.append(rel)
    return all_rels


# 4. Remove duplicate triples
def merge_relationships(rels):
    seen, unique = set(), []
    for rel in rels:
        key = (rel.source.lower(), rel.relation, rel.target.lower())
        if key not in seen:
            seen.add(key)
            unique.append(rel)
    return unique


if __name__ == "__main__":
    docs = load_documents("data/spotify_web_app_architecture.pdf")
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)

    entity_results = extract_entities(chunks[:3])   # test on 3 chunks
    rels = merge_relationships(extract_relationships(entity_results))

    print(f"\nTotal unique relationships: {len(rels)}\n")
    for rel in rels:
        print(f"({rel.source}) --{rel.relation}--> ({rel.target})")
        print(f"     evidence: {rel.evidence}")