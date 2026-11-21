from typing import Literal
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from kg_rag import load_documents, split_documents  # your Step 1 file

load_dotenv()  # reads OPENAI_API_KEY from .env


# 1. Schema: what we want the LLM to return
class Entity(BaseModel):
    name: str = Field(description="Name of the entity, e.g. 'PostgreSQL'")
    type: Literal["Service", "Technology", "Database", "Feature", "Page", "Concept"]
    description: str = Field(description="One short sentence about this entity")


class EntityList(BaseModel):
    entities: list[Entity]


# 2. LLM + prompt
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(EntityList)

prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You extract entities for a knowledge graph from technical documents. "
     "Extract only meaningful entities (services, technologies, databases, features, pages, concepts). "
     "Use the exact name as written in the text. Do not invent entities."),
    ("human", "Text:\n{text}"),
])

chain = prompt | structured_llm


# 3. Extract entities from every chunk
def extract_entities(chunks):
    results = []
    for i, chunk in enumerate(chunks):
        output = chain.invoke({"text": chunk.page_content})
        results.append({
            "chunk_id": i,
            "text": chunk.page_content,
            "entities": output.entities,
        })
    return results


# 4. Merge duplicates across chunks (same name = same entity)
def merge_entities(results):
    merged = {}
    for r in results:
        for e in r["entities"]:
            key = e.name.strip().lower()
            if key not in merged:
                merged[key] = e
    return list(merged.values())


if __name__ == "__main__":
    docs = load_documents("data/spotify_web_app_architecture.pdf")
    chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
    print(f"Total chunks: {len(chunks)}")

    # Test on only the first 3 chunks to save cost
    results = extract_entities(chunks[:3])

    for r in results:
        print(f"\n--- Chunk {r['chunk_id']} ---")
        for e in r["entities"]:
            print(f"  [{e.type}] {e.name}: {e.description}")

    unique = merge_entities(results)
    print(f"\nUnique entities so far: {len(unique)}")