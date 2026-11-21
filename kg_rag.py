from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(file_path: str):
    """Load a .txt or .pdf file into LangChain Document objects."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()


def split_documents(documents, chunk_size=500, chunk_overlap=50):
    """Split documents into smaller chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(documents)


if __name__ == "__main__":
    docs = load_documents("data/spotify_web_app_architecture.pdf")
    print(f"Loaded {len(docs)} document(s)")

    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    print(f"Split into {len(chunks)} chunk(s)\n")

    for i, chunk in enumerate(chunks):
        print(f"--- Chunk {i + 1} ---")
        print(chunk.page_content)
        print(f"metadata: {chunk.metadata}\n")