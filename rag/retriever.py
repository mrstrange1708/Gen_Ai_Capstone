"""
RAG Retriever — singleton Chroma DB loader
============================================
Loads the persisted Chroma DB ONCE at module level.
Called by the agent's retrieve_guidelines tool on every query.
"""

import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

PERSIST_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

# Load embedding model ONCE at module level — not inside the function
_embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
_db = None


def get_db():
    """Return the singleton Chroma DB instance, loading from disk if needed."""
    global _db
    if _db is None:
        db_file = os.path.join(PERSIST_DIR, "chroma.sqlite3")
        if not os.path.exists(db_file):
            raise RuntimeError(
                "Chroma DB not found at rag/chroma_db/. "
                "Run: python rag/create_db.py"
            )
        _db = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=_embedding_fn,
        )
        print("[RAG] Chroma DB loaded from disk.")
    return _db


def retrieve(query: str, k: int = 5) -> list:
    """Search the Chroma DB for relevant guidelines.

    Args:
        query: Natural language search query.
        k: Number of results to return.

    Returns:
        List of dicts with 'text' and 'source' keys.
    """
    db = get_db()
    docs = db.similarity_search(query, k=k)
    return [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "Unknown"),
        }
        for doc in docs
    ]
