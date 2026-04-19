"""
RAG Knowledge Base Builder (Ingest Script)
===========================================
Run this ONCE manually to build the Chroma vector store:
    python rag/create_db.py

If the DB already exists it will NOT re-ingest unless you delete
rag/chroma_db/ first. This prevents duplicate embeddings and
changing UUIDs on every app start.
"""

import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DOCS_DIR = os.path.join(BASE_DIR, "documents")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")


def ingest(force: bool = False):
    """Build the Chroma DB from guideline documents.

    Args:
        force: If True, delete existing DB and re-ingest.
               If False (default), skip if DB already exists.
    """
    # ── GUARD: if DB already exists, do not re-ingest ──
    db_file = os.path.join(PERSIST_DIR, "chroma.sqlite3")
    if os.path.exists(db_file) and not force:
        print("[RAG] Chroma DB already exists. Skipping ingest.")
        print("[RAG] Delete rag/chroma_db/ manually or pass force=True to re-ingest.")
        return

    # If forcing, clean up first
    if force and os.path.exists(PERSIST_DIR):
        shutil.rmtree(PERSIST_DIR)
        print("[RAG] Cleared existing Chroma DB (force=True)")

    print(f"[RAG] Loading documents from: {DOCS_DIR}")

    # ── Load all .txt files from documents/ ──
    all_documents = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        if filename.endswith(".txt"):
            filepath = os.path.join(DOCS_DIR, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            # Attach source metadata to every document
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
            print(f"  ✓ {filename} ({len(docs)} document(s))")

    print(f"\n[RAG] Total documents loaded: {len(all_documents)}")

    # ── Split into chunks ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\nSECTION", "\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"[RAG] Generated {len(chunks)} chunks")

    # ── Embed and persist — runs ONCE ──
    print("[RAG] Generating embeddings (all-MiniLM-L6-v2)...")
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_fn,
        persist_directory=PERSIST_DIR,
    )

    print(f"\n✅ Chroma DB created at {PERSIST_DIR} with {len(chunks)} chunks")
    print("Sources indexed:")
    sources = set(doc.metadata.get("source", "unknown") for doc in chunks)
    for s in sorted(sources):
        count = sum(1 for d in chunks if d.metadata.get("source") == s)
        print(f"  • {s}: {count} chunks")
    print("[RAG] Done. Do not run this again unless documents change.")


if __name__ == "__main__":
    ingest()
