"""
RAG Knowledge Base Builder
===========================
Loads all guideline documents from rag/documents/, chunks them,
embeds with sentence-transformers/all-MiniLM-L6-v2, and stores in Chroma.
"""

import os
import shutil
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(base_dir, "documents")
    persist_dir = os.path.join(base_dir, "chroma_db")

    # ── Load all .txt files from documents/ ──
    print(f"Loading documents from: {docs_dir}")
    all_documents = []

    for filename in sorted(os.listdir(docs_dir)):
        if filename.endswith(".txt"):
            filepath = os.path.join(docs_dir, filename)
            loader = TextLoader(filepath, encoding="utf-8")
            docs = loader.load()
            # Attach source metadata to every document
            for doc in docs:
                doc.metadata["source"] = filename
            all_documents.extend(docs)
            print(f"  ✓ {filename} ({len(docs)} document(s))")

    print(f"\nTotal documents loaded: {len(all_documents)}")

    # ── Split into chunks ──
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"Generated {len(chunks)} chunks")

    # ── Clear existing DB ──
    if os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)
        print("Cleared existing Chroma DB")

    # ── Create embeddings and store ──
    print("Generating embeddings (all-MiniLM-L6-v2)...")
    embedding_fn = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma.from_documents(chunks, embedding_fn, persist_directory=persist_dir)

    print(f"\n✅ Chroma DB created at {persist_dir} with {len(chunks)} chunks")
    print("Sources indexed:")
    sources = set(doc.metadata.get("source", "unknown") for doc in chunks)
    for s in sorted(sources):
        count = sum(1 for d in chunks if d.metadata.get("source") == s)
        print(f"  • {s}: {count} chunks")


if __name__ == "__main__":
    main()
