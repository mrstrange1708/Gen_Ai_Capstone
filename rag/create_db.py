import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def main():
    print("Loading guidelines...")
    
    # Path to guidelines
    base_dir = os.path.dirname(os.path.abspath(__file__))
    guidelines_path = os.path.join(base_dir, "guidelines.txt")
    
    loader = TextLoader(guidelines_path)
    documents = loader.load()

    print(f"Loaded {len(documents)} document(s). Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
    docs = text_splitter.split_documents(documents)

    print(f"Generated {len(docs)} chunks.")
    print("Generating embeddings and creating Chroma DB...")
    
    # Using a fast local model that doesn't need API keys
    embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # create chromadb
    persist_dir = os.path.join(base_dir, "chroma_db")
    db = Chroma.from_documents(
        docs, 
        embedding_function, 
        persist_directory=persist_dir
    )
    print(f"Chroma DB successfully created at {persist_dir}")

if __name__ == "__main__":
    main()
