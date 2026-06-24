import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# ==============================================================================
# PHASE 4: KNOWLEDGE BASE INTEGRATION (Metadata Filtering)
# ==============================================================================
# If we have 100 PDFs, vector search might retrieve chunks from the wrong books.
# We solve this by adding "Metadata" to every chunk (e.g. author, year, category).
# We can then filter the database BEFORE doing the math search.

def main():
    print("Starting KB Metadata Filtering Pipeline...")

    # 1. Create Documents with Metadata
    print("1. Creating documents with rich metadata...")
    docs = [
        Document(
            page_content="AGI performance, generality, and autonomy.", 
            metadata={"source": "agi_pdf", "topic": "infrastructure", "year": 2026}
        ),
        Document(
            page_content="Python is a great programming language.", 
            metadata={"source": "python_guide", "topic": "coding", "year": 2020}
        ),
    ]

    # 2. Setup Vector Database
    print("2. Storing in Vector DB...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embedding=embeddings_model)

    # 3. Test Retrieval WITH Filtering
    print("\n--- Setup Complete! ---\n")
    query = "Tell me about provisioning."
    print(f"User Question: '{query}'")
    
    # We apply a strict metadata filter. It will ONLY search documents tagged 'agi_pdf'.
    print("\nRetrieving chunks with filter {'source': 'agi_pdf'}...")
    results = vector_store.similarity_search(
        query, 
        k=1, 
        filter={"source": "agi_pdf"}
    )
    
    for res in results:
        print(f"\nResult: {res.page_content}")
        print(f"Metadata: {res.metadata}")

if __name__ == "__main__":
    main()
