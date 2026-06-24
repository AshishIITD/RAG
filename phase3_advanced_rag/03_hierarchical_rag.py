import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore

# ==============================================================================
# PHASE 3: HIERARCHICAL RAG (ParentDocumentRetriever)
# ==============================================================================
# Problem: If chunks are too small, they lose context. If they are too big, 
# search accuracy drops.
# Solution: Hierarchical RAG chunks the document twice! It searches over the 
# tiny chunks for accuracy, but then returns the massive Parent Chunk for context.

def main():
    print("Starting Hierarchical RAG Pipeline...")

    # 1. Setup Data
    print("1. Loading PDFs...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()

    # 2. Setup Splitters (The Hierarchy)
    print("2. Setting up Parent and Child text splitters...")
    # Parent chunks are large (to give the LLM full context)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    # Child chunks are tiny (for highly accurate math-based vector searching)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # 3. Setup Storage
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma(
        collection_name="split_parents", embedding_function=embeddings_model
    )
    # We need a place to store the massive parent documents (Chroma only gets the tiny ones)
    store = InMemoryStore()

    # 4. Create the Hierarchical Retriever
    print("3. Building ParentDocumentRetriever...")
    retriever = ParentDocumentRetriever(
        vectorstore=vector_store,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # 5. Index Data
    print("4. Indexing data (Chunking twice!)...")
    retriever.add_documents(docs, ids=None)

    # 6. Test it
    print("\n--- Setup Complete! ---\n")
    query = "What are the levels of AGI performance?"
    print(f"User Question: '{query}'\n")
    
    retrieved_docs = retriever.invoke(query)
    
    print("\n--- Retrieved PARENT Document ---")
    print(f"Notice how massive the retrieved text is! It's {len(retrieved_docs[0].page_content)} characters long.")
    print("The system searched the tiny child vectors, but returned the full parent paragraph!")

if __name__ == "__main__":
    main()
