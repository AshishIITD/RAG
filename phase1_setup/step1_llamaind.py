import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
import chromadb

# ==============================================================================
# PHASE 1: BASIC RAG - LLAMAINDEX IMPLEMENTATION
# ==============================================================================
# To run this, you need to install LlamaIndex and its plugins:
# pip install llama-index llama-index-llms-ollama llama-index-embeddings-huggingface llama-index-vector-stores-chroma chromadb

def main():
    print("Starting LlamaIndex Basic RAG Pipeline...")

    # ---------------------------------------------------------
    # STEP 1: Global Settings (The LlamaIndex Way)
    # ---------------------------------------------------------
    # Unlike LangChain where we pass the LLM and Embedding model around explicitly 
    # to every function, LlamaIndex likes to set them globally in the 'Settings' object.
    
    print("1. Configuring Local Models...")
    # Setup our Local LLM (Llama 3.2 via Ollama)
    Settings.llm = Ollama(model="llama3.2", request_timeout=120.0)
    
    # Setup our Local Embedding Model (HuggingFace)
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    
    # Setup Chunking globally (LlamaIndex calls this node parsing)
    Settings.chunk_size = 1000
    Settings.chunk_overlap = 200

    # ---------------------------------------------------------
    # STEP 2: Data Loading
    # ---------------------------------------------------------
    print("2. Loading PDFs...")
    # LlamaIndex makes loading whole directories or files incredibly easy.
    # SimpleDirectoryReader automatically figures out how to read PDFs!
    documents = SimpleDirectoryReader(input_files=[
        "../2606.12683v1.pdf", 
        "../2311.pdf"
    ]).load_data()
    print(f"   Loaded {len(documents)} pages from the PDFs.")

    # ---------------------------------------------------------
    # STEP 3: Vector Database Setup (ChromaDB)
    # ---------------------------------------------------------
    print("3. Creating Vector Database...")
    # We create the Chroma database exactly like before.
    db = chromadb.PersistentClient(path="./chroma_db_llamaindex")
    chroma_collection = db.get_or_create_collection("agi_collection")
    
    # We wrap the Chroma collection so LlamaIndex can talk to it
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    
    # The StorageContext tells LlamaIndex where to save the vectors
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # ---------------------------------------------------------
    # STEP 4: Indexing and Retrieval (The Magic Step)
    # ---------------------------------------------------------
    print("4. Indexing documents...")
    # This single line of code in LlamaIndex does the job of chunking, 
    # embedding, and saving to the database all at once!
    index = VectorStoreIndex.from_documents(
        documents, 
        storage_context=storage_context
    )

    # ---------------------------------------------------------
    # STEP 5: Testing the System
    # ---------------------------------------------------------
    print("\n--- Setup Complete! ---\n")
    user_query = "What is the main topic of the Levels of AGI?"
    print(f"User Question: '{user_query}'\n")
    
    print("Retrieving chunks and generating answer...")
    
    # We turn our index into a "query engine" (this combines the Retriever and the LLM)
    # This is LlamaIndex's version of LangChain's LCEL pipeline!
    query_engine = index.as_query_engine(similarity_top_k=3)
    
    # Execute the search and generation
    response = query_engine.query(user_query)
    
    print("\n--- Final Answer ---")
    # In LlamaIndex, the response object has the answer, but also contains the 
    # source nodes (chunks) it used, making it very easy to trace where it got the info!
    print(response)

if __name__ == "__main__":
    main()
