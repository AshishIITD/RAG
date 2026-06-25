import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# The secret ingredient for Reranking!
from sentence_transformers import CrossEncoder

# ==============================================================================
# PHASE 3: ADVANCED RAG (Two-Stage Reranking)
# ==============================================================================
# Problem: Standard Vector Search (Bi-Encoder) is fast but inaccurate.
# Solution: 
#   Stage 1: Fetch top 10 chunks using fast Vector Search.
#   Stage 2: Use a highly accurate Cross-Encoder to score those 10 chunks against 
#            the specific query, and only pass the best 3 to the LLM!

def main():
    print("Starting Advanced RAG (Reranker) Pipeline...\n")

    # 1. Setup Data & Fast Vector Store (Stage 1 Retriever)
    print("1. Loading Data and Setting up Stage 1 (Vector Retriever)...")
    # Make path robust so it runs from anywhere
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf_path = os.path.join(base_dir, "../2311.pdf")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings)
    
    # Notice we ask for 10 chunks! We cast a wide net.
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

    # 2. Setup the Reranker (Stage 2)
    print("2. Waking up the Stage 2 Cross-Encoder (Reranker)...")
    # This model is specifically trained to score query-document pairs!
    reranker_model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    # 3. Setup LLM
    print("3. Waking up LLM (Llama 3)...")
    llm = ChatOllama(model="llama3", temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an AI assistant. Answer using ONLY the provided context.\n\nContext: {context}"),
        ("human", "{input}")
    ])
    
    answer_chain = prompt | llm | StrOutputParser()

    # 4. Execute the Pipeline
    print("\n--- Testing the Two-Stage Pipeline ---\n")
    query = "What is the primary difference between AGI and ANI?"
    print(f"User Query: '{query}'\n")

    # ---- STAGE 1: FAST RETRIEVAL ----
    print("Executing Stage 1: Fetching top 10 chunks from Vector DB...")
    stage_1_docs = base_retriever.invoke(query)
    print(f"   Fetched {len(stage_1_docs)} chunks.")

    # ---- STAGE 2: RERANKING ----
    print("\nExecuting Stage 2: Reranking chunks using Cross-Encoder...")
    # Format the input for the CrossEncoder: [ (Query, Chunk1), (Query, Chunk2), ... ]
    pairs = [[query, doc.page_content] for doc in stage_1_docs]
    
    # Get the mathematical relevance scores
    scores = reranker_model.predict(pairs)
    
    # Pair the documents with their scores and sort them from highest to lowest
    scored_docs = list(zip(stage_1_docs, scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    
    print("\n   --- Top 3 Reranked Scores ---")
    for i, (doc, score) in enumerate(scored_docs[:3]):
        print(f"   Rank {i+1}: Score = {score:.4f}")

    # Keep only the top 3 best documents
    best_docs = [doc for doc, score in scored_docs[:3]]
    
    # ---- STAGE 3: GENERATION ----
    print("\nExecuting Stage 3: LLM Generation with Top 3 chunks...")
    context = "\n\n".join(doc.page_content for doc in best_docs)
    
    final_answer = answer_chain.invoke({
        "context": context,
        "input": query
    })

    print("\n================ FINAL ANSWER ================")
    print(final_answer)

if __name__ == "__main__":
    main()
