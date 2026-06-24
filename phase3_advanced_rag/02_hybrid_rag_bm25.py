import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PHASE 3: ADVANCED RAG - HYBRID SEARCH (BM25 + Semantic)
# ==============================================================================
# Semantic Search (Chroma) is great for finding *meaning*, but it can be bad at 
# finding exact names, acronyms, or serial numbers.
# Keyword Search (BM25) is great for exact matches, but doesn't understand meaning.
#
# Hybrid RAG combines both to get the best of both worlds!

def main():
    print("Starting Hybrid RAG Pipeline...")

    # 1. Load Data
    print("1. Loading PDFs...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)

    # 2. Setup Vector Retriever (Semantic Search)
    print("2. Setting up Vector Retriever...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings_model)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 3. Setup BM25 Retriever (Keyword Search)
    print("3. Setting up Keyword (BM25) Retriever...")
    # BM25 operates in-memory, analyzing exact word frequencies.
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2

    # 4. Setup Ensemble Retriever (The Hybrid Magic)
    print("4. Combining them into a Hybrid Retriever...")
    # We assign weights: 50% importance to Semantic, 50% importance to Exact Keywords.
    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], 
        weights=[0.5, 0.5]
    )

    # 5. Setup the Generation Model (LLM)
    print("5. Setting up the Generation Model (LLM)...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 6. Building the RAG Chain
    print("6. Building the LCEL RAG Chain...")
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": hybrid_retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 7. Test It
    print("\n--- Setup Complete! ---\n")
    query = "AGI" # An exact acronym that vector search might struggle with
    print(f"User Question: '{query}'\n")
    
    print("Retrieving chunks using Hybrid Search and Generating Answer...")
    response = rag_chain.invoke(query)
    
    print("\n--- Final Answer ---")
    print(response)

if __name__ == "__main__":
    main()
