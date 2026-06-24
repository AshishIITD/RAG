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
# PHASE 5: ENTERPRISE RAG (The Master Pipeline)
# ==============================================================================
# This combines everything we learned into one powerful chain:
# Query -> Hybrid Search (BM25 + Semantic) -> Format -> Prompt -> Llama 3.2 -> Answer

def main():
    print("Starting Enterprise RAG Pipeline...")

    # 1. Load Data
    print("1. Loading Data & Setting up Hybrid Retriever...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

    vector_retriever = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    ).as_retriever(search_kwargs={"k": 2})
    
    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 2

    # The Hybrid Retriever is our Enterprise Search Engine
    enterprise_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5]
    )

    # 2. Setup LLM
    print("2. Waking up Enterprise LLM (Llama 3.2)...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Setup Enterprise Chain
    print("3. Assembling the LCEL Chain...")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise AI. Answer precisely based ONLY on the context.\n\nContext: {context}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    enterprise_chain = (
        {"context": enterprise_retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # 4. Test It
    print("\n--- Setup Complete! ---\n")
    query = "What is the main topic of the Levels of AGI?"
    print(f"User Question: '{query}'\n")
    
    print("Executing Enterprise Pipeline...")
    response = enterprise_chain.invoke(query)
    
    print("\n--- Final Answer ---")
    print(response)

if __name__ == "__main__":
    main()
