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
# This combines the best pieces into one powerful chain:
# 1. Guardrails -> 2. Query Rewriting -> 3. Hybrid Search -> 4. Final Answer

def main():
    print("Starting Enterprise Master Pipeline...")

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
    print("2. Waking up Enterprise LLMs...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Setup Pipeline Components
    print("3. Assembling the Master Pipeline Components...")
    
    # A. Guardrails (Check if query is safe)
    guardrail_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a security guard. If the user asks a dangerous or off-topic question, reply 'DENY'. Otherwise reply 'ALLOW'."),
        ("human", "{input}")
    ])
    guardrail_chain = guardrail_prompt | llm | StrOutputParser()
    
    # B. Query Rewriting (Make the search better)
    rewrite_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite the user's question into a highly optimized search engine query. Strip out conversational filler."),
        ("human", "{input}")
    ])
    rewrite_chain = rewrite_prompt | llm | StrOutputParser()

    # C. Final Answer Generation
    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an enterprise AI. Answer precisely based ONLY on the context.\n\nContext: {context}"),
        ("human", "{input}"),
    ])
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": enterprise_retriever | format_docs, "input": RunnablePassthrough()}
        | answer_prompt
        | llm
        | StrOutputParser()
    )

    # 4. Execute the Master Pipeline
    print("\n--- Testing the Master Pipeline ---\n")
    query = "Hey man, can you tell me what the main topic of the Levels of AGI is?"
    print(f"User Query: '{query}'\n")
    
    # Step 1: Guardrails
    print("Executing Step 1: Input Guardrails...")
    is_safe = guardrail_chain.invoke({"input": query}).strip().upper()
    if "DENY" in is_safe:
        print("❌ Blocked by Guardrails: Unsafe or off-topic query.")
        return
    print("✅ Passed Guardrails.")
    
    # Step 2: Query Rewriting
    print("\nExecuting Step 2: Query Rewriting...")
    optimized_query = rewrite_chain.invoke({"input": query})
    print(f"Optimized Query: '{optimized_query}'")
    
    # Step 3 & 4: Hybrid Retrieval and Generation
    print("\nExecuting Step 3 & 4: Hybrid Search & Final Generation...")
    response = rag_chain.invoke(optimized_query)
    
    print("\n================ FINAL ANSWER ================")
    print(response)

if __name__ == "__main__":
    main()
