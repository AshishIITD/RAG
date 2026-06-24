import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.runnables import RunnablePassthrough

# ==============================================================================
# PHASE 3: ADAPTIVE RAG (Router)
# ==============================================================================
# Problem: Retrieval is slow and expensive. We shouldn't do it if the user asks
# a simple question like "Hello" or "What is 2+2".
# Solution: An Adaptive RAG uses a fast, cheap LLM to route the question to 
# different systems (VectorDB, Web Search, or direct LLM answer).

def main():
    print("Starting Adaptive RAG Router Pipeline...")

    # 1. Setup Retrieval (Vector Store)
    print("1. Loading Data and Setting up Retriever...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Setup the LLM
    print("2. Waking up the LLMs...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Build the Routing Chain
    print("3. Building the Router Logic...")
    router_template = """You are an expert routing assistant. 
    Analyze the user's question and route it to the appropriate data source.
    Choose exactly one of the following options:
    - 'vectorstore' (if the question is about technical concepts, agi, or architecture)
    - 'websearch' (if the question is about recent news, current events, or live data)
    - 'direct' (if the question is simple math, a greeting, or general knowledge)

    Question: {question}
    Destination:"""
    
    router_prompt = PromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm | StrOutputParser()

    # 4. Build the Execution Chains
    # A. The RAG Chain (for 'vectorstore')
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer the question.\n\nContext: {context}"),
        ("human", "{input}"),
    ])
    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # B. The Direct Chain (for 'direct')
    direct_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Answer the question directly."),
        ("human", "{input}"),
    ])
    direct_chain = direct_prompt | llm | StrOutputParser()

    # 5. Test the Router
    print("\n--- Setup Complete! ---\n")
    
    questions = [
        "What is 2 + 2?",
        "What are the levels of AGI performance?",
        "What is the weather in New York today?"
    ]

    for q in questions:
        print(f"=======================================================")
        print(f"User Question: '{q}'")
        
        destination = router_chain.invoke({"question": q}).strip().lower()
        print(f"Routed Destination: [{destination}]")
        
        if "vectorstore" in destination:
            print("Action: Executing Vector Search and Generation...")
            answer = rag_chain.invoke(q)
            print(f"\n--- Final Answer ---\n{answer}\n")
        elif "websearch" in destination:
            print("Action: Executing Web Search Fallback...")
            print(f"\n--- Final Answer ---\n(Simulated Web Search: The weather in NY is sunny!)\n")
        else:
            print("Action: Executing Direct LLM Generation...")
            answer = direct_chain.invoke({"input": q})
            print(f"\n--- Final Answer ---\n{answer}\n")

if __name__ == "__main__":
    main()
