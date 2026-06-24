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
# PHASE 3: MULTI-HOP RAG (Decomposed Prompting)
# ==============================================================================
# Problem: If you ask a complex question with multiple parts, 
# vector search will look for the exact sentence and fail. 
# 
# Solution: Multi-Hop RAG uses an LLM to "Decompose" the question into 
# multiple sub-questions, executes them sequentially, and combines the answers!

def main():
    print("Starting Multi-Hop RAG Pipeline...\n")

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

    # 3. Build the Decomposition Prompt
    print("3. Building Decomposition Logic...")
    decompose_template = """You are a research assistant. 
    Your job is to take a complex question and break it down into a list of 
    simple, independent sub-questions that need to be answered to solve the main question.
    
    Complex Question: {question}
    
    Sub-questions (output ONLY the questions, one per line):"""
    
    decompose_prompt = PromptTemplate.from_template(decompose_template)
    decompose_chain = decompose_prompt | llm | StrOutputParser()

    # 4. Build the RAG Sub-Chain
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following context to answer the question. Be concise.\n\nContext: {context}"),
        ("human", "{input}"),
    ])
    def format_docs(docs): return "\n\n".join(doc.page_content for doc in docs)
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 5. Build Final Answer Chain
    final_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert synthesizer. Use the provided Scratchpad of sub-questions and answers to formulate a final, comprehensive answer to the user's main question."),
        ("human", "Main Question: {question}\n\nScratchpad:\n{scratchpad}")
    ])
    final_chain = final_prompt | llm | StrOutputParser()

    # 6. Test the System
    print("\n--- Setup Complete! ---\n")
    
    # We will test a complex question that requires multiple jumps of logic.
    complex_question = "What are the levels of AGI performance and how do they relate to autonomy?"
    
    print(f"User's Complex Question: '{complex_question}'\n")
    print("A. Decomposing into sub-questions...")
    
    sub_questions_raw = decompose_chain.invoke({"question": complex_question})
    # Clean the output into a list of actual questions
    sub_questions = [q.strip("- \n") for q in sub_questions_raw.split('\n') if q.strip()]
    
    print("\n--- The Multi-Hop Plan ---")
    for sq in sub_questions:
        print(f"- {sq}")
    
    print("\n--- The Execution ---")
    scratchpad = ""
    for i, sq in enumerate(sub_questions):
        print(f"\nHop {i+1}: Executing RAG for '{sq}'...")
        sub_answer = rag_chain.invoke(sq)
        print(f"-> Answer: {sub_answer}")
        scratchpad += f"Q: {sq}\nA: {sub_answer}\n\n"
        
    print("\n--- Final Synthesis ---")
    print("Passing the scratchpad to the synthesizer LLM...")
    final_answer = final_chain.invoke({
        "question": complex_question,
        "scratchpad": scratchpad
    })
    
    print("\n================ FINAL ANSWER ================\n")
    print(final_answer)

if __name__ == "__main__":
    main()
