import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from pydantic import BaseModel, Field

# ==============================================================================
# PHASE 3: SELF-RAG (Reflection & Critique)
# ==============================================================================
# Problem: Normal RAG just dumps text and hopes the LLM figures it out. It might hallucinate!
# Solution: Self-RAG adds "Grader" nodes. After the LLM generates an answer, another LLM 
# grades it to see if it hallucinated or failed to answer the question. If it fails, it retries!

class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

def main():
    print("Starting Self-RAG (Reflection Workflow) Pipeline...\n")

    # 1. Setup Retrieval
    print("1. Loading Data and Setting up Retriever...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Setup the LLMs
    print("2. Waking up Generator and Grader LLMs...")
    gen_llm = ChatOllama(model="llama3.2", temperature=0)
    grader_llm = ChatOllama(model="llama3.2", temperature=0, format="json")

    # 3. Setup the Chains
    print("3. Building Generation and Critique Chains...")
    
    # Generation Chain
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following pieces of retrieved context to answer the question.\n\nContext: {context}"),
        ("human", "{question}"),
    ])
    gen_chain = gen_prompt | gen_llm | StrOutputParser()
    
    # Hallucination Grader Chain
    hallucination_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. \n 
        Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in and supported by the facts."""),
        ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}"),
    ])
    hallucination_grader = hallucination_prompt | grader_llm | JsonOutputParser(pydantic_object=GradeHallucinations)

    # 4. Execute the Self-RAG Loop
    print("\n--- Setup Complete! ---\n")
    question = "What are the levels of AGI performance?"
    print(f"User Question: '{question}'\n")
    
    max_retries = 3
    attempt = 1
    
    while attempt <= max_retries:
        print(f"--- ATTEMPT {attempt} ---")
        
        # A. Retrieve
        print("A. Retrieving documents...")
        docs = retriever.invoke(question)
        doc_text = "\n\n".join(doc.page_content for doc in docs)
        
        # B. Generate
        print("B. Generating answer...")
        generation = gen_chain.invoke({"context": doc_text, "question": question})
        # We just print a snippet so it's readable
        snippet = generation[:100].replace('\n', ' ')
        print(f"   [Draft Answer]: {snippet}...")
        
        # C. Critique
        print("C. Critiquing for Hallucinations...")
        score = hallucination_grader.invoke({"documents": doc_text, "generation": generation})
        grade = score['binary_score']
        
        print(f"   [Grader Score]: Is grounded in facts? {grade.upper()}")
        
        if grade.lower() == 'yes':
            print("\n✅ Answer passed critique! It is grounded in the facts.")
            print(f"\n--- Final Approved Answer ---\n{generation}")
            break
        else:
            print("\n❌ Answer FAILED critique! Hallucination detected.")
            if attempt == max_retries:
                print("Max retries reached. Returning best effort.")
                print(f"\n--- Best Effort Answer ---\n{generation}")
            else:
                print("Looping back to try again...\n")
            attempt += 1

if __name__ == "__main__":
    main()
