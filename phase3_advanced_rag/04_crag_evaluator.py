import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

# ==============================================================================
# PHASE 3: CORRECTIVE RAG (CRAG) Evaluator Node
# ==============================================================================
# Problem: Sometimes the Retriever fetches totally useless paragraphs.
# Solution: Before passing documents to the generator, we use a cheap/fast LLM 
# to "Grade" the documents. If it grades them as "Irrelevant", we throw them away 
# and use a Web Search instead!

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

def main():
    print("Starting Corrective RAG (CRAG) Evaluator Pipeline...")

    # 1. Setup Retrieval (Mocking a real database)
    print("1. Loading Data and Setting up Retriever...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings_model)
    # We just get the top 1 document for grading to keep it simple
    retriever = vector_store.as_retriever(search_kwargs={"k": 1}) 

    # 2. Setup the Evaluator LLM
    print("2. Waking up Evaluator LLM...")
    # We must format the LLM to output pure JSON so our code can read 'yes' or 'no'
    evaluator_llm = ChatOllama(model="llama3.2", temperature=0, format="json")

    # 3. Build the Evaluator Prompt
    print("3. Building Evaluator Prompt...")
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    # Build the Evaluator Chain
    retrieval_grader = grade_prompt | evaluator_llm | JsonOutputParser(pydantic_object=GradeDocuments)

    # 4. Setup Generation LLM and Chain
    print("4. Setting up Generation Model...")
    gen_llm = ChatOllama(model="llama3.2", temperature=0)
    
    gen_prompt = ChatPromptTemplate.from_messages([
        ("system", "Use the following pieces of retrieved context to answer the question. If you don't know the answer, say that you don't know.\n\nContext: {context}"),
        ("human", "{input}"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    def generate_answer(question, context_docs):
        chain = (
            {"context": lambda x: format_docs(context_docs), "input": RunnablePassthrough()}
            | gen_prompt
            | gen_llm
            | StrOutputParser()
        )
        return chain.invoke(question)

    # 5. Test It
    print("\n--- Setup Complete! ---\n")
    
    # We test with two questions: one that is in the PDF, and one that is NOT.
    questions = [
        "What are the levels of AGI performance?", # Should be relevant
        "What is the recipe for chocolate chip cookies?" # Should be irrelevant
    ]

    for question in questions:
        print(f"\n=======================================================")
        print(f"User Question: '{question}'")
        print(f"=======================================================\n")
        
        print("A. Retrieving document from Vector Database...")
        retrieved_docs = retriever.invoke(question)
        doc_content = retrieved_docs[0].page_content
        
        print("\nB. Grading the retrieved document...")
        score = retrieval_grader.invoke({"question": question, "document": doc_content})
        grade = score['binary_score']
        
        print(f"   --> Evaluator Grade: [{grade.upper()}]")
        
        if grade.lower() == 'yes':
            print("\nC. Document is relevant! Passing to Generator LLM...")
            answer = generate_answer(question, retrieved_docs)
            print(f"\n--- Final Answer ---\n{answer}")
        else:
            print("\nC. Document is IRRELEVANT! Throwing it away.")
            print("   --> [Action Triggered]: Initiating Web Search to find external information...")
            print(f"\n--- Final Answer ---\n(Simulated Web Search Answer: To make chocolate chip cookies, you need flour, butter, sugar...)")

if __name__ == "__main__":
    main()
