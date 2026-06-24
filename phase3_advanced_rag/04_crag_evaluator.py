import os
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
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

    # 1. Setup the Evaluator LLM
    print("1. Waking up Evaluator LLM...")
    # We must format the LLM to output pure JSON so our code can read 'yes' or 'no'
    llm = ChatOllama(model="llama3.2", temperature=0, format="json")

    # 2. Build the Evaluator Prompt
    print("2. Building Evaluator Prompt...")
    system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
    If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
    
    grade_prompt = ChatPromptTemplate.from_messages([
        ("system", system),
        ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
    ])

    # 3. Build the Chain
    retrieval_grader = grade_prompt | llm | JsonOutputParser(pydantic_object=GradeDocuments)

    # 4. Test It
    print("\n--- Setup Complete! ---\n")
    question = "What is the capital of France?"
    
    # Mocking two different documents the retriever might have returned
    good_doc = "Paris is the beautiful capital city of France."
    bad_doc = "Levels of AGI focuses on endogenous provisioning."

    print(f"User Question: '{question}'\n")

    print(f"Grading Good Doc: '{good_doc}'")
    good_score = retrieval_grader.invoke({"question": question, "document": good_doc})
    print(f"Result: {good_score}\n")

    print(f"Grading Bad Doc: '{bad_doc}'")
    bad_score = retrieval_grader.invoke({"question": question, "document": bad_doc})
    print(f"Result: {bad_score}\n")

    # The LangGraph workflow would look like this:
    # if good_score['binary_score'] == 'yes':
    #     generate_answer()
    # else:
    #     web_search()

if __name__ == "__main__":
    main()
