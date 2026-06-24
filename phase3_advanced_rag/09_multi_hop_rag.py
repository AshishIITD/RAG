from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PHASE 3: MULTI-HOP RAG (Decomposed Prompting)
# ==============================================================================
# Problem: If you ask "Who founded the company that created ChatGPT?", 
# vector search will look for that exact sentence and fail. 
# 
# Solution: Multi-Hop RAG uses an LLM to "Decompose" the question into 
# multiple sub-questions, executes them sequentially, and combines the answers!

def main():
    print("Starting Multi-Hop RAG Pipeline...\n")

    # 1. Setup the Decomposer LLM
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 2. Build the Decomposition Prompt
    decompose_template = """You are a research assistant. 
    Your job is to take a complex question and break it down into a list of 
    simple, independent sub-questions that need to be answered to solve the main question.
    
    Complex Question: {question}
    
    Sub-questions (one per line):"""
    
    decompose_prompt = PromptTemplate.from_template(decompose_template)
    decompose_chain = decompose_prompt | llm | StrOutputParser()

    # 3. Test the Decomposer
    complex_question = "Which city is the headquarters of the company that Sam Altman founded?"
    
    print(f"User's Complex Question: '{complex_question}'\n")
    print("Decomposing into sub-questions...")
    
    sub_questions_raw = decompose_chain.invoke({"question": complex_question})
    
    print("\n--- The Multi-Hop Plan ---")
    print(sub_questions_raw)
    
    print("\n--- The Execution ---")
    print("In a real system, we would now loop through these sub-questions:")
    print("Hop 1: Run RAG on 'What company did Sam Altman found?' -> Answer: OpenAI")
    print("Hop 2: Run RAG on 'In which city is OpenAI headquartered?' -> Answer: San Francisco")
    print("Final Step: Give the user the final answer 'San Francisco'!")

if __name__ == "__main__":
    main()
