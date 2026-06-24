import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.tools import create_retriever_tool
from langgraph.prebuilt import create_react_agent

# ==============================================================================
# PHASE 2: AGENTIC RAG (LANGGRAPH)
# ==============================================================================
# In Phase 1, the code blindly searched the database EVERY time you asked a question.
# In Phase 2, we turn the database into a "Tool" and give it to the AI Agent.
# The Agent will "think" and decide if it needs to search the PDF, or if it can 
# just answer from its own general knowledge (e.g. for simple math or greetings).

def main():
    print("Starting Agentic RAG Pipeline...")

    # ---------------------------------------------------------
    # STEP 1: Load the Existing Vector Database
    # ---------------------------------------------------------
    print("1. Loading existing Vector Database...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We don't need to load the PDFs again! We just connect to the ChromaDB we already built.
    vector_store = Chroma(
        persist_directory="../chroma_db", 
        embedding_function=embeddings_model
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ---------------------------------------------------------
    # STEP 2: Create a Tool for the Agent
    # ---------------------------------------------------------
    print("2. Turning the Retriever into a Tool...")
    # The agent needs to know WHAT this tool does so it can decide when to use it.
    pdf_search_tool = create_retriever_tool(
        retriever,
        name="agi_research_search",
        description="Search for information about Levels of AGI and related architectural concepts. Always use this tool if the user asks about AGI."
    )
    
    # We put our tool into a list (you could add a WebSearchTool or CalculatorTool here later!)
    tools = [pdf_search_tool]

    # ---------------------------------------------------------
    # STEP 3: Setup the Agent (LLM)
    # ---------------------------------------------------------
    print("3. Waking up the Agent (Llama 3.2)...")
    # We use local Llama 3.2.
    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # We use LangGraph's pre-built 'ReAct' (Reason + Act) agent.
    # This automatically sets up the loop where the AI thinks -> uses tool -> reads result -> thinks again.
    agent_executor = create_react_agent(llm, tools)

    # ---------------------------------------------------------
    # STEP 4: Testing the Agentic System
    # ---------------------------------------------------------
    print("\n--- Setup Complete! ---\n")
    
    # Let's test the agent with two different questions to see how it routes them.
    questions = [
        "What is 10 + 15? (Don't search the PDF for this)",
        "What is the main topic of the Levels of AGI?"
    ]

    for q in questions:
        print(f"\nUser Question: '{q}'")
        print("Agent is thinking...")
        
        # We pass the question to the agent
        response = agent_executor.invoke({"messages": [("user", q)]})
        
        # The response contains the full history of the conversation (including tool calls).
        # We just want the very last message (the final answer).
        final_answer = response["messages"][-1].content
        
        print(f"\n--- Final Answer ---")
        print(final_answer)
        print("-" * 50)

if __name__ == "__main__":
    main()
