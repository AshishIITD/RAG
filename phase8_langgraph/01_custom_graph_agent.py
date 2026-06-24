import os
from typing import TypedDict
from langgraph.graph import StateGraph, END
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ==============================================================================
# PHASE 8: CUSTOM LANGGRAPH INTEGRATION
# ==============================================================================
# In Phase 2, we used `create_react_agent` which is a convenient "black box".
# Here, we build the state machine graph from scratch to truly understand it!

# 1. Define the State of the Graph
# This TypedDict represents the "memory" that gets passed between every node.
class AgentState(TypedDict):
    question: str
    needs_search: str
    context: str
    answer: str

def main():
    print("Starting Custom LangGraph Agent...\n")

    # 1. Setup Retrieval (Vector Store)
    print("1. Loading Data and Setting up Retriever...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(docs)
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(chunks, embedding=embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})

    # 2. Setup LLM
    print("2. Waking up the LLM...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Define the Nodes (The actual Python functions that do the work)
    print("3. Defining Graph Nodes (Router, Retriever, Generator)...")

    def router_node(state: AgentState):
        """Decides if we need to search the database."""
        print("   [Node: Router] Analyzing question...")
        prompt = f"Does the question '{state['question']}' require searching a technical database about AGI? Reply ONLY 'yes' or 'no'."
        decision = llm.invoke(prompt).content.strip().lower()
        print(f"   [Node: Router] Decision: {decision}")
        return {"needs_search": decision}

    def retrieve_node(state: AgentState):
        """Searches the database."""
        print("   [Node: Retriever] Searching Vector Database...")
        docs = retriever.invoke(state["question"])
        context = "\n\n".join(doc.page_content for doc in docs)
        return {"context": context}

    def generate_node(state: AgentState):
        """Generates the final answer."""
        print("   [Node: Generator] Writing final answer...")
        if state.get("context"):
            prompt = f"Use this context to answer the question: {state['context']}\n\nQuestion: {state['question']}"
        else:
            prompt = f"Answer the question directly: {state['question']}"
        
        answer = llm.invoke(prompt).content
        return {"answer": answer}

    # 4. Define Conditional Edges
    def should_retrieve(state: AgentState):
        """The condition function that routes traffic based on the Router's decision."""
        if "yes" in state["needs_search"]:
            return "retrieve"
        return "generate"

    # 5. Build the Graph
    print("\n4. Compiling the StateGraph...")
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("generator", generate_node)

    # Add Edges (The Flow)
    workflow.set_entry_point("router")
    workflow.add_conditional_edges(
        "router",
        should_retrieve,
        {
            "retrieve": "retriever",
            "generate": "generator"
        }
    )
    workflow.add_edge("retriever", "generator")
    workflow.add_edge("generator", END)

    # Compile the graph into an executable application!
    app = workflow.compile()

    # 6. Test the Graph
    print("\n--- Testing the Custom LangGraph ---\n")
    
    questions = [
        "What is 2+2?", # Shouldn't need search
        "What are the levels of AGI performance?" # Should need search
    ]

    for q in questions:
        print(f"=======================================================")
        print(f"User Question: '{q}'\n")
        
        # Invoke the graph
        # We pass in the initial state
        final_state = app.invoke({
            "question": q, 
            "needs_search": "", 
            "context": "", 
            "answer": ""
        })
        
        print(f"\n--- Final Answer ---")
        print(final_state["answer"])
        print()

if __name__ == "__main__":
    main()
