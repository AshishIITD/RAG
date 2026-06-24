from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PHASE 3: ADAPTIVE RAG (Router)
# ==============================================================================
# Problem: Retrieval is slow and expensive. We shouldn't do it if the user asks
# a simple question like "Hello" or "What is 2+2".
# Solution: An Adaptive RAG uses a fast, cheap LLM to route the question to 
# different systems (VectorDB, Web Search, or direct LLM answer).

def main():
    print("Starting Adaptive RAG Router...")

    # 1. Setup the Router LLM
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 2. Build the Routing Prompt
    # We force the LLM to output ONLY a single word category
    router_template = """You are an expert routing assistant. 
    Analyze the user's question and route it to the appropriate data source.
    Choose exactly one of the following options:
    - 'vectorstore' (if the question is about infrastructure, agi, or specific concepts)
    - 'websearch' (if the question is about recent news or current events)
    - 'direct' (if the question is simple math, a greeting, or general knowledge)

    Question: {question}
    Destination:"""
    
    prompt = PromptTemplate.from_template(router_template)
    router_chain = prompt | llm | StrOutputParser()

    # 3. Test the Router
    print("\n--- Testing Routing Logic ---\n")
    
    questions = [
        "What is 2 + 2?",
        "How does AGI endogenous provisioning work?",
        "What is the weather in New York today?"
    ]

    for q in questions:
        print(f"User Question: '{q}'")
        destination = router_chain.invoke({"question": q}).strip().lower()
        print(f"Routed Destination: [{destination}]")
        
        # In a real app, you would use an if/else block here:
        # if destination == "vectorstore":
        #    return rag_chain.invoke(q)
        # elif destination == "websearch":
        #    return web_search_tool.invoke(q)
        # else:
        #    return llm.invoke(q)
        print("-" * 50)

if __name__ == "__main__":
    main()
