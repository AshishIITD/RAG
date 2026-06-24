import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_ollama import ChatOllama

# ==============================================================================
# PHASE 3: GRAPH RAG & KNOWLEDGE GRAPH RAG (KG-RAG)
# ==============================================================================
# Problem: Vector databases are bad at connecting dots across multiple documents.
# Solution: We use a Graph Database (like Neo4j) to store knowledge as 
# Entities (Nodes) and Relationships (Edges). 
# e.g., (Sam Altman) -[FOUNDED]-> (OpenAI)
#
# KG-RAG goes a step further: It combines Vector Search with Graph Search!

def main():
    print("Starting Graph RAG Pipeline Concept...")

    # 1. Connect to the Graph Database
    # (Requires a running instance of Neo4j on your machine or cloud)
    print("1. Connecting to Neo4j Knowledge Graph...")
    """
    graph = Neo4jGraph(
        url="bolt://localhost:7687", 
        username="neo4j", 
        password="password"
    )
    """
    print("   [Mock] Connected to Neo4j database successfully.")

    # 2. Setup the LLM
    # We need a smart LLM because it has to translate human English into 
    # 'Cypher' (the SQL language for Graph Databases).
    print("2. Waking up the LLM...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Create the Graph QA Chain
    print("3. Building the GraphCypherQAChain...")
    """
    chain = GraphCypherQAChain.from_llm(
        cypher_llm=llm,       # LLM to write the database query
        qa_llm=llm,           # LLM to write the final answer
        graph=graph,          # The Neo4j database
        verbose=True,
    )
    """

    # 4. Test It
    print("\n--- Setup Complete! ---\n")
    query = "Who founded the company that created ChatGPT?"
    print(f"User Question: '{query}'\n")
    
    print("Executing Graph RAG...")
    # The magic here is that the LLM first writes: 
    # `MATCH (p:Person)-[:FOUNDED]->(c:Company)-[:CREATED]->(p:Product {name: 'ChatGPT'}) RETURN p.name`
    # It runs the query against Neo4j, gets "Sam Altman", and writes the final answer!
    
    # response = chain.invoke(query)
    
    print("\n--- Final Answer ---")
    print("Based on the Knowledge Graph, Sam Altman founded OpenAI, which created ChatGPT.")

if __name__ == "__main__":
    main()
