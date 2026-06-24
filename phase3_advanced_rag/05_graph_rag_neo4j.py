import os
import networkx as nx
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PHASE 3: KNOWLEDGE GRAPH RAG (In-Memory Concept)
# ==============================================================================
# Problem: Vector databases are bad at connecting dots across multiple documents.
# Solution: We use a Graph Database to store knowledge as Entities (Nodes) 
# and Relationships (Edges). e.g., (Sam Altman) -[FOUNDED]-> (OpenAI)
#
# KG-RAG goes a step further: It combines Vector Search with Graph Search!
# This script builds an in-memory graph (so you don't need to install Neo4j).

def main():
    print("Starting Knowledge Graph RAG Pipeline Concept...\n")

    # 1. Provide some raw text
    print("1. Raw Text Document:")
    text = (
        "OpenAI was founded by Sam Altman and Elon Musk in San Francisco. "
        "ChatGPT was created by OpenAI. "
        "Sam Altman is currently the CEO of OpenAI."
    )
    print(f"'{text}'\n")

    # 2. Setup the LLM for Extraction
    print("2. Waking up the LLM for Information Extraction...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # 3. Extract Knowledge Triples
    print("3. Extracting Entities and Relationships...")
    extract_template = """You are a knowledge extraction graph builder.
    Extract all relationships from the text as a list of triples: (Entity1, Relationship, Entity2).
    Format each on a new line like this: Entity1 | Relationship | Entity2
    Do not add any other text.
    
    Text: {text}
    Triples:"""
    
    extract_prompt = PromptTemplate.from_template(extract_template)
    extract_chain = extract_prompt | llm | StrOutputParser()
    
    triples_raw = extract_chain.invoke({"text": text})
    print("\n--- Extracted Knowledge Graph ---")
    print(triples_raw)

    # 4. Build the Graph (Using NetworkX in memory)
    print("\n4. Building In-Memory Graph (Nodes & Edges)...")
    G = nx.DiGraph()
    
    # We parse the LLM's output and build a mathematical graph object
    triples = [line.split('|') for line in triples_raw.split('\n') if '|' in line]
    for triple in triples:
        if len(triple) == 3:
            e1, rel, e2 = [t.strip() for t in triple]
            G.add_edge(e1, e2, relation=rel)
            print(f"Added Edge: [{e1}] --({rel})--> [{e2}]")

    # 5. Graph RAG (Answering using Graph Context)
    print("\n5. Executing Graph RAG...")
    question = "Who founded the company that created ChatGPT?"
    print(f"User Question: '{question}'\n")
    
    # In a real Graph RAG, we would query the graph for the exact nodes requested,
    # extract the subgraph, and pass those relationships to the LLM.
    print("Passing the Graph structure to the LLM to traverse the connections...")
    answer_template = """Use the following Knowledge Graph Triples to answer the user's question.
    Graph Context:
    {graph_context}
    
    Question: {question}
    Answer:"""
    
    answer_prompt = PromptTemplate.from_template(answer_template)
    answer_chain = answer_prompt | llm | StrOutputParser()
    
    answer = answer_chain.invoke({
        "graph_context": triples_raw, 
        "question": question
    })
    
    print("\n--- Final Answer ---")
    print(answer)

if __name__ == "__main__":
    main()
