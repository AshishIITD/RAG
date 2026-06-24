import os
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PHASE 9: SCALING TO 100+ PDFs (Agentic Metadata Routing)
# ==============================================================================
# Problem: If you have 100 different PDFs with different contexts, a normal 
# vector search will pull paragraphs from 5 different books and hallucinate 
# a Frankenstein answer.
#
# Solution: 
# 1. We tag every single chunk with the exact name of the PDF it came from.
# 2. We use a Router LLM to read the user's question and decide WHICH PDF to search.
# 3. We execute a strict vector search that ONLY looks at that one PDF!

def main():
    print("Starting Multi-Document Agentic Routing Pipeline...\n")

    # 1. Create Mock Documents (Pretending these are 2 massive PDFs)
    print("1. Ingesting Documents and Tagging with Metadata...")
    docs = [
        Document(
            page_content="The Apple Vision Pro is a mixed-reality headset announced in 2023.", 
            metadata={"source": "apple_hardware_manual.pdf"}
        ),
        Document(
            page_content="An apple is a sweet, edible fruit produced by an apple tree.", 
            metadata={"source": "botany_fruit_guide.pdf"}
        ),
        Document(
            page_content="The new MacBook Pro uses the M3 chip architecture.", 
            metadata={"source": "apple_hardware_manual.pdf"}
        ),
        Document(
            page_content="Bananas are rich in potassium and grow in tropical climates.", 
            metadata={"source": "botany_fruit_guide.pdf"}
        )
    ]

    # 2. Setup Vector Database
    print("2. Storing in Vector DB...")
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(docs, embedding=embeddings_model)

    # 3. Setup the Router Agent
    print("3. Waking up the Router Agent...")
    llm = ChatOllama(model="llama3.2", temperature=0)

    # We give the LLM a list of all available PDFs so it knows what it can choose from
    available_pdfs = ["apple_hardware_manual.pdf", "botany_fruit_guide.pdf"]
    
    router_template = """You are a document routing agent. 
    You have access to the following documents:
    {available_pdfs}
    
    Look at the user's question. Decide which ONE document is most likely to contain the answer.
    Reply ONLY with the exact filename of the document. Do not add any other text.
    
    Question: {question}
    Document Name:"""
    
    router_prompt = PromptTemplate.from_template(router_template)
    router_chain = router_prompt | llm | StrOutputParser()

    # 4. Test the System
    print("\n--- Testing the System ---\n")
    
    questions = [
        "What kind of chip does the new laptop use?",
        "Are there any fruits that contain high potassium?"
    ]

    for q in questions:
        print(f"=======================================================")
        print(f"User Question: '{q}'\n")
        
        # Step A: Agent decides which PDF to read
        print("Agent is thinking...")
        chosen_pdf = router_chain.invoke({
            "available_pdfs": ", ".join(available_pdfs), 
            "question": q
        }).strip()
        
        print(f"Agent Decision: 'I need to search ONLY inside [{chosen_pdf}]'")
        
        # Step B: We execute a Vector Search strictly filtered to that PDF
        print(f"\nExecuting Vector Search with filter={{'source': '{chosen_pdf}'}}...")
        
        # Notice we use the `filter` keyword argument! This blocks out the other 99 PDFs!
        results = vector_store.similarity_search(
            q, 
            k=1, 
            filter={"source": chosen_pdf}
        )
        
        if results:
            print(f"Retrieved Chunk: {results[0].page_content}")
            
            # Step C: Generate final answer
            answer_prompt = f"Use this context to answer the question: {results[0].page_content}\nQuestion: {q}"
            final_answer = llm.invoke(answer_prompt).content
            print(f"\n--- Final Answer ---\n{final_answer}\n")
        else:
            print("No results found in that document.")

if __name__ == "__main__":
    main()
