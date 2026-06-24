import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

# ==============================================================================
# PHASE 3: RAPTOR RAG (Recursive Abstractive Processing for Tree-Organized Retrieval)
# ==============================================================================
# Problem: If you ask a high-level question like "Summarize the entire 500-page book",
# standard RAG fails because it only retrieves tiny isolated paragraphs.
#
# Solution: RAPTOR builds a recursive "Tree" of summaries!

def main():
    print("Starting RAPTOR RAG Pipeline (Hierarchical Tree Summarization)...\n")

    llm = ChatOllama(model="llama3.2", temperature=0)
    
    # 1. Load Data
    print("1. Loading Document (Level 0: Raw Chunks)...")
    loader = PyPDFLoader("../2311.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    raw_chunks = text_splitter.split_documents(docs)
    
    # Just take the first 6 chunks for speed during the demo
    raw_chunks = raw_chunks[:6]
    print(f"   Created {len(raw_chunks)} raw chunks.")

    # 2. Level 1 Summarization (Chapters)
    print("\n2. Building Level 1 Summaries (Grouping chunks into Chapters)...")
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Write a concise, 2-sentence summary of the following text:"),
        ("human", "{text}")
    ])
    summary_chain = summary_prompt | llm | StrOutputParser()

    level_1_summaries = []
    # Group every 3 chunks
    group_size = 3
    for i in range(0, len(raw_chunks), group_size):
        group = raw_chunks[i:i+group_size]
        combined_text = "\n\n".join([doc.page_content for doc in group])
        
        print(f"   Summarizing group {i//group_size + 1}...")
        summary = summary_chain.invoke({"text": combined_text})
        
        # We need to wrap it in a Document object for Chroma
        level_1_summaries.append(Document(page_content=summary, metadata={"level": "Chapter Summary"}))

    # 3. Level 2 Summarization (Root Book Summary)
    print("\n3. Building Level 2 Summary (Root Summary of the whole text)...")
    combined_chapters = "\n\n".join([doc.page_content for doc in level_1_summaries])
    root_summary_text = summary_chain.invoke({"text": combined_chapters})
    root_summary = Document(page_content=root_summary_text, metadata={"level": "Root Summary"})
    print("   Created Root Summary.")

    # 4. Embed the entire Tree into the Vector Database
    print("\n4. Embedding the entire Tree (Levels 0, 1, and 2) into ChromaDB...")
    
    # Add metadata to raw chunks
    for chunk in raw_chunks:
        chunk.metadata["level"] = "Raw Text"

    all_tree_documents = raw_chunks + level_1_summaries + [root_summary]
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = Chroma.from_documents(all_tree_documents, embedding=embeddings_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": 1})

    # 5. Test It!
    print("\n--- Testing RAPTOR Search ---")
    print("Because we embedded all levels of the tree, the Vector Database can answer both")
    print("hyper-specific questions AND broad, high-level summary questions!\n")

    questions = [
        "What is the exact name of the computing system mentioned?", # Specific
        "What is the overarching theme of this entire document?" # Broad
    ]

    for q in questions:
        print(f"User Question: '{q}'")
        docs = retriever.invoke(q)
        matched_doc = docs[0]
        
        print(f"Matched Level: [{matched_doc.metadata.get('level', 'Unknown')}]")
        print(f"Matched Text: {matched_doc.page_content[:150]}...\n")

if __name__ == "__main__":
    main()
