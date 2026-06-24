import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ==============================================================================
# PHASE 1: BASIC RAG - EDUCATIONAL IMPLEMENTATION
# ==============================================================================

# Note: You will need a Google Gemini API key to run the generation part.
# Get one for free at: https://aistudio.google.com/app/apikey
# Set it in your terminal before running: export GOOGLE_API_KEY="your_key_here"

def main():
    print("Starting Basic RAG Pipeline...")

    # ---------------------------------------------------------
    # STEP 1: Data Loading
    # ---------------------------------------------------------
    print("1. Loading PDFs...")
    # Make paths robust so it runs from anywhere
    base_dir = os.path.dirname(os.path.abspath(__file__))
    pdf1_path = os.path.join(base_dir, "../2606.12683v1.pdf")
    pdf2_path = os.path.join(base_dir, "../2311.pdf")
    
    loader1 = PyPDFLoader(pdf1_path)
    loader2 = PyPDFLoader(pdf2_path)
    
    docs1 = loader1.load()
    docs2 = loader2.load()
    all_documents = docs1 + docs2

    # ---------------------------------------------------------
    # STEP 2: Chunking (Splitting)
    # ---------------------------------------------------------
    print("2. Chunking text...")
    # LLMs have a "context window" (they can only read so much text at once).
    # We break the entire PDF into smaller chunks of 1000 characters each.
    # We add a 200-character overlap so sentences don't get cut off abruptly.
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(all_documents)
    print(f"   Created {len(chunks)} chunks from the PDFs.")

    # ---------------------------------------------------------
    # STEP 3 & 4: Embeddings and Vector Space (ChromaDB)
    # ---------------------------------------------------------
    print("3. Creating Vector Database...")
    
    # 🧠 WHAT IS THE VECTOR SPACE?
    # We are using 'ChromaDB' (Chroma). It is a lightweight, open-source 
    # vector database that runs entirely locally on your machine.
    # 
    # HOW IT WORKS:
    # We use an "Embedding Model" to convert human text into arrays of numbers (vectors).
    # Here, we use 'all-MiniLM-L6-v2' from HuggingFace. It's free and runs locally.
    # If two pieces of text have similar meanings, their vectors will be close 
    # together in this mathematical "vector space".
    
    embeddings_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # We take all our text chunks, convert them to vectors, and store them in Chroma.
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings_model,
        persist_directory="./chroma_db" # Saves the DB locally so we don't rebuild it every time
    )

    # ---------------------------------------------------------
    # STEP 5: The Retriever
    # ---------------------------------------------------------
    # 🔍 HOW DOES RETRIEVAL HAPPEN?
    # When you ask a question, the Retriever converts your question into a vector 
    # using the same HuggingFace model. It then searches the ChromaDB vector space 
    # for the closest matching vectors (using a math metric like Cosine Similarity).
    # Here, 'k=3' means it will fetch the top 3 most mathematically similar text chunks.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # ---------------------------------------------------------
    # STEP 6: The Generation Model (LLM)
    # ---------------------------------------------------------
    # 🤖 WHICH MODEL FOR GENERATION?
    # We are using 'Ollama' to run 'llama3.2' 100% locally on your machine.
    # It is completely free, private, and works entirely offline.
    # We set temperature=0 so the model is factual and doesn't hallucinate.
    
    llm = ChatOllama(model="llama3", temperature=0)

    # ---------------------------------------------------------
    # STEP 7: Building the RAG Chain (using modern LCEL)
    # ---------------------------------------------------------
    # We create a prompt that forces the LLM to ONLY use the retrieved chunks.
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, say that you don't know. "
        "Use three sentences maximum and keep the answer concise."
        "\n\n"
        "Context: {context}"
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # Helper function to format the retrieved documents
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # The LCEL (LangChain Expression Language) pipeline:
    # 1. Retrieve docs and format them, pass user input straight through.
    # 2. Feed them into the Prompt.
    # 3. Send the Prompt to the LLM (Gemini).
    # 4. Parse the output into a simple string.
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    # ---------------------------------------------------------
    # STEP 8: Testing the System
    # ---------------------------------------------------------
    print("\n--- Setup Complete! ---\n")
    user_query = "What is the main topic of the Levels of AGI?"
    print(f"User Question: '{user_query}'\n")
    
    print("Retrieving chunks and generating answer...")
    
    # This single call executes the retrieval and generation automatically!
    response = rag_chain.invoke(user_query)
    
    print("\n--- Final Answer ---")
    print(response)

if __name__ == "__main__":
    main()
