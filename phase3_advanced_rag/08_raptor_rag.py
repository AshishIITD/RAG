# ==============================================================================
# PHASE 3: RAPTOR RAG (Recursive Abstractive Processing for Tree-Organized Retrieval)
# ==============================================================================
# Problem: If you ask a high-level question like "Summarize the entire 500-page book",
# standard RAG fails because it only retrieves tiny isolated paragraphs.
#
# Solution: RAPTOR builds a recursive "Tree" of summaries!

def main():
    print("Starting RAPTOR RAG Architecture Demo...")

    print("\n--- How RAPTOR Works (The Tree of Knowledge) ---")
    
    print("\nLevel 0: The Raw Text")
    print("We chunk the 500-page book into 500 individual paragraphs and embed them.")
    
    print("\nLevel 1: First Abstraction")
    print("We use an clustering algorithm (like UMAP + GMM) to group similar paragraphs.")
    print("We pass each group to an LLM and say: 'Summarize these 10 paragraphs into 1 page.'")
    print("We embed these new summaries and save them.")
    
    print("\nLevel 2: Second Abstraction")
    print("We cluster the Level 1 summaries together.")
    print("We pass them to the LLM: 'Summarize these 10 pages into 1 chapter.'")
    print("We embed these new chapters and save them.")
    
    print("\nLevel 3: Root Abstraction")
    print("We cluster the chapters and summarize the entire book into 1 paragraph.")
    print("We embed this root summary and save it.")

    print("\n--- The Result ---")
    print("When the user asks 'What is the overarching theme of the book?', the ")
    print("vector search matches the Level 3 Root Summary!")
    print("When the user asks 'What happened on page 42?', the vector search ")
    print("matches a Level 0 raw paragraph.")
    
    print("\nRAPTOR allows the AI to answer both hyper-specific AND high-level questions!")

if __name__ == "__main__":
    main()
