import os
import base64
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama

# ==============================================================================
# PHASE 3: MULTIMODAL RAG
# ==============================================================================
# Problem: Normal RAG only searches text. If your PDF has an important pie chart 
# or diagram, standard RAG goes completely blind.
#
# Solution: Multimodal RAG embeds text AND images into the exact same vector space!
# And uses a Vision Model to generate answers.

def main():
    print("Starting Multimodal RAG Architecture Demo...\n")

    print("--- How Multimodal RAG Works ---")
    
    print("\n1. Multimodal Embeddings (e.g., CLIP by OpenAI)")
    print("Instead of a text-only embedding model, we use a model that has been ")
    print("trained on both images and text simultaneously. ")
    
    # We will demonstrate the generation part using a Vision Model (LLaVA)
    print("\n2. The Database & Retrieval (Simulated)")
    print("We save both the text chunks and the actual image files (or their vectors)")
    print("into our Vector Database.")
    print("When the user asks 'Describe this architecture diagram', the vector search")
    print("finds the image of the diagram!")

    print("\n3. Generation (Vision Models)")
    print("We cannot use standard Llama 3.2 for this. We MUST use a ")
    print("Vision Model (like LLaVA via Ollama). ")
    
    print("\n--- Executing Multimodal Generation ---")
    try:
        # Initialize the Vision Model
        print("Waking up Vision LLM (llava)...")
        # NOTE: You must run 'ollama run llava' in your terminal before this will work!
        vision_llm = ChatOllama(model="llava", temperature=0)

        # Let's simulate retrieving an image
        print("Simulating retrieved image (a simple 1x1 pixel for demo)...")
        # This is a base64 encoded 1x1 transparent PNG
        image_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="

        # We pass the retrieved image directly into the Vision Model's prompt
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe the contents of this retrieved image in detail."},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                },
            ]
        )
        
        print("Asking Vision Model to analyze the retrieved image...")
        response = vision_llm.invoke([message])
        
        print("\n--- Final Answer ---")
        print(response.content)

    except Exception as e:
        print(f"\n[Error]: Could not run the Vision Model. Did you pull it?")
        print("To fix this, run this command in your terminal:")
        print("ollama run llava")
        print(f"Error details: {e}")

if __name__ == "__main__":
    main()
