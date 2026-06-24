import streamlit as st
import time

# ==============================================================================
# PHASE 7: PRODUCTIZATION (Streamlit Chatbot UI)
# ==============================================================================
# Building a massive Enterprise RAG pipeline is useless if the CEO has to 
# run it from a Linux terminal. 
# We use Streamlit to wrap our pipeline in a beautiful web app in just a few lines!

# Set page config
st.set_page_config(page_title="Enterprise RAG Assistant", page_icon="🤖")

st.title("🤖 Enterprise RAG Assistant")
st.markdown("Welcome to the **AGI Infrastructure AI**. Ask me anything!")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know about AGI?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simulate backend processing (In reality, you would call `enterprise_chain.invoke(prompt)` here)
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # Simulate typing effect for the RAG pipeline
        simulated_answer = f"I have searched the Hybrid BM25/Vector database for '{prompt}'. Based on the Enterprise Pipeline, AGI is a powerful infrastructure paradigm."
        
        for chunk in simulated_answer.split():
            full_response += chunk + " "
            time.sleep(0.05)
            # Add a blinking cursor to simulate typing
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
        
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
