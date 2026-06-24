import streamlit as st
import time

# ==============================================================================
# PHASE 7: PRODUCTIZATION (Streamlit Chatbot UI)
# ==============================================================================
# Building a massive Enterprise RAG pipeline is useless if the CEO has to 
# run it from a Linux terminal. 
# We use Streamlit to wrap our pipeline in a beautiful web app.
# 
# NEW: We have added a Mode Switcher to demonstrate the visual difference 
# between a standard "Chatbot" and an "Agent".

# Set page config
st.set_page_config(page_title="Enterprise AI", page_icon="🤖", layout="wide")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    app_mode = st.radio(
        "Select AI Mode:",
        ("Standard RAG Chatbot", "Agentic RAG"),
        help="Switching modes changes how the AI processes your request."
    )
    st.markdown("---")
    st.markdown("### What's the difference?")
    if app_mode == "Standard RAG Chatbot":
        st.info("🤖 **Chatbot Mode:** Reacts instantly. Blindly searches the vector database and generates a response. Very fast, but cannot use tools or reason.")
    else:
        st.success("🕵️ **Agent Mode:** Proactive and thoughtful. It pauses to 'think', decides what tools to use, executes them, analyzes the results, and writes a comprehensive final answer.")

st.title("🤖 Enterprise AI Assistant")
st.markdown(f"Currently running in **{app_mode}** mode. Ask me anything!")

# Initialize chat history in Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What would you like to know?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Simulate backend processing based on the Mode
    with st.chat_message("assistant"):
        
        if app_mode == "Standard RAG Chatbot":
            # Chatbots are fast and direct.
            message_placeholder = st.empty()
            full_response = ""
            simulated_answer = f"According to the semantic search results, AGI refers to Artificial General Intelligence and involves autonomy and generality."
            
            for chunk in simulated_answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)
            
        else:
            # Agents are thoughtful and deliberate. We show their thought process!
            with st.status("🕵️ Agent is thinking...", expanded=True) as status:
                st.write("💭 Analyzing user prompt to determine required actions...")
                time.sleep(1)
                st.write("🛠️ Decided to use tool: `vector_database_search`")
                time.sleep(1.5)
                st.write("🔍 Searching Vector DB for relevant context...")
                time.sleep(1)
                st.write("📄 Found 3 relevant chunks. Synthesizing final answer...")
                time.sleep(1.5)
                status.update(label="✅ Agent finished processing!", state="complete", expanded=False)
            
            # Now we stream the final answer
            message_placeholder = st.empty()
            full_response = ""
            simulated_answer = f"Based on my research using the vector database, I have synthesized an answer for you. AGI (Artificial General Intelligence) focuses on advanced computational autonomy and generalized intelligence performance across diverse domains."
            
            for chunk in simulated_answer.split():
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
            
            message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
