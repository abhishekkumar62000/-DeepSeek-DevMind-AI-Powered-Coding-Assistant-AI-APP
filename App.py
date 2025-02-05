import streamlit as st
import os
import httpx
from langchain_ollama import ChatOllama
from langchain.llms import HuggingFaceHub  # Fallback
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)

# Check if running locally or on Streamlit Cloud
if "STREAMLIT_CLOUD" in os.environ:
    BASE_URL = "http://your-cloud-vm-ip:11434"  # Change to your remote Ollama server
else:
    BASE_URL = "http://localhost:11434"

# Initialize the model
try:
    llm_engine = ChatOllama(model="deepseek-r1:1.5b", base_url=BASE_URL, temperature=0.3)
except httpx.ConnectError:
    st.error("⚠️ Unable to connect to Ollama. Using Hugging Face as a fallback.")
    llm_engine = HuggingFaceHub(repo_id="meta-llama/Llama-2-7b-chat-hf", model_kwargs={"temperature": 0.3})

# Custom CSS styling
st.markdown(
    """
    <style>
        .main { background-color: #1a1a1a; color: #ffffff; }
        .sidebar .sidebar-content { background-color: #2d2d2d; }
        .stTextInput textarea { color: #ffffff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧠 DeepSeek DevMind AI-powered Coding Assistant")
st.caption("🚀 Your AI Pair Programmer With Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:3b"], index=0)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("- 🐍 Python Expert\n- 🐞 Debugging Assistant\n- 📝 Code Documentation\n- 💡 Solution Design")
    st.divider()
    st.sidebar.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# System Prompt Configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert AI coding assistant. Provide concise, correct solutions with strategic print statements and debugging tips. Always respond in English."""
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm DeepSeek. How can I help you code today? 💻"}
    ]

# Chat Container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input
user_query = st.chat_input("Type your coding question here...")

def generate_ai_response(prompt_chain):
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({"input": user_query})
    except Exception as e:
        return f"⚠️ Error generating response: {str(e)}"

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

if user_query:
    # Add user message to log
    st.session_state.message_log.append({"role": "user", "content": user_query})
    
    # Generate AI response
    with st.spinner("🧠 Processing..."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    
    # Add AI response to log
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    
    # Limit message history to prevent memory overload
    st.session_state.message_log = st.session_state.message_log[-20:]
    
    # Rerun to update chat display
    st.rerun()
