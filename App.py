import streamlit as st
import os
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)
from transformers import pipeline
from langchain_groq import ChatGroq # type: ignore

# Define Ollama server URL
OLLAMA_SERVER = os.getenv("OLLAMA_SERVER", "http://localhost:11434")
USE_OLLAMA = os.getenv("USE_OLLAMA", "true").lower() == "true"
USE_GROQ = os.getenv("USE_GROQ", "false").lower() == "true"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "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")

# Streamlit UI setup
st.set_page_config(page_title="DeepSeek DevMind AI", layout="wide")
st.title("🧠 DeepSeek DevMind AI-powered Coding Assistant")
st.caption("🚀 Your AI Pair Programmer With Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox("Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:3b"], index=0)
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown("""
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """)
    st.divider()
    st.sidebar.markdown("Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)")

# Initialize the model
llm_engine = None
if USE_OLLAMA:
    try:
        llm_engine = ChatOllama(model=selected_model, base_url=OLLAMA_SERVER, temperature=0.3)
    except Exception as e:
        st.error("⚠️ Failed to connect to Ollama. Switching to backup AI.")
        USE_OLLAMA = False

if not USE_OLLAMA and USE_GROQ:
    llm_engine = ChatGroq(api_key=GROQ_API_KEY, model_name="mixtral-8x7b")
    st.info("Using Groq AI as fallback.")

if not llm_engine:
    llm_engine = pipeline("text-generation", model="bigscience/bloom-560m")
    st.info("Using Hugging Face model as fallback.")

# System Prompt Configuration
system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert AI coding Assistant. Provide concise, correct solutions with strategic print statements and debugging tips."""
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm DeepSeek. How Can I help You Code Today? 💻"}]

# Display chat messages
chat_container = st.container()
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Function to generate AI response
def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})

# Function to build prompt sequence
def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log[-20:]:  # Keep only last 20 messages
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

# User Input Handling
user_query = st.chat_input("Type your coding question here...")
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("🧠 Processing..."):
        try:
            prompt_chain = build_prompt_chain()
            ai_response = generate_ai_response(prompt_chain)
        except Exception as e:
            ai_response = "⚠️ Error processing your request. Please try again."
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()
