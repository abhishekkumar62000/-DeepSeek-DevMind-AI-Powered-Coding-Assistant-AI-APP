import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate,
)

# Custom CSS styling
st.markdown(
    """
<style>
    /* Existing styles */
    .main {
        background-color: #1a1a1a;
        color: #ffffff;
    }
    .sidebar .sidebar-content {
        background-color: #2d2d2d;
    }
    .stTextInput textarea {
        color: #ffffff !important;
    }
    
    /* Add these new styles for select box */
    .stSelectbox div[data-baseweb="select"] {
        color: white !important;
        background-color: #3d3d3d !important;
    }
    
    .stSelectbox svg {
        fill: white !important;
    }
    
    .stSelectbox option {
        background-color: #2d2d2d !important;
        color: white !important;
    }
    
    /* For dropdown menu items */
    div[role="listbox"] div {
        background-color: #2d2d2d !important;
        color: white !important;
    }
</style>
""",
    unsafe_allow_html=True,
)
st.title("🧠 DeepSeek DevMind AI-powered Coding Assistant.")
st.caption("🚀 Your AI Pair Programmer With Debugging Superpowers")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    selected_model = st.selectbox(
        "Choose Model", ["deepseek-r1:1.5b", "deepseek-r1:7b", "deepseek-r1:3b"], index=0
    )
    st.divider()
    st.markdown("### Model Capabilities")
    st.markdown(
        """
    - 🐍 Python Expert
    - 🐞 Debugging Assistant
    - 📝 Code Documentation
    - 💡 Solution Design
    """
    )
    st.divider()
st.sidebar.markdown(
    "Built with [Ollama](https://ollama.ai/) | [LangChain](https://python.langchain.com/)"
    "👨‍💻Developer:-Abhishek❤️Kumar"
)
developer_path = "my.jpg"  # Ensure this file is in the same directory as your script
try:
    st.sidebar.image(developer_path)
except FileNotFoundError:
    st.sidebar.warning("my.jpg file not found. Please check the file path.")


# Initialize the model
llm_engine = ChatOllama(
    model=selected_model, base_url="http://localhost:11434", temperature=0.3
)

# System Prompt Configuration

system_prompt = SystemMessagePromptTemplate.from_template(
    """You are an expert AI coding Assistenat. Provide concise, correct solutions with strategic print statements and debugging tips. Alwys respond in English."""
)

# Session state management
if "message_log" not in st.session_state:
    st.session_state.message_log = [
        {"role": "ai", "content": "Hi! I'm DeepSeek. How Can I help You Code Today? 💻"}
    ]

# Chat Container
chat_container = st.container()

# Display chat messages
with chat_container:
    for message in st.session_state.message_log:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# Chat input and processing
user_query = st.chat_input("Type your coding question here...")


def generate_ai_response(prompt_chain):
    processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
    return processing_pipeline.invoke({})


def build_prompt_chain():
    prompt_sequence = [system_prompt]

    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(
                HumanMessagePromptTemplate.from_template(msg["content"])
            )
        elif msg["role"] == "ai":
            prompt_sequence.append(
                AIMessagePromptTemplate.from_template(msg["content"])
            )

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

    # Rerun to update chat display
    st.rerun()
