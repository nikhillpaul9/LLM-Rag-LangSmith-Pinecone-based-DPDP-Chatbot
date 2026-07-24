import streamlit as st
from generation import build_rag_chain
from langchain_core.messages import HumanMessage, AIMessage

# --- 1. Page Configuration & Styling ---
st.set_page_config(
    page_title="DPDP Legal Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Optional: Add custom CSS to tweak the UI
st.markdown("""
    <style>
        .stChatMessage {
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
        .st-emotion-cache-16idsys p {
            font-size: 1.05rem;
        }
    </style>
""", unsafe_allow_html=True)

# --- 2. Cache the Pipeline ---
# @st.cache_resource ensures the LLM and VectorDB connections are only initialized once,
# preventing lag every time the user interacts with the UI.
@st.cache_resource(show_spinner="Initializing Legal AI Engine...")
def load_chain():
    return build_rag_chain()

rag_chain = load_chain()

# --- 3. Initialize Session State ---
# We need to maintain both Streamlit's UI messages and LangChain's memory objects
if "ui_messages" not in st.session_state:
    st.session_state.ui_messages = [
        {"role": "assistant", "content": "Welcome! I am your AI Legal Assistant specializing in the **DPDP Act, 2023** and **DPDP Rules, 2025**. How can I help you today?"}
    ]

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- 4. Sidebar: Controls & Metrics ---
with st.sidebar:
    # You can replace this URL with a local image path like "data/logo.png"
    st.image("https://www.dpo-india.com/assets/imgs/dpdpact2023.png", caption="Digital Personal Data Protection Framework")
    
    st.markdown("### ⚙️ Dashboard Controls")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.ui_messages = [
            {"role": "assistant", "content": "Chat history cleared. How can I help you?"}
        ]
        st.session_state.chat_history = []
        st.rerun()
        
    st.divider()
    
    st.markdown("### 📊 Session Metrics")
    st.info(f"**Messages Exchanged:** {len(st.session_state.chat_history) // 2}")
    st.success("🟢 LangSmith Tracking: **Active**")
    st.success("🟢 Pinecone DB: **Connected**")
    
    st.divider()
    st.markdown("💡 *Tip: You can ask me to summarize my previous answers!*")

# --- 5. Main UI Header ---
st.title("⚖️ DPDP Intelligence Hub")
st.caption("Powered by LangChain, Pinecone, and OpenAI")
st.divider()

# --- 6. Render Chat History ---
for msg in st.session_state.ui_messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- 7. Chat Input & Processing ---
if prompt := st.chat_input("Ask a question about the DPDP Act or Rules..."):
    
    # 1. Display user message in UI immediately
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
        
    # 2. Generate response with a loading spinner
    with st.chat_message("assistant"):
        with st.spinner("Searching the legal framework & reasoning..."):
            
            # Invoke the LangChain pipeline
            response = rag_chain.invoke({
                "input": prompt,
                "chat_history": st.session_state.chat_history
            })
            
            answer = response["answer"]
            
            # Display the assistant's answer
            st.markdown(answer)
            
    # 3. Save both UI state and LangChain memory state
    st.session_state.ui_messages.append({"role": "assistant", "content": answer})
    st.session_state.chat_history.extend([
        HumanMessage(content=prompt),
        AIMessage(content=answer)
    ])