from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains.history_aware_retriever import create_history_aware_retriever
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains.retrieval import create_retrieval_chain
from retrieval import build_advanced_retriever
import config

def build_rag_chain():
    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    retriever = build_advanced_retriever()

    # --- 1. Contextualize the Question ---
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # --- 2. Answer the Question (Updated Persona) ---
    system_prompt = """You are a highly accurate legal assistant specializing in the Digital Personal Data Protection Act, 2023, and the Digital Personal Data Protection Rules, 2025.
    Your task is to answer the user's question relying EXCLUSIVELY on the provided Context below.
    
    Strict Execution Rules:
    1. DO NOT use any outside knowledge, prior training data, or external facts.
    2. You may synthesize, summarize, and aggregate facts that are explicitly stated in the context to form your answer.
    3. If the context describes the obligations, functions, or conditions of an entity (e.g., a Consent Manager), use those descriptions to explain what it is.
    4. DO NOT hallucinate or synthesize new information that is not supported by the context.
    5. If the provided Context genuinely does not contain information related to the query, respond EXACTLY with: "The provided documents do not contain information to answer this query."

    Context:
    {context}
    """

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    # --- 3. Combine into final Stateful RAG Chain ---
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    
    return rag_chain