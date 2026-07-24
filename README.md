\# ⚖️ Digital Personal Data Protection (DPDP) RAG Chatbot

An enterprise-grade Retrieval-Augmented Generation (RAG) pipeline and interactive Streamlit chatbot designed for strict, zero-hallucination factual information extraction from the **\*\*Digital Personal Data Protection Act, 2023\*\*** and **\*\*DPDP Rules, 2025\*\***.

Built with **\*\*LangChain\*\***, **\*\*OpenAI\*\***, **\*\*Pinecone\*\***, **\*\*Cohere Re-rank\*\***, and **\*\*LangSmith\*\*** for full-stack tracing and evaluation.

\---

\#\# 🌟 Key Features

\* **\*\*Multi-Document Dynamic Ingestion:\*\*** Automatically detects, loads, and chunks all PDF documents stored in the \`data/\` directory.  
\* **\*\*Database Reset on Ingestion:\*\*** Automatically wipes and re-provisions the Pinecone vector index prior to embedding to eliminate stale or duplicate data.  
\* **\*\*Advanced Retrieval Strategy:\*\***  
  \* **\*\*Pre-Retrieval (Query Expansion):\*\*** Generates multi-query variations to overcome vocabulary mismatches between natural queries and formal legal jargon.  
  \* **\*\*History-Aware Query Reformulation:\*\*** Translates conversational follow-ups (e.g., *\*"Summarize that in one sentence"\**) into standalone, search-optimized queries.  
  \* **\*\*Post-Retrieval (Cross-Encoder Re-Ranking):\*\*** Utilizes Cohere's \`rerank-english-v3.0\` to re-score and filter retrieved chunks before LLM generation.  
\* **\*\*Strict Anti-Hallucination Guardrails:\*\*** System prompt is constrained to enforce zero-inference extraction (\`temperature=0\`), returning a standardized fallback string if answers are not present in the context.  
\* **\*\*Stateful Conversational Memory:\*\*** Preserves multi-turn chat history across interactions.  
\* **\*\*Observability & Evaluation:\*\*** Native integration with **\*\*LangSmith\*\*** for trace visibility, retrieval scoring, and latency monitoring.  
\* **\*\*Modern Web UI:\*\*** Full-featured **\*\*Streamlit\*\*** dashboard with session metrics, memory controls, visual indicators, and responsive chat UI.

\---

\#\# 🏗️ Pipeline Architecture

\`\`\`text  
                                  \+-----------------------+  
                                  |   User Query / Input  |  
                                  \+-----------+-----------+  
                                              |  
                                              v  
                              \+-------------------------------+  
                              | History-Aware Query Rewriter  |  
                              \+---------------+---------------+  
                                              |  
                                              v  
                              \+-------------------------------+  
                              |  MultiQuery Expansion Engine  |  
                              \+---------------+---------------+  
                                              |  
                                              v  
                              \+-------------------------------+  
                              |   Pinecone Vector Database    |  
                              |  (text-embedding-3-small)     |  
                              \+---------------+---------------+  
                                              |  
                                              v  
                              \+-------------------------------+  
                              |    Cohere Cross-Encoder       |  
                              |        (Re-ranker)            |  
                              \+---------------+---------------+  
                                              |  
                                              v  
                              \+-------------------------------+  
                              | Strict Legal Prompt \+ GPT-4o  |  
                              |       (temperature=0)         |  
                              \+---------------+---------------+  
                                              |  
                                              v  
                                  \+-----------------------+  
                                  | Factual Grounded Output|  
                                  \+-----------------------+

## **📁 Project Structure**

Plaintext  
dpdp-rag-chatbot/  
│  
├── data/                             \# Input directory for legal PDF documents  
│   ├── DPDP\_Act\_2023\_English\_only.pdf  
│   └── DPDP\_Rules\_2025\_English\_only.pdf  
│  
├── .env                              \# API Keys & LangSmith configurations  
├── requirements.txt                  \# Python dependencies  
├── config.py                         \# Centralized parameters & environment constants  
├── ingest.py                         \# Document parsing, index wipe, and Pinecone vector upsert  
├── retrieval.py                      \# Advanced retrieval logic (MultiQuery \+ Cohere Rerank)  
├── generation.py                     \# History-aware stateful RAG chain & prompt guardrails  
├── main.py                           \# CLI interactive chat loop  
└── app.py                            \# Streamlit Web UI application

## **🛠️ Prerequisites & Installation**

### **1\. Requirements**

* Python 3.10+  
* Pinecone API Key  
* OpenAI API Key  
* Cohere API Key  
* LangSmith API Key (Optional, for tracing)

### **2\. Installation Setup**

> 1. **Clone the repository:**  
>    Bash  
>    git clone \[https://github.com/your-username/dpdp-rag-chatbot.git\](https://github.com/your-username/dpdp-rag-chatbot.git)  
>    cd dpdp-rag-chatbot

> 2. **Create and activate a virtual environment:**  
>    Bash  
>    python \-m venv venv  
>    \# On Windows:  
>    venv\\Scripts\\activate  
>    \# On macOS/Linux:  
>    source venv/bin/activate

> 3. **Install dependencies:**  
>    Bash  
>    pip install \-r requirements.txt

## **🔑 Environment Configuration (.env)**

Create a .env file in the root directory and add your credentials:

Code snippet  
\# LLM & Embedding Model Credentials  
OPENAI\_API\_KEY="your\_openai\_api\_key"

\# Vector Database Credentials  
PINECONE\_API\_KEY="your\_pinecone\_api\_key"

\# Re-ranking Provider Credentials  
COHERE\_API\_KEY="your\_cohere\_api\_key"

\# LangSmith Observability & Tracing  
LANGCHAIN\_TRACING\_V2="true"  
LANGCHAIN\_ENDPOINT="\[https://api.smith.langchain.com\](https://api.smith.langchain.com)"  
LANGCHAIN\_API\_KEY="your\_langsmith\_api\_key"  
LANGCHAIN\_PROJECT="dpdp-legal-rag"

## **🚀 Step-by-Step Usage Guide**

### **Step 1: Data Ingestion**

Place all relevant PDF files into the data/ directory and execute the ingestion pipeline:

Bash  
python ingest.py

**What this script does:**

> 1. Dynamically scans the data/ directory for .pdf files.  
> 2. Checks if the designated Pinecone index exists, **deletes it**, and provisions a fresh one.  
> 3. Chunks the documents using RecursiveCharacterTextSplitter (chunk\_size=2500, chunk\_overlap=500).  
> 4. Generates embeddings using text-embedding-3-small and upserts them into Pinecone.

### **Step 2: Running the Chatbot**

#### **Option A: Interactive Streamlit Web UI (Recommended)**

Launch the graphical interface in your browser:

Bash  
streamlit run app.py

* Access the dashboard at http://localhost:8501.  
* Monitor live chat history metrics and LangSmith connectivity from the sidebar.  
* Use the **Clear Chat History** button to reset conversational memory at any time.

#### **Option B: Terminal Command Line Interface**

Run the interactive command-line application:

Bash  
python main.py

## **📊 Observability with LangSmith**

Since LANGCHAIN\_TRACING\_V2=true is set in .env, every execution path is traced automatically. You can log into your [LangSmith Dashboard](https://smith.langchain.com/) to inspect:

* Latency across vector search vs. re-ranking vs. LLM generation.  
* Exact reformulated queries produced by the history-aware retriever.  
* The exact context chunks passed into the prompt after Cohere re-ranking.  
* Input and output token usage.

## **🛡️ License**

This project is open-source and available under the [MIT License](https://www.google.com/search?q=LICENSE).