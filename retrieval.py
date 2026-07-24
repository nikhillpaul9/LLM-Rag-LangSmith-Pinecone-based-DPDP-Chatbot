import os
from config import RERANK_MODEL
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_cohere import CohereRerank
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
import config

def build_advanced_retriever():
    """Wraps the base retriever with query expansion and Cohere re-ranking."""

    llm = ChatOpenAI(model=config.LLM_MODEL, temperature=0)
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = PineconeVectorStore(
        index_name=config.INDEX_NAME, 
        embedding=embeddings
    )
    
    # 1. Base Retriever
    base_retriever = vectorstore.as_retriever(search_kwargs={"k": 10})
    
    # 2. Pre-Retrieval: Multi-Query Expansion
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )
    
    # 3. Post-Retrieval: Cross-Encoder Re-ranking
    compressor = CohereRerank(
        model=RERANK_MODEL,
        top_n=4, 
        cohere_api_key=os.getenv("COHERE_API_KEY")
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=multi_query_retriever
    )
    
    return compression_retriever