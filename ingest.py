from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
import config
import os
import time

def initialize_vector_db():
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    
    # 1. Wipe the database by deleting the existing index
    if pc.has_index(config.INDEX_NAME):
        print(f"Emptying database: Deleting existing index '{config.INDEX_NAME}'...")
        pc.delete_index(config.INDEX_NAME)
        # Give the Pinecone API a moment to register the deletion
        time.sleep(3)
        
    # 2. Create a fresh, empty index
    print(f"Creating Pinecone index: {config.INDEX_NAME}")
    pc.create_index(
        name=config.INDEX_NAME,
        dimension=config.EMBEDDING_DIMENSION,
        metric="cosine",
        spec=ServerlessSpec(cloud=config.PINECONE_CLOUD, region=config.PINECONE_REGION)
    )
    
    # 3. Block execution until the new index is fully provisioned and ready
    print("Waiting for index initialization...")
    while not pc.describe_index(config.INDEX_NAME).status['ready']:
        time.sleep(1)
        
    print("Database is clean and ready.")
    return pc

def ingest_documents(file_paths: list):
    all_splits = []
    
    # 1. Loop through all provided documents
    for file_path in file_paths:
        print(f"Loading document: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()

        print(f"Splitting {file_path} into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, 
            chunk_overlap=500
        )
        splits = text_splitter.split_documents(docs)
        all_splits.extend(splits)

    print("Initializing vector database and generating embeddings...")
    initialize_vector_db()
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    
    # 2. Upsert all combined chunks into Pinecone
    vectorstore = PineconeVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        index_name=config.INDEX_NAME
    )
    
    print(f"Successfully ingested {len(all_splits)} total chunks into {config.INDEX_NAME}.")
    return vectorstore

if __name__ == "__main__":
    # Get the absolute path of the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the full path to the data/ folder
    data_folder = os.path.join(current_dir, config.DATA_DIR)
    
    # Verify the folder exists
    if not os.path.exists(data_folder):
        print(f"ERROR: Data directory not found at {data_folder}")
        exit(1)
        
    # Dynamically find all PDF files in the directory
    documents_to_load = [
        os.path.join(data_folder, filename) 
        for filename in os.listdir(data_folder) 
        if filename.lower().endswith(".pdf")
    ]
    
    # Safety check in case the folder is empty
    if not documents_to_load:
        print(f"ERROR: No PDF files found in {data_folder}")
        exit(1)
            
    print(f"Found {len(documents_to_load)} PDF document(s). Starting clean ingestion pipeline...")
    ingest_documents(documents_to_load)