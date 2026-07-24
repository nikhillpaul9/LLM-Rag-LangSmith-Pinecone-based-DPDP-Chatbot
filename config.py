import os
from dotenv import load_dotenv

# Load environment variables once
load_dotenv()

# Constants
INDEX_NAME = "dpdp-rules-index"
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o"
RERANK_MODEL = "rerank-english-v3.0"
DATA_DIR = "data"
EMBEDDING_DIMENSION = 1536
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"