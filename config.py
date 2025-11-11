import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGCHAIN_TRACING_V2 = "true"
    LANGSMITH_TRACING = "true"
    LANGSMITH_ENDPOINT = "https://api.smith.langchain.com"
    LANGCHAIN_PROJECT = os.getenv("LANGCHAIN_PROJECT", "pr-best-graduate-81")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "pr-best-graduate-81")
    QDRANT_URL = os.getenv("QDRANT_URL", "https://b3f5d22c-a939-4bec-bcb4-f3eb23e82349.us-east4-0.gcp.cloud.qdrant.io")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    
    LLM_MODEL = "gpt-4"
    EMBEDDING_MODEL = "text-embedding-3-small"
    VECTOR_COLLECTION = "neuro"
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200
    RETRIEVAL_K = 3
    RETRIEVAL_SCORE_THRESHOLD = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0"))
    
    @classmethod
    def validate(cls):
        required = ["OPENAI_API_KEY", "OPENWEATHERMAP_API_KEY", "LANGCHAIN_API_KEY"]
        missing = [key for key in required if not getattr(cls, key)]
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")
        return True