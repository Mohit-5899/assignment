import requests
import json
import logging
import time
from typing import List, Optional
import fitz  # PyMuPDF
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.tools.retriever import create_retriever_tool
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@tool
def get_weather_tool(query: str) -> str:
    """Get current weather information for a city mentioned in the query."""
    return WeatherAgent().get_weather(query)

class WeatherAgent:
    """Agent for fetching real-time weather data."""
    
    def __init__(self):
        self.api_key = Config.OPENWEATHERMAP_API_KEY
        self.base_url = "https://api.openweathermap.org/data/2.5/weather"
    
    def extract_city_from_query(self, query: str) -> str:
        """Extract city name from natural language query."""
        common_weather_words = ['weather', 'temperature', 'climate', 'forecast', 'hot', 'cold', 'rain', 'sunny']
        words = query.lower().split()
        
        filtered_words = []
        for word in words:
            clean_word = word.strip('.,!?')
            if (clean_word not in common_weather_words and 
                clean_word not in ['in', 'at', 'the', 'what', 'how', 'is', 'like', 'today'] and
                len(clean_word) > 2):
                filtered_words.append(clean_word)
        
        return filtered_words[0] if filtered_words else "London"
    
    def get_weather(self, query: str) -> str:
        """Fetch weather data for a city mentioned in the query."""
        city = self.extract_city_from_query(query)
        
        try:
            params = {
                'q': city,
                'appid': self.api_key,
                'units': 'metric'
            }
            
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            weather_info = {
                'city': data['name'],
                'country': data['sys']['country'],
                'temperature': data['main']['temp'],
                'feels_like': data['main']['feels_like'],
                'humidity': data['main']['humidity'],
                'pressure': data['main']['pressure'],
                'description': data['weather'][0]['description'],
                'wind_speed': data['wind']['speed']
            }
            
            formatted_weather = f"""
            Weather in {weather_info['city']}, {weather_info['country']}:
            - Temperature: {weather_info['temperature']}°C (feels like {weather_info['feels_like']}°C)
            - Conditions: {weather_info['description'].title()}
            - Humidity: {weather_info['humidity']}%
            - Pressure: {weather_info['pressure']} hPa
            - Wind Speed: {weather_info['wind_speed']} m/s
            """
            
            return formatted_weather.strip()
            
        except requests.RequestException as e:
            return f"Error fetching weather data: {str(e)}"
        except KeyError as e:
            return f"Error parsing weather data: Missing field {str(e)}"
        except Exception as e:
            return f"Unexpected error: {str(e)}"

class RAGAgent:
    """Agent for Retrieval-Augmented Generation from documents."""
    
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            model=Config.EMBEDDING_MODEL,
            api_key=Config.OPENAI_API_KEY
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len
        )
        # Connect to Qdrant Cloud with timeout settings
        if Config.QDRANT_API_KEY:
            try:
                self.qdrant_client = QdrantClient(
                    url=Config.QDRANT_URL,
                    api_key=Config.QDRANT_API_KEY,
                    timeout=30  # 30 second timeout
                )
                logger.info(f"🌐 Connected to Qdrant Cloud: {Config.QDRANT_URL}")
            except Exception as e:
                logger.warning(f"⚠️ Qdrant Cloud connection failed: {e}")
                logger.info("💾 Falling back to in-memory Qdrant")
                self.qdrant_client = QdrantClient(":memory:")
        else:
            # Fallback to in-memory for development
            self.qdrant_client = QdrantClient(":memory:")
            logger.info("💾 Using in-memory Qdrant (add QDRANT_API_KEY for cloud)")
        self.collection_name = Config.VECTOR_COLLECTION
        self._collection_created = False
    
    def _ensure_collection(self, vector_size: int):
        """Ensure Qdrant collection exists with retry logic."""
        if not self._collection_created:
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🔄 Checking collection (attempt {attempt + 1}/{max_retries})")
                    # Try to get collection info first to see vector config
                    collection_info = self.qdrant_client.get_collection(self.collection_name)
                    logger.info(f"📋 Collection '{self.collection_name}' exists with config: {collection_info.config.params}")
                    
                    # Extract vector names from existing collection
                    if hasattr(collection_info.config.params, 'vectors') and hasattr(collection_info.config.params.vectors, 'size'):
                        # Single unnamed vector
                        self.vector_name = None
                        logger.info("📋 Using unnamed vector format")
                    elif hasattr(collection_info.config.params, 'vectors') and isinstance(collection_info.config.params.vectors, dict):
                        # Named vectors - use the first one
                        vector_names = list(collection_info.config.params.vectors.keys())
                        self.vector_name = vector_names[0] if vector_names else "Neuro_assignment"
                        logger.info(f"📋 Using named vector: '{self.vector_name}'")
                    else:
                        # Fallback to known name from screenshot
                        self.vector_name = "Neuro_assignment"
                        logger.info(f"📋 Using fallback vector name: '{self.vector_name}'")
                    
                    self._collection_created = True
                    break
                    
                except Exception as e:
                    if "not found" in str(e).lower():
                        try:
                            # Create new collection
                            self.qdrant_client.create_collection(
                                collection_name=self.collection_name,
                                vectors_config={
                                    "Neuro_assignment": rest.VectorParams(
                                        size=vector_size,
                                        distance=rest.Distance.COSINE
                                    )
                                }
                            )
                            self.vector_name = "Neuro_assignment"
                            self._collection_created = True
                            logger.info(f"✅ Created new collection '{self.collection_name}' with vector 'Neuro_assignment'")
                            break
                        except Exception as create_error:
                            logger.error(f"❌ Failed to create collection: {create_error}")
                            if attempt < max_retries - 1:
                                logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                                time.sleep(retry_delay)
                            else:
                                raise create_error
                    elif "timed out" in str(e).lower() or "timeout" in str(e).lower():
                        logger.warning(f"⏰ Timeout on attempt {attempt + 1}: {e}")
                        if attempt < max_retries - 1:
                            logger.info(f"⏳ Retrying in {retry_delay} seconds...")
                            time.sleep(retry_delay)
                        else:
                            logger.error("❌ All retry attempts failed, using fallback")
                            # Fallback: assume collection exists with standard config
                            self.vector_name = "Neuro_assignment"
                            self._collection_created = True
                            break
                    else:
                        logger.error(f"❌ Failed to get collection info: {e}")
                        if attempt < max_retries - 1:
                            time.sleep(retry_delay)
                        else:
                            # Final fallback
                            self.vector_name = "Neuro_assignment"
                            self._collection_created = True
                            logger.warning("⚠️ Using fallback configuration due to connection issues")
                            break
    
    def load_document(self, file_path: str) -> dict:
        """Load and process a PDF document into the vector database."""
        logger.info(f"🔄 Starting document processing for: {file_path}")
        start_time = time.time()
        
        try:
            # PDF Text Extraction
            logger.info("📄 Extracting text from PDF...")
            doc = fitz.open(file_path)
            text = ""
            page_count = 0
            for page in doc:
                text += page.get_text()
                page_count += 1
            doc.close()
            logger.info(f"✅ Extracted text from {page_count} pages, total characters: {len(text)}")
            
            if not text.strip():
                return {"success": False, "error": "No text found in PDF"}
            
            # Text Chunking
            logger.info("✂️ Splitting text into chunks...")
            chunks = self.text_splitter.split_text(text)
            logger.info(f"✅ Created {len(chunks)} text chunks (size: {Config.CHUNK_SIZE}, overlap: {Config.CHUNK_OVERLAP})")
            
            if not chunks:
                return {"success": False, "error": "No text chunks created"}
            
            # Embedding Generation
            logger.info(f"🧠 Generating embeddings using {Config.EMBEDDING_MODEL}...")
            embedding_start = time.time()
            embeddings = self.embeddings.embed_documents(chunks)
            embedding_time = time.time() - embedding_start
            logger.info(f"✅ Generated {len(embeddings)} embeddings in {embedding_time:.2f}s (vector dim: {len(embeddings[0])})")
            
            if not embeddings:
                return {"success": False, "error": "Failed to generate embeddings"}
            
            # Vector Database Setup
            logger.info("🗄️ Setting up Qdrant collection...")
            self._ensure_collection(len(embeddings[0]))
            logger.info(f"✅ Collection '{self.collection_name}' ready")
            
            # Store in Vector Database
            logger.info("💾 Storing embeddings in vector database...")
            points = []
            for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                # Use dynamic vector name
                if hasattr(self, 'vector_name') and self.vector_name:
                    vector_data = {self.vector_name: embedding}
                else:
                    vector_data = embedding
                    
                points.append(
                    rest.PointStruct(
                        id=i,
                        vector=vector_data,
                        payload={"text": chunk, "source": file_path}
                    )
                )
            
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            
            total_time = time.time() - start_time
            logger.info(f"🎉 Document processing completed in {total_time:.2f}s")
            
            # Generate document summary for better routing
            summary_text = text[:500] + "..." if len(text) > 500 else text
            
            return {
                "success": True, 
                "chunks_processed": len(chunks),
                "embedding_dimensions": len(embeddings[0]),
                "processing_time": round(total_time, 2),
                "document_summary": summary_text,
                "message": f"Successfully loaded {len(chunks)} text chunks from {file_path}"
            }
            
        except Exception as e:
            logger.error(f"❌ Error loading document: {str(e)}")
            return {"success": False, "error": f"Error loading document: {str(e)}"}
    
    def query_documents(self, query: str, k: int = None) -> str:
        """Query the vector database for relevant document chunks."""
        if k is None:
            k = Config.RETRIEVAL_K
        
        logger.info(f"🔍 Querying documents for: '{query}' (top-{k} results)")
        start_time = time.time()
        
        try:
            if not self._collection_created:
                logger.warning("⚠️ No documents loaded in vector database")
                return "No documents loaded. Please upload a document first."
            
            # Generate query embedding
            logger.info(f"🧠 Generating query embedding using {Config.EMBEDDING_MODEL}...")
            embed_start = time.time()
            query_embedding = self.embeddings.embed_query(query)
            embed_time = time.time() - embed_start
            logger.info(f"✅ Query embedding generated in {embed_time:.3f}s (dim: {len(query_embedding)})")
            
            # Vector similarity search
            logger.info(f"🔎 Searching vector database (threshold: 0.3)...")
            search_start = time.time()
            # Use dynamic vector name for search
            if hasattr(self, 'vector_name') and self.vector_name:
                query_vector = (self.vector_name, query_embedding)
            else:
                query_vector = query_embedding
                
            search_results = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                score_threshold=0.3
            )
            search_time = time.time() - search_start
            logger.info(f"✅ Vector search completed in {search_time:.3f}s")
            
            if not search_results:
                logger.info("❌ No relevant documents found above threshold")
                return "No relevant information found in the loaded documents."
            
            # Process results
            context_chunks = []
            logger.info(f"📋 Processing {len(search_results)} relevant chunks:")
            for i, result in enumerate(search_results):
                score = result.score
                chunk_preview = result.payload["text"][:100] + "..." if len(result.payload["text"]) > 100 else result.payload["text"]
                logger.info(f"  {i+1}. Score: {score:.3f} | Preview: {chunk_preview}")
                context_chunks.append(result.payload["text"])
            
            context = "\n\n---\n\n".join(context_chunks)
            total_time = time.time() - start_time
            logger.info(f"🎉 Query completed in {total_time:.2f}s | Context length: {len(context)} chars")
            
            return context
            
        except Exception as e:
            logger.error(f"❌ Error querying documents: {str(e)}")
            return f"Error querying documents: {str(e)}"
    
    def get_collection_info(self) -> dict:
        """Get information about the current collection."""
        try:
            info = self.qdrant_client.get_collection(self.collection_name)
            return {
                "exists": True,
                "count": info.points_count,
                "vector_size": getattr(info.config.params.vectors, 'size', 'N/A') if hasattr(info.config.params.vectors, 'size') else 'Named vectors'
            }
        except Exception as e:
            logger.warning(f"Could not get collection info: {e}")
            return {"exists": False, "count": 0}
    
    def get_stored_documents(self) -> list:
        """Get list of documents already stored in the vector database."""
        try:
            # Search for all points to get document sources
            search_results = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                with_payload=True
            )
            
            # Extract unique document sources and create simplified document list
            sources = {}  # source -> first chunk info
            for point in search_results[0]:  # scroll returns (points, next_page_offset)
                if 'source' in point.payload:
                    source = point.payload['source']
                    if source not in sources:  # Only store first occurrence
                        text_preview = point.payload.get('text', '')[:100]
                        sources[source] = {
                            'source': source,
                            'preview': text_preview + '...' if len(text_preview) == 100 else text_preview,
                            'filename': source.split('/')[-1] if '/' in source else source
                        }
            
            documents = list(sources.values())
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting stored documents: {e}")
            return []
    
    def get_retriever_tool(self):
        """Create a retriever tool for the vector database."""
        if not self._collection_created:
            raise ValueError("No documents loaded. Please upload a document first.")
        
        # Create a simple retriever function that uses our query_documents method
        class SimpleRetriever:
            def __init__(self, rag_agent):
                self.rag_agent = rag_agent
            
            def get_relevant_documents(self, query):
                result = self.rag_agent.query_documents(query)
                # Return as document-like objects
                from langchain_core.documents import Document
                return [Document(page_content=result)]
        
        retriever = SimpleRetriever(self)
        
        return create_retriever_tool(
            retriever,
            "retrieve_documents",
            "Search and return information from uploaded documents. Use this when users ask questions about document content."
        )

if __name__ == "__main__":
    weather_agent = WeatherAgent()
    rag_agent = RAGAgent()
    
    print("Testing Weather Agent:")
    weather_result = weather_agent.get_weather("What's the weather in Paris?")
    print(weather_result)
    
    print("\nTesting RAG Agent:")
    info = rag_agent.get_collection_info()
    print(f"Collection info: {info}")
    
    query_result = rag_agent.query_documents("Tell me about AI")
    print(f"Query result: {query_result}")