# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI pipeline demonstrating LangGraph-based agentic RAG (Retrieval-Augmented Generation) with intelligent query routing. The system routes user queries to either a weather API agent or a document RAG agent using LangChain, LangGraph, and Qdrant vector database.

## Key Commands

### Development
```bash
# Run the main pipeline (CLI interface)
python main.py

# Run the Streamlit web interface
streamlit run streamlit_app.py

# Test individual agents
python agents.py
```

### Testing
```bash
# Run all tests (when test file exists)
pytest test_pipeline.py -v

# Run specific test categories
pytest test_pipeline.py::TestWeatherAgent -v
pytest test_pipeline.py::TestRAGAgent -v
pytest test_pipeline.py::TestAIPipeline -v

# Run with coverage
pytest test_pipeline.py --cov=. --cov-report=html
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure .env file with required API keys:
# - OPENAI_API_KEY (for LLM and embeddings)
# - OPENWEATHERMAP_API_KEY (for weather data)
# - LANGCHAIN_API_KEY (for LangSmith tracing)
# - QDRANT_API_KEY (optional, for cloud vector DB)
```

## Architecture

### Core Components

**main.py** - LangGraph workflow orchestration
- `AIPipeline` class: Main entry point that builds and executes the LangGraph workflow
- Graph nodes: `agent` (tool selection), `tools` (tool execution), `rewrite` (query refinement), `generate` (final answer)
- Uses `AgentState` TypedDict with message history for state management
- Implements conditional routing based on tool results and document relevance grading
- Tools are dynamically updated when documents are loaded

**agents.py** - Specialized agents and tools
- `WeatherAgent`: Fetches real-time weather from OpenWeatherMap API, extracts city names from natural language queries
- `RAGAgent`: Handles PDF document processing, text chunking, embedding generation (OpenAI), and vector storage (Qdrant)
- `get_weather_tool`: LangChain tool wrapper for weather queries
- RAG agent creates dynamic retriever tools when documents are loaded

**config.py** - Centralized configuration
- Environment variable management via python-dotenv
- Model settings: `LLM_MODEL="gpt-4"`, `EMBEDDING_MODEL="text-embedding-3-small"`
- RAG parameters: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=200`, `RETRIEVAL_K=3`
- Qdrant configuration with fallback to in-memory mode

**streamlit_app.py** - Interactive web interface
- Chat interface for user queries
- Sidebar for PDF document uploads
- System status showing document chunks and stored documents
- Uses `@st.cache_resource` for pipeline singleton

### LangGraph Workflow

1. **START → agent**: LLM decides whether to use tools or respond directly
2. **agent → tools** (conditional): Executes weather_tool or retriever_tool based on query
3. **tools → grade_documents**: Assesses relevance of retrieved context
4. **grade_documents → generate**: If relevant, generates final answer
5. **grade_documents → rewrite**: If not relevant, rewrites query and loops back to agent
6. **generate → END**: Returns final response

### Vector Database (Qdrant)

- Uses Qdrant Cloud (via QDRANT_URL and QDRANT_API_KEY) or in-memory fallback
- Collection name: "neuro" (from `Config.VECTOR_COLLECTION`)
- Named vector: "Neuro_assignment" (dynamic detection with retry logic)
- Cosine similarity for semantic search with score threshold of 0.3
- Stores document chunks with payload: `{text: str, source: str}`

### Document Processing Pipeline

1. PDF text extraction using PyMuPDF (fitz)
2. Text splitting with RecursiveCharacterTextSplitter
3. Embedding generation via OpenAI embeddings
4. Vector storage in Qdrant with automatic collection creation
5. Dynamic tool registration in LangGraph workflow

## Important Patterns

### Tool Management
The pipeline dynamically updates tools when documents are loaded:
- Initially only `get_weather_tool` is available
- After document upload, `retrieve_documents` tool is added
- Graph is rebuilt with new tools via `_build_graph()`

### Query Routing
LangGraph automatically routes queries based on tool calls:
- LLM analyzes query and selects appropriate tool
- Weather queries → `get_weather_tool`
- Document queries → `retrieve_documents` tool
- No explicit classifier needed; LLM decides via tool binding

### Error Handling
- Config validation ensures all required API keys are present
- Qdrant operations include retry logic (3 attempts with 2s delays)
- Timeout handling for API calls (10s for weather, 30s for Qdrant)
- Graceful fallback to in-memory Qdrant if cloud connection fails

### LangSmith Integration
- Full tracing enabled via `LANGCHAIN_TRACING_V2=true`
- Project name: "neurodyno-ai-pipeline" (from `Config.LANGCHAIN_PROJECT`)
- Traces appear automatically in LangSmith dashboard when configured

## File Structure

```
neurodyno_assignment/
├── main.py              # LangGraph pipeline orchestration
├── agents.py            # WeatherAgent and RAGAgent implementations
├── config.py            # Configuration and environment variables
├── streamlit_app.py     # Web interface
├── requirements.txt     # Python dependencies
├── .env                 # API keys (not in git)
├── sample.pdf           # Example document
└── langgraph_agentic_rag.ipynb  # Development notebook
```

## Dependencies

Core libraries:
- `langchain==0.3.10`, `langgraph==0.2.58`, `langsmith==0.1.147`
- `langchain-openai==1.0.2` (LLM and embeddings)
- `qdrant-client==1.15.1` (vector database)
- `streamlit==1.40.2` (web UI)
- `PyMuPDF==1.24.14` (PDF processing)
- `requests==2.32.3` (weather API)

## Notes

- This project uses in-memory Qdrant by default for development; set `QDRANT_API_KEY` for cloud persistence
- LangGraph conditional edges enable dynamic workflow based on document relevance
- The `_grade_documents` method uses structured output to assess retrieval quality
- Temporary PDF files are created with `temp_` prefix during Streamlit uploads
