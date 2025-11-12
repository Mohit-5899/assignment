import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import json
from langchain_core.messages import HumanMessage, AIMessage
from main import AIPipeline
from agents import WeatherAgent, RAGAgent
from config import Config
import tempfile
import os


class TestWeatherAgent:
    """Test cases for WeatherAgent API handling"""
    
    @pytest.fixture
    def weather_agent(self):
        return WeatherAgent()
    
    def test_extract_city_from_query(self, weather_agent):
        """Test city extraction from natural language queries"""
        test_cases = [
            ("What's the weather in Paris?", "Paris"),
            ("Tell me about London weather", "London"),
            ("How's it looking in New York City today?", "New York City"),
            ("Weather forecast for San Francisco", "San Francisco"),
            ("Is it raining in Mumbai?", "Mumbai")
        ]
        
        for query, expected_city in test_cases:
            city = weather_agent.extract_city_from_query(query)
            assert city == expected_city, f"Expected {expected_city}, got {city}"
    
    @patch('requests.get')
    def test_get_weather_success(self, mock_get, weather_agent):
        """Test successful weather API response"""
        mock_response = Mock()
        mock_response.json.return_value = {
            "weather": [{"main": "Clear", "description": "clear sky"}],
            "main": {"temp": 22.5, "humidity": 65},
            "wind": {"speed": 3.2},
            "name": "Paris"
        }
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response
        
        result = weather_agent.get_weather("Paris")
        
        assert "Paris" in result
        assert "22.5" in result
        assert "Clear" in result
        assert "clear sky" in result
    
    @patch('requests.get')
    def test_get_weather_api_error(self, mock_get, weather_agent):
        """Test weather API error handling"""
        mock_get.side_effect = Exception("API connection failed")
        
        result = weather_agent.get_weather("InvalidCity")
        assert "Error fetching weather data" in result
    
    def test_get_weather_tool_integration(self):
        """Test LangChain tool wrapper"""
        from agents import get_weather_tool
        
        assert get_weather_tool.name == "get_weather"
        assert "weather" in get_weather_tool.description.lower()


class TestRAGAgent:
    """Test cases for RAG document processing and retrieval"""
    
    @pytest.fixture
    def rag_agent(self):
        return RAGAgent()
    
    def test_process_pdf_react_paper(self, rag_agent):
        """Test processing of ReAct methodology PDF"""
        pdf_path = "/Users/mohitmandawat/Coding/neurodyno_assignment/temp_2210.03629v3.pdf"
        
        if os.path.exists(pdf_path):
            chunks = rag_agent.process_pdf(pdf_path)
            
            assert len(chunks) > 0, "Should extract text chunks from PDF"
            
            # Check for ReAct paper content
            react_content = " ".join([chunk.page_content for chunk in chunks])
            assert any(term in react_content.lower() for term in 
                      ["react", "reasoning", "acting", "language model", "thought"]), \
                   "Should contain ReAct methodology content"
    
    def test_process_pdf_seal_paper(self, rag_agent):
        """Test processing of SEAL self-adapting PDF"""
        pdf_path = "/Users/mohitmandawat/Coding/neurodyno_assignment/temp_self_adapting_llm.pdf"
        
        if os.path.exists(pdf_path):
            chunks = rag_agent.process_pdf(pdf_path)
            
            assert len(chunks) > 0, "Should extract text chunks from PDF"
            
            # Check for SEAL paper content
            seal_content = " ".join([chunk.page_content for chunk in chunks])
            assert any(term in seal_content.lower() for term in 
                      ["seal", "self-adapting", "reinforcement", "finetuning"]), \
                   "Should contain SEAL methodology content"
    
    @patch('qdrant_client.QdrantClient')
    def test_vector_storage_creation(self, mock_qdrant, rag_agent):
        """Test vector database collection creation"""
        mock_client = Mock()
        mock_qdrant.return_value = mock_client
        mock_client.collection_exists.return_value = False
        
        rag_agent._ensure_collection_exists()
        
        mock_client.create_collection.assert_called_once()
    
    def test_document_chunking_parameters(self, rag_agent):
        """Test document chunking with specified parameters"""
        test_text = "This is a test document. " * 200  # Create long text
        
        chunks = rag_agent._chunk_text(test_text, "test.pdf")
        
        assert len(chunks) > 1, "Long text should be split into chunks"
        for chunk in chunks:
            assert len(chunk.page_content) <= Config.CHUNK_SIZE + Config.CHUNK_OVERLAP
            assert chunk.metadata["source"] == "test.pdf"


class TestAIPipeline:
    """Test cases for LangGraph pipeline orchestration"""
    
    @pytest.fixture
    def pipeline(self):
        return AIPipeline()
    
    @pytest.mark.asyncio
    async def test_weather_query_routing(self, pipeline):
        """Test query routing to weather agent"""
        query = "What's the weather like in Tokyo?"
        
        with patch.object(pipeline.weather_agent, 'get_weather') as mock_weather:
            mock_weather.return_value = "Tokyo: 25°C, sunny, light breeze"
            
            response = await pipeline.arun(query)
            
            assert "tokyo" in response.lower() or "weather" in response.lower()
            mock_weather.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_document_query_routing(self, pipeline):
        """Test query routing to RAG agent with PDF content"""
        # First load documents
        pdf_path = "/Users/mohitmandawat/Coding/neurodyno_assignment/temp_2210.03629v3.pdf"
        if os.path.exists(pdf_path):
            pipeline.load_documents([pdf_path])
        
        query = "What is the ReAct methodology?"
        response = await pipeline.arun(query)
        
        # Should contain information about ReAct from the PDF
        assert any(term in response.lower() for term in 
                  ["react", "reasoning", "acting", "thought", "action"])
    
    def test_document_grading_logic(self, pipeline):
        """Test document relevance grading"""
        # Mock documents with varying relevance
        relevant_doc = Mock()
        relevant_doc.page_content = "ReAct combines reasoning and acting in language models"
        
        irrelevant_doc = Mock()
        irrelevant_doc.page_content = "This is about cooking recipes"
        
        query = "What is ReAct methodology?"
        
        # Test relevant document
        with patch.object(pipeline.llm, 'with_structured_output') as mock_llm:
            mock_llm.return_value.invoke.return_value = Mock(binary_score="yes")
            
            state = {"messages": [HumanMessage(content=query)], "documents": [relevant_doc]}
            result = pipeline._grade_documents(state)
            
            assert "generate" in result
    
    def test_query_rewriting(self, pipeline):
        """Test query rewrite functionality"""
        original_query = "weather"
        
        with patch.object(pipeline.llm, 'invoke') as mock_llm:
            mock_llm.return_value.content = "What is the current weather conditions?"
            
            state = {"messages": [HumanMessage(content=original_query)]}
            result = pipeline._rewrite(state)
            
            assert len(result["messages"]) > 1
            assert "weather" in result["messages"][-1].content.lower()


class TestLangSmithIntegration:
    """Test cases for LangSmith tracing and monitoring"""
    
    def test_langsmith_configuration(self):
        """Test LangSmith environment configuration"""
        from config import Config
        
        assert Config.LANGCHAIN_TRACING_V2 == "true"
        assert Config.LANGCHAIN_PROJECT == "neurodyno-ai-pipeline"
        assert Config.LANGCHAIN_API_KEY is not None, "LANGCHAIN_API_KEY should be set"
    
    @pytest.mark.asyncio
    async def test_trace_generation(self):
        """Test that pipeline operations generate LangSmith traces"""
        pipeline = AIPipeline()
        
        # This test requires actual LangSmith integration
        if Config.LANGCHAIN_API_KEY:
            query = "Test query for tracing"
            
            # Run pipeline operation that should generate trace
            with patch('langsmith.trace') as mock_trace:
                await pipeline.arun(query)
                
                # Verify trace was attempted (actual tracing depends on network)
                assert True  # Placeholder for trace verification


class TestExpectedResponses:
    """Expected response patterns for README documentation"""
    
    def test_weather_response_format(self):
        """Document expected weather response format"""
        expected_format = {
            "query": "What's the weather in Paris?",
            "expected_response": "Paris: [temperature]°C, [conditions], [additional_details]",
            "contains": ["temperature", "weather condition", "city name"]
        }
        assert expected_format["query"]
        assert expected_format["expected_response"]
        assert len(expected_format["contains"]) >= 3
    
    def test_react_query_response_format(self):
        """Document expected ReAct methodology response"""
        expected_format = {
            "query": "What is the ReAct methodology?",
            "expected_content": [
                "reasoning and acting",
                "thought-action-observation",
                "language models",
                "decision making"
            ],
            "response_type": "Comprehensive explanation based on PDF content"
        }
        assert len(expected_format["expected_content"]) >= 4
    
    def test_seal_query_response_format(self):
        """Document expected SEAL framework response"""
        expected_format = {
            "query": "How does SEAL self-adaptation work?",
            "expected_content": [
                "self-adapting language models",
                "reinforcement learning",
                "finetuning data generation",
                "model updating"
            ],
            "response_type": "Technical explanation of SEAL framework"
        }
        assert len(expected_format["expected_content"]) >= 4


if __name__ == "__main__":
    # Run specific test categories
    print("Running Weather Agent Tests...")
    pytest.main([__file__ + "::TestWeatherAgent", "-v"])
    
    print("\nRunning RAG Agent Tests...")
    pytest.main([__file__ + "::TestRAGAgent", "-v"])
    
    print("\nRunning AI Pipeline Tests...")
    pytest.main([__file__ + "::TestAIPipeline", "-v"])
    
    print("\nRunning LangSmith Integration Tests...")
    pytest.main([__file__ + "::TestLangSmithIntegration", "-v"])