import os
from typing import TypedDict, Annotated, Literal, NotRequired
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import tools_condition, ToolNode
import asyncio
from config import Config
from agents import WeatherAgent, RAGAgent, get_weather_tool

# Set LangSmith environment variables
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["LANGSMITH_API_KEY"] = Config.LANGSMITH_API_KEY
os.environ["LANGSMITH_PROJECT"] = Config.LANGSMITH_PROJECT
os.environ["LANGCHAIN_PROJECT"] = Config.LANGSMITH_PROJECT

Config.validate()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    retry_count: NotRequired[int]

class AIPipeline:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=Config.LLM_MODEL,
            api_key=Config.OPENAI_API_KEY,
            temperature=0
        )
        self.weather_agent = WeatherAgent()
        self.rag_agent = RAGAgent()
        self.tools = [get_weather_tool]
        self.graph = self._build_graph()
        self._load_existing_documents()
    
    def _load_existing_documents(self):
        """Load existing documents and update tools."""
        try:
            # Check if collection exists and has documents
            collection_info = self.rag_agent.get_collection_info()
            if collection_info["exists"] and collection_info["count"] > 0:
                # Refresh collection metadata so get_retriever_tool() works
                self.rag_agent._ensure_collection(force_refresh=True)

                stored_docs = self.rag_agent.get_stored_documents()
                if stored_docs:
                    retriever_tool = self.rag_agent.get_retriever_tool()
                    self.tools = [get_weather_tool, retriever_tool]
                    self.graph = self._build_graph()
                    print(f"📄 Found {len(stored_docs)} existing documents - retriever tool enabled")
        except Exception as e:
            print(f"⚠️ Could not load existing documents: {e}")
    
    def _agent(self, state: AgentState) -> AgentState:
        """Agent decides whether to use tools or respond directly."""
        print("---CALL AGENT---")
        messages = state["messages"]

        # Add system message to guide tool usage when documents are available
        if len(self.tools) > 1:  # More than just weather tool
            from langchain_core.messages import SystemMessage
            system_msg = SystemMessage(content="""You are a helpful assistant with access to uploaded documents and a weather API.
When documents are available (you have a 'retrieve_documents' tool), you MUST use it to search for relevant information before answering any question, even if you think you know the answer.
Only use your general knowledge if the documents don't contain relevant information.""")
            messages_with_system = [system_msg] + messages
        else:
            messages_with_system = messages

        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages_with_system)
        return {"messages": [response]}
    
    def _grade_documents(self, state: AgentState) -> Literal["generate", "rewrite"]:
        """Grade retrieved documents for relevance."""
        print("---CHECK RELEVANCE---")

        # Check retry count to prevent infinite loops
        retry_count = state.get("retry_count", 0)
        if retry_count >= 2:
            print(f"---MAX RETRIES REACHED ({retry_count})---")
            return "generate"

        class Grade(BaseModel):
            binary_score: str = Field(description="Relevance score 'yes' or 'no'")

        model = ChatOpenAI(temperature=0, model=Config.LLM_MODEL)
        llm_with_tool = model.with_structured_output(Grade)

        prompt = PromptTemplate(
            template="""You are a grader assessing relevance of retrieved documents to a user question.
            Here is the retrieved document: {context}
            Here is the user question: {question}
            If the document contains keywords or semantic meaning related to the question, grade it as relevant.
            Give a binary score 'yes' or 'no'.""",
            input_variables=["context", "question"]
        )

        chain = prompt | llm_with_tool
        messages = state["messages"]
        last_message = messages[-1]

        # If last message is weather tool result, go directly to generate
        if hasattr(last_message, 'name') and last_message.name == 'get_weather_tool':
            print("---DECISION: WEATHER DATA RETRIEVED---")
            return "generate"

        question = messages[0].content
        docs = last_message.content

        scored_result = chain.invoke({"question": question, "context": docs})

        if scored_result.binary_score == "yes":
            print("---DECISION: DOCS RELEVANT---")
            return "generate"
        else:
            print("---DECISION: DOCS NOT RELEVANT---")
            return "rewrite"
    
    def _rewrite(self, state: AgentState) -> AgentState:
        """Rewrite query for better retrieval."""
        print("---TRANSFORM QUERY---")
        messages = state["messages"]
        question = messages[0].content

        # Increment retry count
        retry_count = state.get("retry_count", 0) + 1
        print(f"---RETRY COUNT: {retry_count}---")

        msg = HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent.
        Here is the initial question: {question}
        Formulate an improved question:""")

        model = ChatOpenAI(temperature=0, model=Config.LLM_MODEL)
        response = model.invoke([msg])
        return {"messages": [response], "retry_count": retry_count}
    
    def _generate(self, state: AgentState) -> AgentState:
        """Generate final answer."""
        print("---GENERATE---")
        messages = state["messages"]
        question = messages[0].content
        last_message = messages[-1]
        docs = last_message.content
        
        prompt = PromptTemplate(
            template="""You are an assistant for question-answering tasks. Use the retrieved context to answer the question.
            If you don't know the answer, say so. Use three sentences maximum and keep the answer concise.
            Question: {question}
            Context: {context}
            Answer:""",
            input_variables=["question", "context"]
        )
        
        llm = ChatOpenAI(model=Config.LLM_MODEL, temperature=0)
        response = (prompt | llm).invoke({"context": docs, "question": question})
        return {"messages": [response.content]}
    
    
    def _build_graph(self):
        """Build the agentic RAG workflow."""
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._agent)
        tool_node = ToolNode(self.tools)
        workflow.add_node("tools", tool_node)
        workflow.add_node("rewrite", self._rewrite)
        workflow.add_node("generate", self._generate)
        
        workflow.add_edge(START, "agent")
        
        workflow.add_conditional_edges(
            "agent",
            tools_condition,
            {"tools": "tools", END: END}
        )
        
        workflow.add_conditional_edges(
            "tools",
            self._grade_documents,
            {"generate": "generate", "rewrite": "rewrite"}
        )
        
        workflow.add_edge("generate", END)
        workflow.add_edge("rewrite", "agent")
        
        return workflow.compile()
    
    async def process_query(self, query: str) -> dict:
        """Process a user query through the agentic pipeline."""
        initial_state = {"messages": [HumanMessage(content=query)], "retry_count": 0}
        result = await self.graph.ainvoke(initial_state)
        
        # Extract response from final message
        final_message = result["messages"][-1]
        response_content = final_message.content if hasattr(final_message, 'content') else str(final_message)
        
        return {
            "messages": result["messages"],
            "response": response_content,
            "query_type": "agentic"
        }
    
    def process_query_sync(self, query: str) -> dict:
        """Synchronous version of process_query."""
        return asyncio.run(self.process_query(query))
    
    def load_document(self, file_path: str):
        """Load a document and update tools."""
        result = self.rag_agent.load_document(file_path)
        
        if result.get("success"):
            try:
                retriever_tool = self.rag_agent.get_retriever_tool()
                self.tools = [get_weather_tool, retriever_tool]
                self.graph = self._build_graph()  # Rebuild graph with new tools
                print(f"🔧 Tools updated with document retriever")
            except Exception as e:
                print(f"⚠️ Could not create retriever tool: {e}")
        
        return result

    def load_documents(self, file_paths: list):
        """Load multiple documents and update tools."""
        results = []
        for file_path in file_paths:
            result = self.load_document(file_path)
            results.append(result)
        return results
    
    async def arun(self, query: str) -> str:
        """Async run method for pipeline."""
        result = await self.process_query(query)
        return result['response']
    
    def run_evaluation_dataset(self, dataset_path: str = None):
        """Run evaluation on predefined dataset."""
        evaluation_dataset = [
            {
                "id": 1,
                "query": "What's the weather like in Paris today?",
                "expected_tool": "get_weather",
                "category": "weather_api"
            },
            {
                "id": 2,
                "query": "How's the weather in Mumbai right now?",
                "expected_tool": "get_weather", 
                "category": "weather_api"
            },
            {
                "id": 3,
                "query": "What is the ReAct methodology and how does it work?",
                "expected_tool": "retrieve_documents",
                "category": "rag_react_paper"
            },
            {
                "id": 4,
                "query": "Explain the thought-action-observation cycle in ReAct",
                "expected_tool": "retrieve_documents",
                "category": "rag_react_paper"
            },
            {
                "id": 5,
                "query": "How does SEAL self-adaptation work in language models?",
                "expected_tool": "retrieve_documents",
                "category": "rag_seal_paper"
            },
            {
                "id": 6,
                "query": "What are the benefits of self-adapting language models according to SEAL?",
                "expected_tool": "retrieve_documents",
                "category": "rag_seal_paper"
            },
            {
                "id": 7,
                "query": "Compare weather in London and New York",
                "expected_tool": "get_weather",
                "category": "weather_api_complex"
            },
            {
                "id": 8,
                "query": "How do ReAct and SEAL approaches differ in their methodology?",
                "expected_tool": "retrieve_documents",
                "category": "rag_comparative"
            },
            {
                "id": 9,
                "query": "What's the temperature in Tokyo and explain artificial intelligence reasoning?",
                "expected_tool": "both_tools",
                "category": "mixed_query"
            },
            {
                "id": 10,
                "query": "Is it raining in San Francisco and what does the ReAct paper say about decision making?",
                "expected_tool": "both_tools",
                "category": "mixed_query"
            }
        ]
        
        print("🧪 Running Evaluation Dataset (10 Examples)")
        print("=" * 60)
        
        results = []
        for item in evaluation_dataset:
            print(f"\n📋 Test {item['id']}: {item['category']}")
            print(f"❓ Query: {item['query']}")
            print(f"🎯 Expected Tool: {item['expected_tool']}")
            print("-" * 40)
            
            try:
                result = self.process_query_sync(item['query'])
                response = result['response']
                
                # Determine which tools were likely used based on response content
                used_tools = []
                if any(city_indicator in response.lower() for city_indicator in ['°c', '°f', 'temperature', 'weather', 'humidity', 'wind']):
                    used_tools.append('get_weather')
                if any(doc_indicator in response.lower() for doc_indicator in ['react', 'seal', 'reasoning', 'acting', 'self-adapting']):
                    used_tools.append('retrieve_documents')
                
                tool_match = (
                    (item['expected_tool'] == 'get_weather' and 'get_weather' in used_tools) or
                    (item['expected_tool'] == 'retrieve_documents' and 'retrieve_documents' in used_tools) or  
                    (item['expected_tool'] == 'both_tools' and len(used_tools) >= 2)
                )
                
                print(f"✅ Response: {response[:150]}{'...' if len(response) > 150 else ''}")
                print(f"🔧 Detected Tools: {used_tools}")
                print(f"🎯 Tool Match: {'✅' if tool_match else '❌'}")
                
                results.append({
                    "id": item['id'],
                    "query": item['query'],
                    "response": response,
                    "expected_tool": item['expected_tool'],
                    "detected_tools": used_tools,
                    "tool_match": tool_match,
                    "category": item['category']
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                results.append({
                    "id": item['id'],
                    "query": item['query'],
                    "error": str(e),
                    "tool_match": False,
                    "category": item['category']
                })
        
        # Summary statistics
        total_tests = len(results)
        successful_tools = sum(1 for r in results if r.get('tool_match', False))
        accuracy = successful_tools / total_tests if total_tests > 0 else 0
        
        print("\n" + "=" * 60)
        print("📊 EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Total Tests: {total_tests}")
        print(f"Successful Tool Routing: {successful_tools}")
        print(f"Accuracy: {accuracy:.1%}")
        
        # Category breakdown
        categories = {}
        for result in results:
            cat = result['category']
            if cat not in categories:
                categories[cat] = {'total': 0, 'success': 0}
            categories[cat]['total'] += 1
            if result.get('tool_match', False):
                categories[cat]['success'] += 1
        
        print("\n📈 Category Breakdown:")
        for cat, stats in categories.items():
            accuracy = stats['success'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {cat}: {stats['success']}/{stats['total']} ({accuracy:.1%})")
        
        return results

if __name__ == "__main__":
    pipeline = AIPipeline()
    
    # Load evaluation PDFs if they exist
    pdf_files = [
        "/Users/mohitmandawat/Coding/neurodyno_assignment/temp_2210.03629v3.pdf",
        "/Users/mohitmandawat/Coding/neurodyno_assignment/temp_self_adapting_llm.pdf"
    ]
    
    import os
    existing_pdfs = [f for f in pdf_files if os.path.exists(f)]
    if existing_pdfs:
        print(f"📚 Loading {len(existing_pdfs)} PDF documents...")
        pipeline.load_documents(existing_pdfs)
    
    # Run evaluation dataset
    print("\n🚀 Starting Evaluation...")
    evaluation_results = pipeline.run_evaluation_dataset()
    
    print("\n✨ Evaluation Complete! Check LangSmith traces at:")
    print("🔗 https://smith.langchain.com/")
    print("🏷️  Project: pr-best-graduate-81")