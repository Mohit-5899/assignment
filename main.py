from typing import TypedDict, Annotated, Literal
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

Config.validate()

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

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
            stored_docs = self.rag_agent.get_stored_documents()
            if stored_docs:
                retriever_tool = self.rag_agent.get_retriever_tool()
                self.tools = [get_weather_tool, retriever_tool]
                print(f"📄 Found {len(stored_docs)} existing documents - retriever tool enabled")
        except Exception as e:
            print(f"⚠️ Could not load existing documents: {e}")
    
    def _agent(self, state: AgentState) -> AgentState:
        """Agent decides whether to use tools or respond directly."""
        print("---CALL AGENT---")
        messages = state["messages"]
        model = self.llm.bind_tools(self.tools)
        response = model.invoke(messages)
        return {"messages": [response]}
    
    def _grade_documents(self, state: AgentState) -> Literal["generate", "rewrite"]:
        """Grade retrieved documents for relevance."""
        print("---CHECK RELEVANCE---")
        
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
        
        msg = HumanMessage(content=f"""Look at the input and try to reason about the underlying semantic intent.
        Here is the initial question: {question}
        Formulate an improved question:""")
        
        model = ChatOpenAI(temperature=0, model=Config.LLM_MODEL)
        response = model.invoke([msg])
        return {"messages": [response]}
    
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
        initial_state = {"messages": [HumanMessage(content=query)]}
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

if __name__ == "__main__":
    pipeline = AIPipeline()
    
    test_queries = [
        "What's the weather like in New York?",
        "Tell me about artificial intelligence."
    ]
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = pipeline.process_query_sync(query)
        print(f"Response: {result['response']}")
        print(f"Type: {result['query_type']}")