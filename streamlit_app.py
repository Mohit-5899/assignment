import streamlit as st
from main import AIPipeline
from config import Config

st.set_page_config(
    page_title="AI Pipeline Demo",
    page_icon="🤖",
    layout="wide"
)

@st.cache_resource
def load_pipeline():
    """Load the AI pipeline (cached for performance)."""
    try:
        Config.validate()
        return AIPipeline()
    except Exception as e:
        st.error(f"Error loading pipeline: {e}")
        return None

def main():
    st.title("🤖 AI Pipeline Demo")
    st.markdown("Ask about weather or upload a document to query!")
    
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not Config.OPENAI_API_KEY or Config.OPENAI_API_KEY == "your_openai_api_key_here":
            st.warning("⚠️ Please set your OpenAI API key in the .env file")
        else:
            st.success("✅ OpenAI API configured")
        
        if not Config.OPENWEATHERMAP_API_KEY or Config.OPENWEATHERMAP_API_KEY == "your_openweathermap_api_key_here":
            st.warning("⚠️ Please set your OpenWeatherMap API key in the .env file")
        else:
            st.success("✅ Weather API configured")
        
        if not Config.LANGCHAIN_API_KEY or Config.LANGCHAIN_API_KEY == "your_langsmith_api_key_here":
            st.warning("⚠️ Please set your LangSmith API key for tracing")
        else:
            st.success("✅ LangSmith configured")
        
        st.divider()
        
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=['pdf'],
            help="Upload a PDF to enable document-based queries"
        )
        
        if uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    with open(f"temp_{uploaded_file.name}", "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    pipeline = load_pipeline()
                    if pipeline:
                        result = pipeline.load_document(f"temp_{uploaded_file.name}")
                        
                        if result["success"]:
                            st.success(f"✅ {result['message']}")
                        else:
                            st.error(f"❌ {result['error']}")
                    
                except Exception as e:
                    st.error(f"Error processing document: {e}")
        
        st.divider()
        
        st.header("📊 System Status")
        pipeline = load_pipeline()
        if pipeline:
            collection_info = pipeline.rag_agent.get_collection_info()
            if collection_info["exists"]:
                st.metric("Document Chunks", collection_info["count"])
                
                # Show existing documents
                stored_docs = pipeline.rag_agent.get_stored_documents()
                if stored_docs:
                    st.subheader("📚 Stored Documents")
                    for doc in stored_docs:
                        with st.expander(f"📄 {doc['filename']}"):
                            st.text(f"Preview: {doc['preview']}")
                            st.text(f"Source: {doc['source']}")
                    
                    st.info("💡 These documents are already available for queries!")
                else:
                    st.info("No documents stored yet")
            else:
                st.info("No documents loaded")
    
    pipeline = load_pipeline()
    if not pipeline:
        st.error("Failed to load AI pipeline. Please check your configuration.")
        return
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about weather or document content..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Processing..."):
                try:
                    result = pipeline.process_query_sync(prompt)
                    response = result.get("response", "Sorry, I couldn't process your request.")
                    query_type = result.get("query_type", "unknown")
                    
                    st.markdown(response)
                    
                    with st.expander("Debug Info"):
                        st.json({
                            "query_type": query_type,
                            "messages_count": len(result.get("messages", [])),
                            "langsmith_project": Config.LANGCHAIN_PROJECT
                        })
                    
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": response
                    })
                    
                except Exception as e:
                    error_msg = f"Error processing query: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": error_msg
                    })
    
    if st.session_state.messages:
        _, col2 = st.columns([3, 1])
        with col2:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = []
                st.rerun()
    
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        **Weather Queries:**
        - "What's the weather in London?"
        - "How's the temperature in Tokyo?"
        - "Tell me about the climate in Paris"
        
        **Document Queries:**
        - Upload a PDF using the sidebar
        - Ask questions about the document content
        - "What is this document about?"
        - "Summarize the key points"
        
        **Example Document Questions:**
        - "What are the main topics covered?"
        - "Can you explain the methodology?"
        - "What are the conclusions?"
        """)

if __name__ == "__main__":
    main()