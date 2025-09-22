"""
Streamlit Web Interface for Medical Insurance RAG Agent
"""

import streamlit as st
import time
from pathlib import Path
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import SimpleChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Medical Insurance RAG Agent",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure global settings
@st.cache_resource
def setup_llm_settings():
    """Setup LLM settings with caching."""
    Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
    Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    return True

class StreamlitInsuranceRAG:
    """Streamlit-based Insurance RAG Agent."""
    
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = storage_dir
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        
    @st.cache_resource
    def load_rag_system(_self):
        """Load RAG system with caching."""
        try:
            if Path(_self.storage_dir).exists():
                # Load existing index
                storage_context = StorageContext.from_defaults(persist_dir=_self.storage_dir)
                _self.index = load_index_from_storage(storage_context)
                
                # Create query engine
                _self.query_engine = _self.index.as_query_engine(
                    similarity_top_k=8,
                    response_mode="compact"
                )
                
                # Create chat engine with memory
                memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
                _self.chat_engine = SimpleChatEngine.from_defaults(
                    llm=Settings.llm,
                    memory=memory,
                    system_prompt=(
                        "You are a helpful assistant specializing in medical insurance policies. "
                        "Answer questions about insurance coverage, benefits, exclusions, and procedures. "
                        "Be specific about coverage amounts, time limits, and any conditions."
                    )
                )
                
                return True
            else:
                return False
        except Exception as e:
            st.error(f"Error loading RAG system: {e}")
            return False
    
    def query_with_sources(self, question):
        """Query with source information."""
        if not self.query_engine:
            return "RAG system not loaded. Please check your setup.", []
        
        try:
            response = self.query_engine.query(question)
            sources = []
            
            if hasattr(response, 'source_nodes') and response.source_nodes:
                for i, node in enumerate(response.source_nodes[:3], 1):
                    sources.append({
                        'rank': i,
                        'score': getattr(node, 'score', 0),
                        'text': node.text[:200] + "..." if len(node.text) > 200 else node.text,
                        'metadata': node.metadata
                    })
            
            return str(response), sources
        except Exception as e:
            return f"Error processing query: {e}", []
    
    def chat_query(self, question):
        """Process chat query with enhanced retrieval."""
        if not self.query_engine or not self.chat_engine:
            return "Chat system not loaded. Please check your setup."
        
        try:
            # Get information using query engine for better retrieval
            query_response = self.query_engine.query(question)
            
            # Use chat engine with retrieved information
            enhanced_message = f"Based on the insurance policy information: {query_response}\n\nUser question: {question}"
            chat_response = self.chat_engine.chat(enhanced_message)
            
            return str(chat_response)
        except Exception as e:
            return f"Error processing chat query: {e}"

# Initialize the RAG system
@st.cache_resource
def get_rag_agent():
    """Get RAG agent with caching."""
    setup_llm_settings()
    agent = StreamlitInsuranceRAG()
    success = agent.load_rag_system()
    return agent, success

def main():
    """Main Streamlit application."""
    
    # Header
    st.title("Medical Insurance RAG Agent")
    st.markdown("Ask questions about your medical insurance policy and get accurate, contextual answers.")
    
    # Initialize RAG system
    with st.spinner("Loading RAG system..."):
        rag_agent, system_loaded = get_rag_agent()
    
    if not system_loaded:
        st.error("Could not load RAG system. Please ensure the index exists in ./storage directory.")
        st.info("Run `python query.py` first to create the index.")
        return

    st.success("RAG system loaded successfully!")
    
    # Sidebar
    with st.sidebar:
        st.header("Settings")

        # Query mode selection
        query_mode = st.selectbox(
            "Select Query Mode:",
            ["Chat Mode", "Query with Sources"],
            help="Chat mode maintains conversation context, Query mode shows source citations"
        )

        # Suggested questions
        st.header("Suggested Questions")
        suggested_questions = [
            "What is my annual dental coverage limit?",
            "What vision benefits do I have?",
            "Are there any deductibles I need to pay?",
            "What are the main exclusions in my policy?",
            "How do I submit a claim?",
            "What happens to my benefits when I retire?",
            "What is covered for prescription drugs?",
            "Do I have travel insurance coverage?"
        ]
        
        for question in suggested_questions:
            if st.button(question, key=f"btn_{hash(question)}"):
                st.session_state.selected_question = question
        
        # Reset chat button
        if st.button("Reset Conversation"):
            if hasattr(rag_agent.chat_engine, 'reset'):
                rag_agent.chat_engine.reset()
            st.session_state.messages = []
            st.success("Conversation reset!")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Chat interface
        if query_mode == "Chat Mode":
            st.header("Chat Interface")
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # Handle selected question from sidebar
            if "selected_question" in st.session_state:
                user_input = st.session_state.selected_question
                del st.session_state.selected_question
            else:
                user_input = st.chat_input("Ask about your insurance policy...")
            
            if user_input:
                # Add user message to chat history
                st.session_state.messages.append({"role": "user", "content": user_input})
                with st.chat_message("user"):
                    st.markdown(user_input)
                
                # Get assistant response
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        start_time = time.time()
                        response = rag_agent.chat_query(user_input)
                        response_time = time.time() - start_time
                    
                    st.markdown(response)
                    st.caption(f"Response time: {response_time:.2f} seconds")
                
                # Add assistant response to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
        
        else:  # Query with Sources mode
            st.header("Query with Sources")
            
            # Handle selected question from sidebar
            if "selected_question" in st.session_state:
                user_input = st.session_state.selected_question
                del st.session_state.selected_question
            else:
                user_input = st.text_input("Enter your question:", placeholder="What is my dental coverage?")
            
            if st.button("Search") or user_input:
                if user_input:
                    with st.spinner("Searching..."):
                        start_time = time.time()
                        response, sources = rag_agent.query_with_sources(user_input)
                        response_time = time.time() - start_time
                    
                    # Display response
                    st.subheader("Response")
                    st.write(response)
                    st.caption(f"Response time: {response_time:.2f} seconds")

                    # Display sources
                    if sources:
                        st.subheader("Sources")
                        for source in sources:
                            with st.expander(f"Source {source['rank']} (Score: {source['score']:.3f})"):
                                st.write(source['text'])
                                if source['metadata']:
                                    st.json(source['metadata'])
    
    with col2:
        # System information
        st.header("System Info")

        # System status
        st.metric("System Status", "Online")
        st.metric("Query Mode", query_mode)
        
        # Quick stats
        if hasattr(rag_agent, 'index') and rag_agent.index:
            try:
                # Get some basic stats
                st.metric("Documents Loaded", "Ready")
                st.metric("Storage", "Persistent")
            except:
                pass

        # Help section
        st.header("Help")
        st.markdown("""
        **Chat Mode:**
        - Maintains conversation context
        - Natural conversation flow
        - Memory of previous questions
        
        **Query with Sources:**
        - Shows source citations
        - Similarity scores
        - Document references
        
        **Tips:**
        - Be specific in your questions
        - Use suggested questions as examples
        - Reset conversation to start fresh
        """)
        
        # About section
        with st.expander("About"):
            st.markdown("""
            This RAG (Retrieval-Augmented Generation) agent uses:
            - **LlamaParse** for PDF processing
            - **OpenAI GPT-4o-mini** for responses
            - **LlamaIndex** for vector storage
            - **Streamlit** for web interface
            
            Built for medical insurance policy queries with high accuracy and source attribution.
            """)

if __name__ == "__main__":
    main()
