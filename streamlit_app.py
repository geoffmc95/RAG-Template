"""
Streamlit Web Interface for Generic RAG Template
Configurable for any domain and data type.
"""

import streamlit as st
import time
import json
from pathlib import Path
from dotenv import load_dotenv
import os
from typing import Dict, Any, List

from config import RAGConfig, DomainType, create_config_for_domain
from generic_rag import GenericRAGSystem
from data_processor import DataProcessor

# Load environment variables
load_dotenv()

# Configure page
st.set_page_config(
    page_title="Generic RAG Template",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitGenericRAG:
    """Streamlit-based Generic RAG System."""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.rag_system = None
        self.is_initialized = False

    def initialize_system(self, force_rebuild: bool = False):
        """Initialize the RAG system."""
        try:
            # Debug: Check config before initialization
            st.write(f"DEBUG: Config type: {type(self.config)}")
            st.write(f"DEBUG: Retrieval type: {type(self.config.retrieval)}")
            st.write(f"DEBUG: Retrieval value: {self.config.retrieval}")

            self.rag_system = GenericRAGSystem(self.config)
            success = self.rag_system.initialize_system(force_rebuild)
            self.is_initialized = success
            return success
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            st.write(f"DEBUG: Full error details: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
            return False

    def query_with_sources(self, question):
        """Query with source information."""
        if not self.is_initialized or not self.rag_system:
            return "RAG system not initialized. Please check your setup.", []

        try:
            response = self.rag_system.query_engine.query(question)
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
        if not self.is_initialized or not self.rag_system:
            return "RAG system not initialized. Please check your setup."

        try:
            return self.rag_system.chat(question)
        except Exception as e:
            return f"Error processing chat query: {e}"

    def get_suggested_questions(self):
        """Get domain-specific suggested questions."""
        if self.config:
            return self.config.domain_config.suggested_questions
        return []

    def get_system_info(self):
        """Get system information."""
        if self.rag_system:
            return self.rag_system.get_system_info()
        return {}

    def reset_chat_memory(self):
        """Reset chat memory."""
        if self.rag_system:
            self.rag_system.reset_chat_memory()

# Configuration and system setup functions
def load_or_create_config() -> RAGConfig:
    """Load configuration from session state or create default."""
    if 'rag_config' not in st.session_state:
        st.session_state.rag_config = RAGConfig()
    return st.session_state.rag_config

@st.cache_resource
def get_rag_agent(config_json: str):
    """Get RAG agent with caching based on configuration."""
    try:
        config_dict = json.loads(config_json)

        # Reconstruct config properly
        config = RAGConfig()

        # Handle simple attributes
        for key, value in config_dict.items():
            if hasattr(config, key) and key not in ['chunking', 'retrieval', 'domain_config', 'domain']:
                setattr(config, key, value)

        # Handle domain enum
        if 'domain' in config_dict:
            from config import DomainType
            if isinstance(config_dict['domain'], str):
                config.domain = DomainType(config_dict['domain'])
            else:
                config.domain = config_dict['domain']

        # Handle nested objects - let __post_init__ handle them properly
        # by triggering re-initialization
        config.__post_init__()

        # Override with any specific values from the dict
        if 'chunking' in config_dict and isinstance(config_dict['chunking'], dict):
            for key, value in config_dict['chunking'].items():
                if hasattr(config.chunking, key):
                    setattr(config.chunking, key, value)

        if 'retrieval' in config_dict and isinstance(config_dict['retrieval'], dict):
            for key, value in config_dict['retrieval'].items():
                if hasattr(config.retrieval, key):
                    setattr(config.retrieval, key, value)

        agent = StreamlitGenericRAG(config)
        return agent
    except Exception as e:
        st.error(f"Error creating RAG agent: {e}")
        # Return agent with default config as fallback
        return StreamlitGenericRAG(RAGConfig())

def show_configuration_sidebar():
    """Show configuration options in sidebar."""
    with st.sidebar:
        st.header("Configuration")

        # Domain selection
        domain_options = {
            "General": DomainType.GENERAL,
            "Medical": DomainType.MEDICAL,
            "Financial": DomainType.FINANCIAL,
            "Legal": DomainType.LEGAL,
            "Research": DomainType.RESEARCH,
            "Technical": DomainType.TECHNICAL
        }

        selected_domain = st.selectbox(
            "Select Domain:",
            options=list(domain_options.keys()),
            help="Choose the domain that best matches your documents"
        )

        # Project name
        project_name = st.text_input(
            "Project Name:",
            value=st.session_state.get('project_name', 'My RAG System'),
            help="Give your RAG system a descriptive name"
        )

        # Advanced settings
        with st.expander("Advanced Settings"):
            chunk_size = st.slider("Chunk Size", 256, 1024, 512)
            chunk_overlap = st.slider("Chunk Overlap", 0, 100, 50)
            similarity_top_k = st.slider("Retrieved Chunks", 3, 10, 5)
            temperature = st.slider("Temperature", 0.0, 1.0, 0.1)

        # Update configuration
        if st.button("Update Configuration"):
            try:
                config = RAGConfig()
                config.domain = domain_options[selected_domain]
                config.project_name = project_name

                # Ensure nested configs are initialized
                if config.chunking is None:
                    from config import ChunkingConfig
                    config.chunking = ChunkingConfig()
                if config.retrieval is None:
                    from config import RetrievalConfig
                    config.retrieval = RetrievalConfig()

                config.chunking.chunk_size = chunk_size
                config.chunking.chunk_overlap = chunk_overlap
                config.retrieval.similarity_top_k = similarity_top_k
                config.retrieval.temperature = temperature

                st.session_state.rag_config = config
                st.session_state.project_name = project_name
                st.success("Configuration updated!")
                st.rerun()
            except Exception as e:
                st.error(f"Error updating configuration: {e}")

        # Return configuration with safe defaults
        try:
            return st.session_state.get('rag_config', RAGConfig())
        except Exception as e:
            st.error(f"Error loading configuration: {e}")
            return RAGConfig()

def main():
    """Main Streamlit application."""

    # Header
    st.title("🤖 Generic RAG Template")
    st.markdown("Upload any documents and get intelligent answers tailored to your domain.")

    # Configuration sidebar
    config = show_configuration_sidebar()

    # File upload section
    st.header("Document Upload")
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['pdf', 'xlsx', 'xls', 'csv', 'txt', 'docx', 'json', 'md'],
        help="Upload documents in various formats. The system will automatically process them."
    )

    # Handle file uploads
    if uploaded_files:
        data_dir = Path(config.data_directory)
        data_dir.mkdir(exist_ok=True)

        for uploaded_file in uploaded_files:
            file_path = data_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

        st.success(f"Uploaded {len(uploaded_files)} files to {config.data_directory}")

    # Initialize RAG system
    if st.button("Initialize RAG System") or st.session_state.get('system_initialized', False):
        with st.spinner("Initializing RAG system..."):
            # Use proper serialization
            from dataclasses import asdict
            config_dict = asdict(config)
            # Convert enum to string for JSON serialization
            if hasattr(config, 'domain'):
                config_dict['domain'] = config.domain.value
            config_json = json.dumps(config_dict, default=str)
            rag_agent = get_rag_agent(config_json)

            force_rebuild = st.session_state.get('force_rebuild', False)
            success = rag_agent.initialize_system(force_rebuild)

            if success:
                st.session_state.system_initialized = True
                st.session_state.rag_agent = rag_agent
                st.success("RAG system initialized successfully!")
            else:
                st.error("Failed to initialize RAG system. Please check your documents and configuration.")
                return

    # Main interface (only show if system is initialized)
    if st.session_state.get('system_initialized', False):
        rag_agent = st.session_state.get('rag_agent')

        # Sidebar for query options
        with st.sidebar:
            st.header("Query Options")

            # Query mode selection
            query_mode = st.selectbox(
                "Select Query Mode:",
                ["Chat Mode", "Query with Sources"],
                help="Chat mode maintains conversation context, Query mode shows source citations"
            )

            # Suggested questions
            st.header("Suggested Questions")
            suggested_questions = rag_agent.get_suggested_questions()

            for i, question in enumerate(suggested_questions):
                if st.button(question, key=f"btn_{i}"):
                    st.session_state.selected_question = question

            # Reset chat button
            if st.button("Reset Conversation"):
                rag_agent.reset_chat_memory()
                st.session_state.messages = []
                st.success("Conversation reset!")

            # Force rebuild button
            if st.button("Rebuild Index"):
                st.session_state.force_rebuild = True
                st.session_state.system_initialized = False
                st.success("Will rebuild index on next initialization")

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
                    user_input = st.chat_input("Ask about your documents...")

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
                    user_input = st.text_input("Enter your question:", placeholder="What information can you find?")

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

            system_info = rag_agent.get_system_info()

            # System status
            st.metric("System Status", "Online" if system_info.get('has_index') else "Offline")
            st.metric("Query Mode", query_mode)
            st.metric("Domain", system_info.get('domain_name', 'Unknown'))

            # Configuration info
            with st.expander("Configuration Details"):
                st.write(f"**Project:** {system_info.get('project_name', 'N/A')}")
                st.write(f"**LLM Model:** {system_info.get('llm_model', 'N/A')}")
                st.write(f"**Embedding Model:** {system_info.get('embedding_model', 'N/A')}")
                st.write(f"**Chunk Size:** {system_info.get('chunk_size', 'N/A')}")
                st.write(f"**Chunking Strategy:** {system_info.get('chunking_strategy', 'N/A')}")
                st.write(f"**Retrieved Chunks:** {system_info.get('similarity_top_k', 'N/A')}")

            # Help section
            st.header("❓ Help")
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
            - Upload documents first
            - Configure domain for better results
            - Use suggested questions as examples
            - Reset conversation to start fresh
            """)

            # About section
            with st.expander("About Generic RAG Template"):
                st.markdown(f"""
                This Generic RAG Template adapts to any domain and data type:

                **Current Configuration:**
                - **Domain:** {system_info.get('domain_name', 'General')}
                - **Description:** {config.domain_config.description}

                **Supported File Types:**
                - PDF documents
                - Excel spreadsheets (.xlsx, .xls)
                - CSV files
                - Text files (.txt, .md)
                - Word documents (.docx)
                - JSON files

                **Technology Stack:**
                - **LlamaParse** for advanced document processing
                - **OpenAI GPT-4o-mini** for intelligent responses
                - **LlamaIndex** for vector storage and retrieval
                - **Streamlit** for web interface

                Built to be flexible and adaptable to any use case!
                """)

    else:
        # Show instructions if system not initialized
        st.info("Upload your documents and click 'Initialize RAG System' to get started!")

        # Show example use cases
        st.header("Example Use Cases")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.subheader("Medical Records")
            st.write("Upload test results, medical reports, or health documents to get insights about your health data.")

        with col2:
            st.subheader("Financial Analysis")
            st.write("Upload financial statements, investment reports, or expense data for financial insights.")

        with col3:
            st.subheader("Research Papers")
            st.write("Upload academic papers or research documents to extract key findings and insights.")

if __name__ == "__main__":
    main()
