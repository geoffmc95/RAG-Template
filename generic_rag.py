"""
Generic RAG System - Configurable for any domain and data type
Replaces the medical insurance specific implementation with a flexible, configurable system.
"""

from dotenv import load_dotenv
import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from llama_index.core import VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter, SemanticSplitterNodeParser, TokenTextSplitter
from llama_index.core.chat_engine import SimpleChatEngine, CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

from config import RAGConfig, DomainType, ChunkingConfig
from data_processor import DataProcessor

load_dotenv()

class GenericRAGSystem:
    """Generic RAG system that adapts to any domain and data type."""
    
    def __init__(self, config: RAGConfig = None, config_file: str = None):
        """Initialize the generic RAG system."""
        # Load configuration
        if config_file:
            self.config = RAGConfig.load_from_file(config_file)
        elif config:
            self.config = config
        else:
            self.config = RAGConfig()
        
        # Initialize components
        self.index = None
        self.query_engine = None
        self.chat_engine = None
        self.data_processor = None
        
        # Setup API keys
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.llamacloud_api_key = os.getenv("LLAMACLOUD_API_KEY")
        
        # Configure LLM settings
        self._setup_llm_settings()
        
        print(f"Initialized Generic RAG System for: {self.config.domain_config.name}")
    
    def _setup_llm_settings(self):
        """Configure LLM and embedding models."""
        try:
            # Ensure retrieval config is properly initialized
            if not hasattr(self.config, 'retrieval') or self.config.retrieval is None:
                from config import RetrievalConfig
                self.config.retrieval = RetrievalConfig()

            # Get temperature with fallback
            if hasattr(self.config.retrieval, 'temperature'):
                temperature = self.config.retrieval.temperature
            else:
                print("WARNING: retrieval config missing temperature, using default 0.1")
                temperature = 0.1

            Settings.llm = OpenAI(
                model=self.config.llm_model,
                temperature=temperature
            )
            Settings.embed_model = OpenAIEmbedding(model=self.config.embedding_model)

        except Exception as e:
            print(f"ERROR: Failed to setup LLM settings: {e}")
            # Use safe defaults
            Settings.llm = OpenAI(
                model="gpt-4o-mini",
                temperature=0.1
            )
            Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
    
    def _get_node_parser(self):
        """Get the appropriate node parser based on configuration."""
        chunking = self.config.chunking
        
        if chunking.strategy == "sentence":
            return SentenceSplitter(
                chunk_size=chunking.chunk_size,
                chunk_overlap=chunking.chunk_overlap,
                separator=chunking.separator
            )
        elif chunking.strategy == "token":
            return TokenTextSplitter(
                chunk_size=chunking.chunk_size,
                chunk_overlap=chunking.chunk_overlap
            )
        elif chunking.strategy == "semantic":
            return SemanticSplitterNodeParser(
                buffer_size=1,
                breakpoint_percentile_threshold=95,
                embed_model=Settings.embed_model
            )
        else:
            # Default to sentence splitter
            return SentenceSplitter(
                chunk_size=chunking.chunk_size,
                chunk_overlap=chunking.chunk_overlap,
                separator=chunking.separator
            )
    
    def create_or_load_index(self, force_rebuild: bool = False) -> bool:
        """Create a new index or load existing one."""
        storage_dir = Path(self.config.storage_directory)
        
        try:
            # Try to load existing index
            if not force_rebuild and storage_dir.exists():
                print("Loading existing index from storage...")
                storage_context = StorageContext.from_defaults(persist_dir=str(storage_dir))
                self.index = load_index_from_storage(storage_context)
                print("Index loaded successfully!")
                return True
        except Exception as e:
            print(f"WARNING: Could not load existing index: {e}")
            print("Creating new index...")
        
        # Create new index
        return self._create_new_index()
    
    def _create_new_index(self) -> bool:
        """Create a new index from documents."""
        try:
            print("Creating new index...")

            # Initialize data processor
            self.data_processor = DataProcessor(self.config, self.llamacloud_api_key)

            # Process all documents
            documents = self.data_processor.process_all_files()

            if not documents:
                print("ERROR: No documents found to process!")
                return False

            print(f"Processing {len(documents)} documents...")
            
            # Get node parser
            node_parser = self._get_node_parser()
            
            # Create index
            self.index = VectorStoreIndex.from_documents(
                documents,
                node_parser=node_parser,
                show_progress=True
            )
            
            # Persist the index
            storage_dir = Path(self.config.storage_directory)
            storage_dir.mkdir(exist_ok=True)
            self.index.storage_context.persist(persist_dir=str(storage_dir))
            
            print(f"Index saved to {storage_dir}")
            print("Index created successfully!")
            return True

        except Exception as e:
            print(f"ERROR: Error creating index: {e}")
            return False
    
    def setup_query_engine(self):
        """Setup the query engine with configured parameters."""
        if not self.index:
            print("ERROR: No index available. Please create or load an index first.")
            return False

        try:
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.config.retrieval.similarity_top_k,
                response_mode=self.config.retrieval.response_mode
            )
            print("Query engine ready!")
            return True
        except Exception as e:
            print(f"ERROR: Error setting up query engine: {e}")
            return False
    
    def setup_chat_engine(self):
        """Setup the chat engine with domain-specific configuration."""
        if not self.index:
            print("ERROR: No index available. Please create or load an index first.")
            return False

        try:
            # Create memory buffer
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

            # Create chat engine with domain-specific system prompt
            self.chat_engine = SimpleChatEngine.from_defaults(
                llm=Settings.llm,
                memory=memory,
                system_prompt=self.config.domain_config.system_prompt
            )

            print("Chat engine ready!")
            return True
        except Exception as e:
            print(f"ERROR: Error setting up chat engine: {e}")
            return False
    
    def initialize_system(self, force_rebuild: bool = False) -> bool:
        """Initialize the complete RAG system."""
        print(f"Initializing {self.config.project_name}...")
        print(f"Domain: {self.config.domain_config.name}")
        print(f"Data directory: {self.config.data_directory}")
        print(f"Storage directory: {self.config.storage_directory}")
        
        # Show data summary
        if not hasattr(self, 'data_processor') or not self.data_processor:
            self.data_processor = DataProcessor(self.config, self.llamacloud_api_key)
        
        summary = self.data_processor.get_processing_summary()
        print(f"Found {summary['total_files']} files to process")
        for file_type, info in summary['files_by_type'].items():
            print(f"   - {file_type}: {info['count']} files")
        
        # Create or load index
        if not self.create_or_load_index(force_rebuild):
            return False
        
        # Setup engines
        if not self.setup_query_engine():
            return False
        
        if not self.setup_chat_engine():
            return False
        
        print("Generic RAG system initialized successfully!")
        return True
    
    def query(self, question: str) -> str:
        """Query the RAG system."""
        if not self.query_engine:
            return "ERROR: Query engine not initialized. Please initialize the system first."
        
        try:
            response = self.query_engine.query(question)
            return str(response)
        except Exception as e:
            return f"ERROR: Error processing query: {e}"
    
    def chat(self, message: str) -> str:
        """Chat with the RAG system."""
        if not self.chat_engine or not self.query_engine:
            return "ERROR: Chat engine not initialized. Please initialize the system first."
        
        try:
            # First get relevant information using query engine
            query_response = self.query_engine.query(message)
            
            # Then use chat engine with the retrieved information
            enhanced_message = f"Based on the document information: {query_response}\n\nUser question: {message}"
            chat_response = self.chat_engine.chat(enhanced_message)
            
            return str(chat_response)
        except Exception as e:
            return f"ERROR: Error processing chat message: {e}"
    
    def get_suggested_questions(self) -> List[str]:
        """Get domain-specific suggested questions."""
        return self.config.domain_config.suggested_questions
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get information about the current system configuration."""
        return {
            "project_name": self.config.project_name,
            "domain": self.config.domain.value,
            "domain_name": self.config.domain_config.name,
            "domain_description": self.config.domain_config.description,
            "llm_model": self.config.llm_model,
            "embedding_model": self.config.embedding_model,
            "supported_file_types": self.config.supported_file_types,
            "chunking_strategy": self.config.chunking.strategy,
            "chunk_size": self.config.chunking.chunk_size,
            "similarity_top_k": self.config.retrieval.similarity_top_k,
            "has_index": self.index is not None,
            "has_query_engine": self.query_engine is not None,
            "has_chat_engine": self.chat_engine is not None
        }
    
    def reset_chat_memory(self):
        """Reset the chat memory."""
        if self.chat_engine and hasattr(self.chat_engine, 'memory'):
            self.chat_engine.memory.reset()
            print("Chat memory reset!")

def create_rag_system_for_domain(domain: DomainType, project_name: str = None, data_dir: str = "./data") -> GenericRAGSystem:
    """Create a RAG system configured for a specific domain."""
    from config import create_config_for_domain
    
    config = create_config_for_domain(domain, project_name)
    config.data_directory = data_dir
    
    return GenericRAGSystem(config)

def main():
    """Demo the generic RAG system."""
    print("Generic RAG System Demo")
    print("=" * 50)
    
    # Create a general-purpose RAG system
    rag = GenericRAGSystem()
    
    # Initialize the system
    start_time = time.time()
    success = rag.initialize_system()
    setup_time = time.time() - start_time
    
    if not success:
        print("ERROR: Failed to initialize RAG system")
        return
    
    print(f"Setup completed in {setup_time:.2f} seconds")
    
    # Show system info
    info = rag.get_system_info()
    print(f"\nSystem Information:")
    print(f"   Project: {info['project_name']}")
    print(f"   Domain: {info['domain_name']}")
    print(f"   LLM: {info['llm_model']}")
    print(f"   Chunking: {info['chunking_strategy']} ({info['chunk_size']} chars)")
    
    # Test queries
    print(f"\nSuggested questions:")
    for i, question in enumerate(rag.get_suggested_questions(), 1):
        print(f"   {i}. {question}")
    
    # Interactive demo
    print(f"\nTry asking a question (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("Goodbye!")
                break
            
            if not user_input:
                continue
            
            print("Thinking...")
            start_time = time.time()
            response = rag.query(user_input)
            response_time = time.time() - start_time
            
            print(f"\nRAG: {response}")
            print(f"Response time: {response_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
