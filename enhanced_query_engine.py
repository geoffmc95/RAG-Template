"""
Enhanced Query Engine with Citations and Advanced Features
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.query_engine import CitationQueryEngine, RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.response_synthesizers import get_response_synthesizer
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
import time

load_dotenv()

# Configure API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
llamacloud_api_key = os.getenv("LLAMACLOUD_API_KEY")

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class EnhancedInsuranceRAG:
    """Enhanced RAG system with multiple query engines and citation support."""
    
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = storage_dir
        self.index = None
        self.citation_query_engine = None
        self.retriever_query_engine = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the enhanced RAG system."""
        print("🚀 Initializing Enhanced Insurance RAG System...")
        
        # Load or create index
        self.index = self._load_or_create_index()
        
        # Create multiple query engines
        self._create_query_engines()
        
        print("✅ Enhanced RAG system ready!")
    
    def _load_or_create_index(self):
        """Load existing index or create new one."""
        if Path(self.storage_dir).exists():
            try:
                print("📂 Loading existing index...")
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                index = load_index_from_storage(storage_context)
                print("✅ Index loaded successfully!")
                return index
            except Exception as e:
                print(f"❌ Error loading index: {e}")
                print("🔄 Creating new index...")
        
        return self._create_new_index()
    
    def _create_new_index(self):
        """Create a new index from documents."""
        print("🔄 Processing documents with LlamaParse...")
        
        # Initialize LlamaParse
        parser = LlamaParse(
            api_key=llamacloud_api_key,
            result_type="markdown",
            verbose=False,
            language="en",
            parsing_instruction="Extract insurance policy details, coverage amounts, deductibles, and benefit information clearly."
        )
        
        # Load documents
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            "data", 
            file_extractor=file_extractor
        ).load_data()
        
        print(f"📄 Processed {len(documents)} document sections")
        
        # Configure chunking
        node_parser = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=50,
            separator=" "
        )
        
        # Create index
        index = VectorStoreIndex.from_documents(
            documents, 
            node_parser=node_parser,
            show_progress=True
        )
        
        # Persist index
        print("💾 Saving index...")
        index.storage_context.persist(persist_dir=self.storage_dir)
        print(f"✅ Index saved to {self.storage_dir}")
        
        return index
    
    def _create_query_engines(self):
        """Create different types of query engines."""
        print("🔧 Creating enhanced query engines...")
        
        # 1. Citation Query Engine - provides source citations
        self.citation_query_engine = CitationQueryEngine.from_args(
            index=self.index,
            similarity_top_k=8,
            citation_chunk_size=512,
        )
        
        # 2. Custom Retriever Query Engine - more control over retrieval
        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )
        
        response_synthesizer = get_response_synthesizer(
            response_mode="compact",
            use_async=False,
        )
        
        self.retriever_query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=response_synthesizer,
        )
        
        print("✅ Query engines created!")
    
    def query_with_citations(self, question):
        """Query with source citations."""
        print(f"🔍 Querying with citations: {question}")
        start_time = time.time()
        
        response = self.citation_query_engine.query(question)
        
        query_time = time.time() - start_time
        
        print(f"⏱️  Query time: {query_time:.2f} seconds")
        print(f"📝 Response: {response}")
        
        # Extract and display sources
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print("\n📚 Sources:")
            for i, node in enumerate(response.source_nodes, 1):
                print(f"   {i}. Page {node.metadata.get('source', 'Unknown')}")
                print(f"      Text: {node.text[:100]}...")
        
        return response
    
    def query_with_retriever(self, question):
        """Query using custom retriever."""
        print(f"🔍 Querying with custom retriever: {question}")
        start_time = time.time()
        
        response = self.retriever_query_engine.query(question)
        
        query_time = time.time() - start_time
        
        print(f"⏱️  Query time: {query_time:.2f} seconds")
        print(f"📝 Response: {response}")
        
        # Display retrieved nodes
        if hasattr(response, 'source_nodes') and response.source_nodes:
            print(f"\n📊 Retrieved {len(response.source_nodes)} relevant chunks")
            for i, node in enumerate(response.source_nodes[:3], 1):  # Show top 3
                print(f"   {i}. Score: {node.score:.3f} | Page: {node.metadata.get('source', 'Unknown')}")
                print(f"      Text: {node.text[:100]}...")
        
        return response
    
    def compare_engines(self, question):
        """Compare responses from different engines."""
        print(f"\n{'='*60}")
        print(f"🔬 COMPARING QUERY ENGINES")
        print(f"Question: {question}")
        print(f"{'='*60}")
        
        print("\n1️⃣ CITATION ENGINE:")
        print("-" * 40)
        citation_response = self.query_with_citations(question)
        
        print("\n2️⃣ RETRIEVER ENGINE:")
        print("-" * 40)
        retriever_response = self.query_with_retriever(question)
        
        return citation_response, retriever_response

def main():
    """Main function to demonstrate enhanced query engines."""
    rag = EnhancedInsuranceRAG()
    
    # Test questions
    test_questions = [
        "What is my dental coverage limit?",
        "What vision benefits do I have?",
        "Are there any exclusions for travel insurance?",
        "How do I submit a claim?"
    ]
    
    print("\n" + "="*60)
    print("🏥 ENHANCED INSURANCE RAG SYSTEM DEMO")
    print("="*60)
    
    for question in test_questions:
        rag.compare_engines(question)
        print("\n" + "="*60)
        
        # Pause between questions
        input("Press Enter to continue to next question...")

if __name__ == "__main__":
    main()
