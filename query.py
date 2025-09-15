from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()  # This loads variables from .env in the project root

# Configure API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
llamacloud_api_key = os.getenv("LLAMACLOUD_API_KEY")

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

def create_enhanced_rag_system(force_rebuild=False):
    """Create an enhanced RAG system with LlamaParse and persistent storage."""

    # Define storage directory
    storage_dir = "./storage"

    # Check if index already exists and load it
    if not force_rebuild and Path(storage_dir).exists():
        print("Loading existing index from storage...")
        try:
            # Load the existing index
            storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            index = load_index_from_storage(storage_context)
            print("Successfully loaded existing index!")
        except Exception as e:
            print(f"Error loading existing index: {e}")
            print("Creating new index...")
            force_rebuild = True

    if force_rebuild or not Path(storage_dir).exists():
        print("Creating new index with LlamaParse...")

        # Initialize LlamaParse for better PDF extraction
        parser = LlamaParse(
            api_key=llamacloud_api_key,
            result_type="markdown",  # Better for structured content
            verbose=True,
            language="en",
            parsing_instruction="Focus on extracting insurance policy details, coverage amounts, deductibles, and benefit information clearly."
        )

        # Use SimpleDirectoryReader with LlamaParse
        file_extractor = {".pdf": parser}
        documents = SimpleDirectoryReader(
            "data",
            file_extractor=file_extractor
        ).load_data()

        print(f"Loaded {len(documents)} documents")

        # Configure advanced chunking
        node_parser = SentenceSplitter(
            chunk_size=512,  # Smaller chunks for better precision
            chunk_overlap=50,  # Some overlap to maintain context
            separator=" "
        )

        # Create index with custom node parser
        index = VectorStoreIndex.from_documents(
            documents,
            node_parser=node_parser,
            show_progress=True
        )

        # Persist the index
        print("Persisting index to storage...")
        index.storage_context.persist(persist_dir=storage_dir)
        print(f"Index saved to {storage_dir}")

    # Create query engine with better configuration
    query_engine = index.as_query_engine(
        similarity_top_k=5,  # Retrieve more relevant chunks
        response_mode="compact"  # Better response synthesis
    )

    return query_engine

def main():
    """Main function to demonstrate the enhanced RAG system."""
    import time

    print("Creating enhanced RAG system with persistent storage...")
    start_time = time.time()

    # Create the RAG system (will load from storage if available)
    query_engine = create_enhanced_rag_system()

    setup_time = time.time() - start_time
    print(f"Setup completed in {setup_time:.2f} seconds")

    # Test queries
    test_queries = [
        "How much dental insurance coverage do I receive per year?",
        "What is my deductible for medical expenses?",
        "What vision benefits are included in my plan?",
        "Are there any exclusions I should be aware of?"
    ]

    print("\n" + "="*50)
    print("TESTING ENHANCED RAG SYSTEM")
    print("="*50)

    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("-" * 40)
        response = query_engine.query(query)
        print(f"Response: {response}")
        print()

def rebuild_index():
    """Utility function to force rebuild the index."""
    print("Force rebuilding index...")
    query_engine = create_enhanced_rag_system(force_rebuild=True)
    print("Index rebuilt successfully!")
    return query_engine

if __name__ == "__main__":
    main()