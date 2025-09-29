"""
Example: Financial Analysis RAG System
Demonstrates how to configure the Generic RAG Template for financial documents.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RAGConfig, DomainType, create_config_for_domain
from generic_rag import GenericRAGSystem

def create_financial_rag():
    """Create a RAG system configured for financial documents."""
    
    # Create financial domain configuration
    config = create_config_for_domain(DomainType.FINANCIAL, "Financial Analysis System")
    
    # Customize for financial use case
    config.data_directory = "./examples/data/financial"
    config.storage_directory = "./examples/storage/financial"
    
    # Optimize for financial documents (larger chunks for context)
    config.chunking.chunk_size = 768
    config.chunking.chunk_overlap = 75
    config.retrieval.similarity_top_k = 5
    config.retrieval.temperature = 0.05  # More deterministic for financial data
    
    # Focus on Excel and CSV files for financial data
    config.supported_file_types = ["pdf", "excel", "csv", "json", "txt"]
    
    # Save configuration for reuse
    config.save_to_file("examples/financial_config.json")
    
    return config

def main():
    """Demo financial RAG system."""
    print("Financial Analysis RAG System Example")
    print("=" * 50)

    # Create configuration
    config = create_financial_rag()

    # Initialize RAG system
    rag = GenericRAGSystem(config)

    # Create data directory if it doesn't exist
    Path(config.data_directory).mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {config.data_directory}")
    print(f"Storage directory: {config.storage_directory}")
    print(f"Domain: {config.domain_config.name}")
    print(f"Description: {config.domain_config.description}")

    # Show configuration details
    print(f"\nConfiguration:")
    print(f"   - Chunk size: {config.chunking.chunk_size} characters")
    print(f"   - Chunk overlap: {config.chunking.chunk_overlap} characters")
    print(f"   - Retrieved chunks: {config.retrieval.similarity_top_k}")
    print(f"   - Temperature: {config.retrieval.temperature}")
    print(f"   - Supported files: {', '.join(config.supported_file_types)}")

    # Show suggested questions
    print(f"\nExample questions for financial documents:")
    for i, question in enumerate(config.domain_config.suggested_questions, 1):
        print(f"   {i}. {question}")

    # Show example file types
    print(f"\nIdeal file types for financial analysis:")
    print(f"   - Excel spreadsheets: Financial statements, budgets, forecasts")
    print(f"   - CSV files: Transaction data, time series, metrics")
    print(f"   - PDF reports: Annual reports, investment summaries")
    print(f"   - JSON files: API data, structured financial data")

    # Instructions for use
    print(f"\nTo use this configuration:")
    print(f"   1. Place financial documents in: {config.data_directory}")
    print(f"   2. Best formats: Excel (.xlsx), CSV, PDF reports")
    print(f"   3. Run: python examples/financial_example.py")
    print(f"   4. Or load config: RAGConfig.load_from_file('examples/financial_config.json')")
    
    # Check if data directory has files
    data_path = Path(config.data_directory)
    if data_path.exists():
        files = list(data_path.glob("*"))
        if files:
            print(f"\nFound {len(files)} files:")
            for file in files[:5]:  # Show first 5 files
                print(f"   - {file.name}")
            if len(files) > 5:
                print(f"   ... and {len(files) - 5} more")

            # Initialize and demo
            print(f"\nInitializing RAG system...")
            success = rag.initialize_system()

            if success:
                print("System initialized successfully!")

                # Demo queries
                demo_questions = [
                    "What are the key financial metrics?",
                    "How has revenue changed over time?",
                    "What are the major expenses?"
                ]

                for question in demo_questions:
                    print(f"\nDemo query: {question}")
                    response = rag.query(question)
                    print(f"Response: {response[:200]}..." if len(response) > 200 else f"Response: {response}")

                # Interactive mode
                print(f"\nInteractive mode (type 'quit' to exit):")
                while True:
                    try:
                        user_input = input("\nYour financial question: ").strip()
                        if user_input.lower() in ['quit', 'exit']:
                            break
                        if user_input:
                            response = rag.query(user_input)
                            print(f"Response: {response}")
                    except KeyboardInterrupt:
                        break
            else:
                print("ERROR: Failed to initialize system")
        else:
            print(f"\nNo files found in {config.data_directory}")
            print("   Add some financial documents to get started!")
            print("   Try: financial statements, expense reports, investment data")
    else:
        print(f"\nCreated data directory: {config.data_directory}")
        print("   Add some financial documents to get started!")

if __name__ == "__main__":
    main()
