"""
Example: Medical Records RAG System
Demonstrates how to configure the Generic RAG Template for medical documents.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RAGConfig, DomainType, create_config_for_domain
from generic_rag import GenericRAGSystem

def create_medical_rag():
    """Create a RAG system configured for medical documents."""
    
    # Create medical domain configuration
    config = create_config_for_domain(DomainType.MEDICAL, "Medical Records Analysis")
    
    # Customize for medical use case
    config.data_directory = "./examples/data/medical"
    config.storage_directory = "./examples/storage/medical"
    
    # Optimize for medical documents (smaller chunks for detailed analysis)
    config.chunking.chunk_size = 256
    config.chunking.chunk_overlap = 25
    config.retrieval.similarity_top_k = 7  # More context for medical queries
    
    # Save configuration for reuse
    config.save_to_file("examples/medical_config.json")
    
    return config

def main():
    """Demo medical RAG system."""
    print("Medical Records RAG System Example")
    print("=" * 50)

    # Create configuration
    config = create_medical_rag()

    # Initialize RAG system
    rag = GenericRAGSystem(config)

    # Create data directory if it doesn't exist
    Path(config.data_directory).mkdir(parents=True, exist_ok=True)

    print(f"Data directory: {config.data_directory}")
    print(f"Storage directory: {config.storage_directory}")
    print(f"Domain: {config.domain_config.name}")
    print(f"Description: {config.domain_config.description}")

    # Show suggested questions
    print(f"\nExample questions for medical documents:")
    for i, question in enumerate(config.domain_config.suggested_questions, 1):
        print(f"   {i}. {question}")

    # Instructions for use
    print(f"\nTo use this configuration:")
    print(f"   1. Place medical documents in: {config.data_directory}")
    print(f"   2. Supported formats: PDF, Excel, CSV, TXT, DOCX")
    print(f"   3. Run: python examples/medical_example.py")
    print(f"   4. Or load config: RAGConfig.load_from_file('examples/medical_config.json')")
    
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

                # Demo query
                demo_question = "What are the key findings in my test results?"
                print(f"\nDemo query: {demo_question}")
                response = rag.query(demo_question)
                print(f"Response: {response}")

                # Interactive mode
                print(f"\nInteractive mode (type 'quit' to exit):")
                while True:
                    try:
                        user_input = input("\nYour question: ").strip()
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
            print("   Add some medical documents to get started!")
    else:
        print(f"\nCreated data directory: {config.data_directory}")
        print("   Add some medical documents to get started!")

if __name__ == "__main__":
    main()
