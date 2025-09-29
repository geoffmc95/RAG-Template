"""
Example: Research Papers RAG System
Demonstrates how to configure the Generic RAG Template for academic research documents.
"""

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config import RAGConfig, DomainType, create_config_for_domain
from generic_rag import GenericRAGSystem

def create_research_rag():
    """Create a RAG system configured for research documents."""
    
    # Create research domain configuration
    config = create_config_for_domain(DomainType.RESEARCH, "Research Paper Analysis")
    
    # Customize for research use case
    config.data_directory = "./examples/data/research"
    config.storage_directory = "./examples/storage/research"
    
    # Optimize for research papers (medium chunks for balanced context)
    config.chunking.chunk_size = 512
    config.chunking.chunk_overlap = 50
    config.chunking.strategy = "semantic"  # Better for academic content
    config.retrieval.similarity_top_k = 6
    config.retrieval.temperature = 0.2  # Slightly more creative for research insights
    
    # Focus on academic document formats
    config.supported_file_types = ["pdf", "txt", "docx", "md", "csv"]
    
    # Save configuration for reuse
    config.save_to_file("examples/research_config.json")
    
    return config

def main():
    """Demo research RAG system."""
    print("Research Papers RAG System Example")
    print("=" * 50)

    # Create configuration
    config = create_research_rag()

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
    print(f"   - Chunking strategy: {config.chunking.strategy}")
    print(f"   - Chunk size: {config.chunking.chunk_size} characters")
    print(f"   - Chunk overlap: {config.chunking.chunk_overlap} characters")
    print(f"   - Retrieved chunks: {config.retrieval.similarity_top_k}")
    print(f"   - Temperature: {config.retrieval.temperature}")
    print(f"   - Supported files: {', '.join(config.supported_file_types)}")

    # Show suggested questions
    print(f"\nExample questions for research documents:")
    for i, question in enumerate(config.domain_config.suggested_questions, 1):
        print(f"   {i}. {question}")

    # Show example research areas
    print(f"\nIdeal for research areas:")
    print(f"   - Literature reviews and meta-analyses")
    print(f"   - Experimental studies and clinical trials")
    print(f"   - Survey research and data analysis")
    print(f"   - Theoretical papers and reviews")
    print(f"   - Conference proceedings and abstracts")

    # Show file type recommendations
    print(f"\nRecommended file types:")
    print(f"   - PDF: Published papers, preprints, reports")
    print(f"   - DOCX: Draft papers, manuscripts")
    print(f"   - TXT/MD: Notes, summaries, abstracts")
    print(f"   - CSV: Research data, survey results")

    # Instructions for use
    print(f"\nTo use this configuration:")
    print(f"   1. Place research documents in: {config.data_directory}")
    print(f"   2. Best formats: PDF papers, DOCX manuscripts, CSV data")
    print(f"   3. Run: python examples/research_example.py")
    print(f"   4. Or load config: RAGConfig.load_from_file('examples/research_config.json')")
    
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

                # Demo queries for research
                demo_questions = [
                    "What are the main research findings?",
                    "What methodology was used in this study?",
                    "What are the key conclusions and implications?",
                    "Are there any limitations mentioned?",
                    "What future research directions are suggested?"
                ]

                print(f"\nDemo queries:")
                for i, question in enumerate(demo_questions[:3], 1):
                    print(f"\n{i}. {question}")
                    response = rag.query(question)
                    print(f"   Response: {response[:150]}..." if len(response) > 150 else f"   Response: {response}")

                # Interactive mode
                print(f"\nInteractive mode (type 'quit' to exit):")
                print("Try asking about methodology, findings, conclusions, or limitations...")

                while True:
                    try:
                        user_input = input("\nYour research question: ").strip()
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
            print("   Add some research documents to get started!")
            print("   Try: research papers, conference proceedings, survey data")
    else:
        print(f"\nCreated data directory: {config.data_directory}")
        print("   Add some research documents to get started!")

if __name__ == "__main__":
    main()
