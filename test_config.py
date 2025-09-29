#!/usr/bin/env python3
"""
Test script to debug configuration issues.
"""

from config import RAGConfig, DomainType, create_config_for_domain
from generic_rag import GenericRAGSystem

def test_default_config():
    """Test default configuration creation."""
    print("Testing default RAGConfig creation...")
    try:
        config = RAGConfig()
        print(f"✓ Default config created successfully")
        print(f"  - Domain: {config.domain}")
        print(f"  - Chunking: {config.chunking}")
        print(f"  - Retrieval: {config.retrieval}")
        print(f"  - Temperature: {config.retrieval.temperature}")
        return config
    except Exception as e:
        print(f"✗ Error creating default config: {e}")
        return None

def test_domain_config():
    """Test domain-specific configuration creation."""
    print("\nTesting domain-specific config creation...")
    try:
        config = create_config_for_domain(DomainType.MEDICAL, "Test Medical RAG")
        print(f"✓ Medical config created successfully")
        print(f"  - Domain: {config.domain}")
        print(f"  - Project name: {config.project_name}")
        print(f"  - Chunking: {config.chunking}")
        print(f"  - Retrieval: {config.retrieval}")
        print(f"  - Temperature: {config.retrieval.temperature}")
        return config
    except Exception as e:
        print(f"✗ Error creating domain config: {e}")
        return None

def test_rag_system_init(config):
    """Test RAG system initialization."""
    print(f"\nTesting RAG system initialization...")
    try:
        rag = GenericRAGSystem(config)
        print(f"✓ RAG system created successfully")
        print(f"  - Config type: {type(rag.config)}")
        print(f"  - Retrieval type: {type(rag.config.retrieval)}")
        print(f"  - Temperature: {rag.config.retrieval.temperature}")
        return rag
    except Exception as e:
        print(f"✗ Error creating RAG system: {e}")
        return None

def test_config_serialization():
    """Test configuration save/load."""
    print(f"\nTesting configuration serialization...")
    try:
        # Create and save config
        config = RAGConfig()
        config.save_to_file("test_config.json")
        print(f"✓ Config saved successfully")
        
        # Load config
        loaded_config = RAGConfig.load_from_file("test_config.json")
        print(f"✓ Config loaded successfully")
        print(f"  - Retrieval type: {type(loaded_config.retrieval)}")
        print(f"  - Temperature: {loaded_config.retrieval.temperature}")
        
        # Clean up
        import os
        os.remove("test_config.json")
        print(f"✓ Test file cleaned up")
        
        return loaded_config
    except Exception as e:
        print(f"✗ Error with config serialization: {e}")
        return None

def main():
    """Run all tests."""
    print("Configuration Debug Tests")
    print("=" * 50)
    
    # Test 1: Default config
    default_config = test_default_config()
    
    # Test 2: Domain config
    domain_config = test_domain_config()
    
    # Test 3: RAG system with default config
    if default_config:
        test_rag_system_init(default_config)
    
    # Test 4: RAG system with domain config
    if domain_config:
        test_rag_system_init(domain_config)
    
    # Test 5: Serialization
    test_config_serialization()
    
    print(f"\nAll tests completed!")

if __name__ == "__main__":
    main()
