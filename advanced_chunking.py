"""
Advanced Chunking Strategies for Insurance Documents
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, Document
from llama_index.core.node_parser import (
    SentenceSplitter, 
    SemanticSplitterNodeParser,
    TokenTextSplitter,
    SimpleNodeParser
)
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

class AdvancedChunkingDemo:
    """Demonstrate different chunking strategies for insurance documents."""
    
    def __init__(self):
        self.documents = None
        self.load_documents()
    
    def load_documents(self):
        """Load documents using LlamaParse."""
        print("📄 Loading documents with LlamaParse...")
        
        parser = LlamaParse(
            api_key=llamacloud_api_key,
            result_type="markdown",
            verbose=False,
            language="en",
            parsing_instruction="Extract insurance policy details clearly, maintaining structure."
        )
        
        file_extractor = {".pdf": parser}
        self.documents = SimpleDirectoryReader(
            "data", 
            file_extractor=file_extractor
        ).load_data()
        
        print(f"✅ Loaded {len(self.documents)} document sections")
    
    def create_sentence_splitter_chunks(self, chunk_size=512, chunk_overlap=50):
        """Create chunks using SentenceSplitter."""
        print(f"\n🔪 Creating chunks with SentenceSplitter (size={chunk_size}, overlap={chunk_overlap})")
        
        node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
        
        start_time = time.time()
        nodes = node_parser.get_nodes_from_documents(self.documents)
        processing_time = time.time() - start_time
        
        print(f"✅ Created {len(nodes)} chunks in {processing_time:.2f} seconds")
        
        # Show sample chunks
        self._show_sample_chunks(nodes, "SentenceSplitter")
        
        return nodes
    
    def create_semantic_chunks(self, buffer_size=1, breakpoint_percentile_threshold=95):
        """Create chunks using SemanticSplitterNodeParser."""
        print(f"\n🧠 Creating semantic chunks (buffer={buffer_size}, threshold={breakpoint_percentile_threshold})")
        
        try:
            node_parser = SemanticSplitterNodeParser(
                buffer_size=buffer_size,
                breakpoint_percentile_threshold=breakpoint_percentile_threshold,
                embed_model=Settings.embed_model
            )
            
            start_time = time.time()
            nodes = node_parser.get_nodes_from_documents(self.documents)
            processing_time = time.time() - start_time
            
            print(f"✅ Created {len(nodes)} semantic chunks in {processing_time:.2f} seconds")
            
            # Show sample chunks
            self._show_sample_chunks(nodes, "SemanticSplitter")
            
            return nodes
        except Exception as e:
            print(f"❌ Error creating semantic chunks: {e}")
            return []
    
    def create_token_chunks(self, chunk_size=400, chunk_overlap=40):
        """Create chunks using TokenTextSplitter."""
        print(f"\n🔤 Creating token-based chunks (size={chunk_size}, overlap={chunk_overlap})")
        
        node_parser = TokenTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" "
        )
        
        start_time = time.time()
        nodes = node_parser.get_nodes_from_documents(self.documents)
        processing_time = time.time() - start_time
        
        print(f"✅ Created {len(nodes)} token chunks in {processing_time:.2f} seconds")
        
        # Show sample chunks
        self._show_sample_chunks(nodes, "TokenSplitter")
        
        return nodes
    
    def create_simple_chunks(self, chunk_size=1024):
        """Create chunks using SimpleNodeParser."""
        print(f"\n📝 Creating simple chunks (size={chunk_size})")
        
        node_parser = SimpleNodeParser.from_defaults(
            chunk_size=chunk_size,
            chunk_overlap=0
        )
        
        start_time = time.time()
        nodes = node_parser.get_nodes_from_documents(self.documents)
        processing_time = time.time() - start_time
        
        print(f"✅ Created {len(nodes)} simple chunks in {processing_time:.2f} seconds")
        
        # Show sample chunks
        self._show_sample_chunks(nodes, "SimpleParser")
        
        return nodes
    
    def _show_sample_chunks(self, nodes, parser_name, num_samples=3):
        """Show sample chunks from the parser."""
        print(f"\n📋 Sample chunks from {parser_name}:")
        print("-" * 50)
        
        for i, node in enumerate(nodes[:num_samples]):
            print(f"\nChunk {i+1} (Length: {len(node.text)} chars):")
            print(f"Text: {node.text[:200]}...")
            if hasattr(node, 'metadata') and node.metadata:
                print(f"Metadata: {node.metadata}")
    
    def compare_chunking_strategies(self):
        """Compare different chunking strategies."""
        print("\n" + "="*60)
        print("🔬 COMPARING CHUNKING STRATEGIES")
        print("="*60)
        
        strategies = [
            ("Sentence Splitter (512/50)", lambda: self.create_sentence_splitter_chunks(512, 50)),
            ("Sentence Splitter (256/25)", lambda: self.create_sentence_splitter_chunks(256, 25)),
            ("Token Splitter (400/40)", lambda: self.create_token_chunks(400, 40)),
            ("Simple Parser (1024)", lambda: self.create_simple_chunks(1024)),
            ("Semantic Splitter", lambda: self.create_semantic_chunks(1, 95))
        ]
        
        results = {}
        
        for strategy_name, strategy_func in strategies:
            print(f"\n{'='*60}")
            print(f"Testing: {strategy_name}")
            print("="*60)
            
            try:
                nodes = strategy_func()
                results[strategy_name] = {
                    'num_chunks': len(nodes),
                    'avg_chunk_size': sum(len(node.text) for node in nodes) / len(nodes) if nodes else 0,
                    'nodes': nodes
                }
            except Exception as e:
                print(f"❌ Error with {strategy_name}: {e}")
                results[strategy_name] = {'error': str(e)}
        
        # Summary
        print(f"\n{'='*60}")
        print("📊 CHUNKING STRATEGY SUMMARY")
        print("="*60)
        
        for strategy, result in results.items():
            if 'error' in result:
                print(f"{strategy}: ERROR - {result['error']}")
            else:
                print(f"{strategy}:")
                print(f"  - Number of chunks: {result['num_chunks']}")
                print(f"  - Average chunk size: {result['avg_chunk_size']:.0f} characters")
        
        return results
    
    def test_retrieval_quality(self, nodes, strategy_name, test_query="What is my dental coverage?"):
        """Test retrieval quality for a chunking strategy."""
        print(f"\n🔍 Testing retrieval quality for {strategy_name}")
        print(f"Query: {test_query}")
        
        try:
            # Create index
            index = VectorStoreIndex(nodes, show_progress=False)
            
            # Create query engine
            query_engine = index.as_query_engine(similarity_top_k=3)
            
            # Query
            start_time = time.time()
            response = query_engine.query(test_query)
            query_time = time.time() - start_time
            
            print(f"⏱️  Query time: {query_time:.2f} seconds")
            print(f"📝 Response: {str(response)[:200]}...")
            
            # Show retrieved chunks
            if hasattr(response, 'source_nodes'):
                print(f"📊 Retrieved {len(response.source_nodes)} chunks")
                for i, node in enumerate(response.source_nodes[:2], 1):
                    print(f"  {i}. Score: {node.score:.3f} | Length: {len(node.text)} chars")
            
            return {
                'response': str(response),
                'query_time': query_time,
                'num_retrieved': len(response.source_nodes) if hasattr(response, 'source_nodes') else 0
            }
        
        except Exception as e:
            print(f"❌ Error testing retrieval: {e}")
            return {'error': str(e)}

def main():
    """Main function to demonstrate advanced chunking."""
    demo = AdvancedChunkingDemo()
    
    # Compare chunking strategies
    results = demo.compare_chunking_strategies()
    
    # Test retrieval quality for best strategies
    print(f"\n{'='*60}")
    print("🎯 TESTING RETRIEVAL QUALITY")
    print("="*60)
    
    test_strategies = [
        ("Sentence Splitter (512/50)", results.get("Sentence Splitter (512/50)", {}).get('nodes', [])),
        ("Sentence Splitter (256/25)", results.get("Sentence Splitter (256/25)", {}).get('nodes', [])),
    ]
    
    for strategy_name, nodes in test_strategies:
        if nodes:
            demo.test_retrieval_quality(nodes, strategy_name)

if __name__ == "__main__":
    main()
