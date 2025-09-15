"""
Interactive Chat Interface for Medical Insurance RAG Agent
"""

from dotenv import load_dotenv
import os
from pathlib import Path
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.memory import ChatMemoryBuffer
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

class InsuranceChatAgent:
    """Interactive chat agent for medical insurance queries."""
    
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = storage_dir
        self.index = None
        self.chat_engine = None
        self.setup_system()
    
    def setup_system(self):
        """Initialize the RAG system and chat engine."""
        print("🏥 Initializing Medical Insurance Chat Agent...")
        
        # Load or create index
        self.index = self._load_or_create_index()
        
        # Create chat engine with memory - using SimpleChatEngine for better retrieval
        memory = ChatMemoryBuffer.from_defaults(token_limit=3000)

        # Create a query engine first
        query_engine = self.index.as_query_engine(
            similarity_top_k=8,  # More chunks for better context
            response_mode="compact"
        )

        # Use the query engine as a chat engine
        from llama_index.core.chat_engine import SimpleChatEngine
        self.chat_engine = SimpleChatEngine.from_defaults(
            llm=Settings.llm,
            memory=memory,
            system_prompt=(
                "You are a helpful assistant specializing in medical insurance policies. "
                "You have access to a medical insurance policy document. "
                "Answer questions about insurance coverage, benefits, exclusions, and procedures. "
                "Be specific about coverage amounts, time limits, and any conditions. "
                "If you need to look up specific information, I will provide it to you."
            )
        )

        # Store query engine for fallback
        self.query_engine = query_engine
        
        print("✅ Chat agent ready!")
    
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
    
    def chat(self, message):
        """Process a chat message and return response."""
        if not self.chat_engine or not self.query_engine:
            return "❌ Chat engine not initialized. Please restart the application."

        try:
            # First try to get information using the query engine for better retrieval
            query_response = self.query_engine.query(message)

            # Then use the chat engine with the retrieved information
            enhanced_message = f"Based on the insurance policy information: {query_response}\n\nUser question: {message}"
            chat_response = self.chat_engine.chat(enhanced_message)

            return str(chat_response)
        except Exception as e:
            return f"❌ Error processing your question: {e}"
    
    def reset_conversation(self):
        """Reset the conversation memory."""
        if self.chat_engine:
            self.chat_engine.reset()
            print("🔄 Conversation history cleared!")
    
    def get_suggested_questions(self):
        """Return a list of suggested questions."""
        return [
            "What is my annual dental coverage limit?",
            "What vision benefits do I have?",
            "Are there any deductibles I need to pay?",
            "What are the main exclusions in my policy?",
            "How do I submit a claim?",
            "What is covered for prescription drugs?",
            "Do I have travel insurance coverage?",
            "What happens to my benefits when I retire?"
        ]

def main():
    """Main interactive chat loop."""
    agent = InsuranceChatAgent()
    
    print("\n" + "="*60)
    print("🏥 MEDICAL INSURANCE CHAT AGENT")
    print("="*60)
    print("Ask me anything about your insurance policy!")
    print("Type 'quit' to exit, 'reset' to clear conversation history")
    print("Type 'help' for suggested questions")
    print("-"*60)
    
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye! Stay healthy!")
                break
            
            if user_input.lower() == 'reset':
                agent.reset_conversation()
                continue
            
            if user_input.lower() == 'help':
                print("\n💡 Suggested questions:")
                for i, question in enumerate(agent.get_suggested_questions(), 1):
                    print(f"   {i}. {question}")
                continue
            
            if not user_input:
                print("Please enter a question or type 'help' for suggestions.")
                continue
            
            print("🤔 Thinking...")
            start_time = time.time()
            response = agent.chat(user_input)
            response_time = time.time() - start_time
            
            print(f"\n🏥 Agent: {response}")
            print(f"\n⏱️  Response time: {response_time:.2f} seconds")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Stay healthy!")
            break
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
