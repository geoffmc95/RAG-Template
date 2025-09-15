"""
Evaluation Framework for Insurance RAG System
"""

from dotenv import load_dotenv
import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.evaluation import (
    FaithfulnessEvaluator,
    RelevancyEvaluator,
    CorrectnessEvaluator,
    SemanticSimilarityEvaluator
)
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

load_dotenv()

# Configure API keys
openai_api_key = os.getenv("OPENAI_API_KEY")
llamacloud_api_key = os.getenv("LLAMACLOUD_API_KEY")

# Configure global settings
Settings.llm = OpenAI(model="gpt-4o-mini", temperature=0.1)
Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

class InsuranceRAGEvaluator:
    """Comprehensive evaluation framework for insurance RAG system."""
    
    def __init__(self, storage_dir="./storage"):
        self.storage_dir = storage_dir
        self.index = None
        self.query_engine = None
        self.evaluators = {}
        self.test_cases = []
        self.setup_system()
        self.setup_evaluators()
        self.create_test_cases()
    
    def setup_system(self):
        """Initialize the RAG system."""
        print("🔧 Setting up RAG system for evaluation...")
        
        # Load existing index
        if Path(self.storage_dir).exists():
            try:
                storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
                self.index = load_index_from_storage(storage_context)
                print("✅ Index loaded successfully!")
            except Exception as e:
                print(f"❌ Error loading index: {e}")
                return
        else:
            print("❌ No existing index found. Please run the main system first.")
            return
        
        # Create query engine
        self.query_engine = self.index.as_query_engine(
            similarity_top_k=5,
            response_mode="compact"
        )
    
    def setup_evaluators(self):
        """Initialize evaluation metrics."""
        print("📊 Setting up evaluators...")
        
        try:
            self.evaluators = {
                'faithfulness': FaithfulnessEvaluator(llm=Settings.llm),
                'relevancy': RelevancyEvaluator(llm=Settings.llm),
                'correctness': CorrectnessEvaluator(llm=Settings.llm),
                'semantic_similarity': SemanticSimilarityEvaluator(embed_model=Settings.embed_model)
            }
            print("✅ Evaluators initialized!")
        except Exception as e:
            print(f"❌ Error setting up evaluators: {e}")
    
    def create_test_cases(self):
        """Create comprehensive test cases for insurance queries."""
        self.test_cases = [
            {
                "query": "What is my annual dental coverage limit?",
                "expected_answer": "$1,500 per calendar year",
                "category": "dental_coverage",
                "difficulty": "easy"
            },
            {
                "query": "What vision benefits do I have for eye examinations?",
                "expected_answer": "One eye examination every 24 months for adults, every 12 months for dependent children under 21",
                "category": "vision_coverage",
                "difficulty": "medium"
            },
            {
                "query": "Are there any exclusions for dental coverage?",
                "expected_answer": "Cosmetic procedures, services not supervised by a dentist",
                "category": "exclusions",
                "difficulty": "medium"
            },
            {
                "query": "What is the co-insurance rate for dental services?",
                "expected_answer": "80% coverage (20% patient responsibility)",
                "category": "dental_coverage",
                "difficulty": "easy"
            },
            {
                "query": "How much can I claim for contact lenses due to disease?",
                "expected_answer": "Maximum $200 every two consecutive calendar years with written authorization from physician",
                "category": "vision_coverage",
                "difficulty": "hard"
            },
            {
                "query": "What happens to my benefits when I retire?",
                "expected_answer": "Benefits cease at retirement or termination of employment",
                "category": "general_policy",
                "difficulty": "medium"
            },
            {
                "query": "How long do I have to submit claims?",
                "expected_answer": "Claims must be submitted within 24 months of receiving services or supplies",
                "category": "claims_process",
                "difficulty": "medium"
            },
            {
                "query": "What is the maximum for visual training benefits?",
                "expected_answer": "$150 lifetime maximum provided by registered optometrist or ophthalmologist",
                "category": "vision_coverage",
                "difficulty": "hard"
            }
        ]
        
        print(f"📝 Created {len(self.test_cases)} test cases")
    
    def evaluate_single_query(self, test_case: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single query."""
        query = test_case["query"]
        expected = test_case["expected_answer"]
        
        print(f"🔍 Evaluating: {query}")
        
        # Get response
        start_time = time.time()
        response = self.query_engine.query(query)
        response_time = time.time() - start_time
        
        response_text = str(response)
        
        # Evaluate with different metrics
        results = {
            "query": query,
            "expected_answer": expected,
            "actual_answer": response_text,
            "response_time": response_time,
            "category": test_case["category"],
            "difficulty": test_case["difficulty"]
        }
        
        # Run evaluations
        try:
            # Faithfulness - does the response align with retrieved context?
            if hasattr(response, 'source_nodes') and response.source_nodes:
                contexts = [node.text for node in response.source_nodes]
                faithfulness_result = self.evaluators['faithfulness'].evaluate_response(
                    query=query,
                    response=response
                )
                results["faithfulness_score"] = faithfulness_result.score
                results["faithfulness_feedback"] = faithfulness_result.feedback
            
            # Relevancy - is the response relevant to the query?
            relevancy_result = self.evaluators['relevancy'].evaluate_response(
                query=query,
                response=response
            )
            results["relevancy_score"] = relevancy_result.score
            results["relevancy_feedback"] = relevancy_result.feedback
            
            # Correctness - how correct is the response compared to expected?
            correctness_result = self.evaluators['correctness'].evaluate(
                query=query,
                response=response_text,
                reference=expected
            )
            results["correctness_score"] = correctness_result.score
            results["correctness_feedback"] = correctness_result.feedback
            
            # Semantic similarity
            similarity_result = self.evaluators['semantic_similarity'].evaluate(
                query=query,
                response=response_text,
                reference=expected
            )
            results["semantic_similarity_score"] = similarity_result.score
            
        except Exception as e:
            print(f"⚠️  Error in evaluation: {e}")
            results["evaluation_error"] = str(e)
        
        return results
    
    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run evaluation on all test cases."""
        print("\n" + "="*60)
        print("🧪 RUNNING FULL RAG EVALUATION")
        print("="*60)
        
        all_results = []
        category_scores = {}
        
        for i, test_case in enumerate(self.test_cases, 1):
            print(f"\n[{i}/{len(self.test_cases)}] ", end="")
            result = self.evaluate_single_query(test_case)
            all_results.append(result)
            
            # Track category performance
            category = result["category"]
            if category not in category_scores:
                category_scores[category] = []
            
            # Add scores if available
            if "correctness_score" in result:
                category_scores[category].append(result["correctness_score"])
        
        # Calculate summary statistics
        summary = self.calculate_summary_stats(all_results, category_scores)
        
        # Save results
        self.save_evaluation_results(all_results, summary)
        
        return {
            "individual_results": all_results,
            "summary": summary
        }
    
    def calculate_summary_stats(self, results: List[Dict], category_scores: Dict) -> Dict:
        """Calculate summary statistics."""
        print("\n📊 Calculating summary statistics...")
        
        # Overall metrics
        total_queries = len(results)
        avg_response_time = sum(r.get("response_time", 0) for r in results) / total_queries
        
        # Score averages
        metrics = ["faithfulness_score", "relevancy_score", "correctness_score", "semantic_similarity_score"]
        avg_scores = {}
        
        for metric in metrics:
            scores = [r.get(metric) for r in results if r.get(metric) is not None]
            if scores:
                avg_scores[metric] = sum(scores) / len(scores)
        
        # Category performance
        category_performance = {}
        for category, scores in category_scores.items():
            if scores:
                category_performance[category] = {
                    "avg_score": sum(scores) / len(scores),
                    "num_queries": len(scores)
                }
        
        summary = {
            "total_queries": total_queries,
            "avg_response_time": avg_response_time,
            "avg_scores": avg_scores,
            "category_performance": category_performance
        }
        
        return summary
    
    def save_evaluation_results(self, results: List[Dict], summary: Dict):
        """Save evaluation results to file."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"evaluation_results_{timestamp}.json"
        
        output = {
            "timestamp": timestamp,
            "summary": summary,
            "individual_results": results
        }
        
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        print(f"💾 Results saved to {filename}")
    
    def print_summary_report(self, summary: Dict):
        """Print a formatted summary report."""
        print("\n" + "="*60)
        print("📋 EVALUATION SUMMARY REPORT")
        print("="*60)
        
        print(f"Total Queries Evaluated: {summary['total_queries']}")
        print(f"Average Response Time: {summary['avg_response_time']:.2f} seconds")
        
        print("\n📊 Average Scores:")
        for metric, score in summary['avg_scores'].items():
            print(f"  {metric.replace('_', ' ').title()}: {score:.3f}")
        
        print("\n🏷️  Category Performance:")
        for category, perf in summary['category_performance'].items():
            print(f"  {category.replace('_', ' ').title()}: {perf['avg_score']:.3f} ({perf['num_queries']} queries)")

def main():
    """Main function to run evaluation."""
    evaluator = InsuranceRAGEvaluator()
    
    if not evaluator.query_engine:
        print("❌ Cannot run evaluation without a working query engine.")
        return
    
    # Run full evaluation
    results = evaluator.run_full_evaluation()
    
    # Print summary
    evaluator.print_summary_report(results["summary"])

if __name__ == "__main__":
    main()
