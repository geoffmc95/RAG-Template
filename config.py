"""
Configuration system for Generic RAG Template
Allows users to customize the RAG system for their specific domain and use case.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

class DataType(Enum):
    """Supported data types for the RAG system."""
    PDF = "pdf"
    EXCEL = "excel"
    CSV = "csv"
    TXT = "txt"
    DOCX = "docx"
    JSON = "json"
    MARKDOWN = "md"

class DomainType(Enum):
    """Common domain types with pre-configured settings."""
    MEDICAL = "medical"
    FINANCIAL = "financial"
    LEGAL = "legal"
    RESEARCH = "research"
    TECHNICAL = "technical"
    GENERAL = "general"
    CUSTOM = "custom"

@dataclass
class ChunkingConfig:
    """Configuration for document chunking strategy."""
    chunk_size: int = 512
    chunk_overlap: int = 50
    separator: str = " "
    strategy: str = "sentence"  # sentence, token, semantic, simple

@dataclass
class RetrievalConfig:
    """Configuration for retrieval settings."""
    similarity_top_k: int = 5
    response_mode: str = "compact"  # compact, tree_summarize, simple_summarize
    temperature: float = 0.1

@dataclass
class DomainConfig:
    """Domain-specific configuration."""
    name: str
    description: str
    system_prompt: str
    suggested_questions: List[str]
    parsing_instructions: str
    file_types: List[str]

@dataclass
class RAGConfig:
    """Main configuration class for the RAG system."""
    # Basic settings
    project_name: str = "Generic RAG System"
    domain: DomainType = DomainType.GENERAL
    data_directory: str = "./data"
    storage_directory: str = "./storage"
    
    # Model settings
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    
    # Processing settings
    chunking: ChunkingConfig = None
    retrieval: RetrievalConfig = None
    domain_config: DomainConfig = None
    
    # File processing
    supported_file_types: List[str] = None
    max_file_size_mb: int = 100
    
    def __post_init__(self):
        """Initialize default configurations if not provided."""
        if self.chunking is None:
            self.chunking = ChunkingConfig()
        if self.retrieval is None:
            self.retrieval = RetrievalConfig()
        if self.domain_config is None:
            self.domain_config = self._get_default_domain_config()
        if self.supported_file_types is None:
            self.supported_file_types = [dt.value for dt in DataType]
    
    def _get_default_domain_config(self) -> DomainConfig:
        """Get default domain configuration based on domain type."""
        domain_configs = {
            DomainType.MEDICAL: DomainConfig(
                name="Medical Documents",
                description="Medical records, test results, and healthcare documents",
                system_prompt=(
                    "You are a helpful assistant specializing in medical documents and healthcare information. "
                    "Answer questions about medical records, test results, treatments, and procedures. "
                    "Be specific about medical findings, dates, and any important health information. "
                    "Always recommend consulting healthcare professionals for medical decisions."
                ),
                suggested_questions=[
                    "What are the key findings in my test results?",
                    "What treatments or medications are mentioned?",
                    "Are there any abnormal values or concerns?",
                    "What follow-up actions are recommended?"
                ],
                parsing_instructions="Focus on extracting medical findings, test results, dates, medications, and treatment recommendations clearly.",
                file_types=["pdf", "excel", "csv", "txt"]
            ),
            DomainType.FINANCIAL: DomainConfig(
                name="Financial Documents",
                description="Financial statements, reports, and investment documents",
                system_prompt=(
                    "You are a helpful assistant specializing in financial documents and analysis. "
                    "Answer questions about financial statements, investment performance, expenses, and financial metrics. "
                    "Be specific about amounts, dates, and financial trends. "
                    "Provide context for financial data and highlight important insights."
                ),
                suggested_questions=[
                    "What are the key financial metrics?",
                    "How has performance changed over time?",
                    "What are the major expenses or investments?",
                    "Are there any concerning financial trends?"
                ],
                parsing_instructions="Focus on extracting financial figures, dates, performance metrics, and key financial insights.",
                file_types=["pdf", "excel", "csv", "json"]
            ),
            DomainType.LEGAL: DomainConfig(
                name="Legal Documents",
                description="Contracts, legal agreements, and regulatory documents",
                system_prompt=(
                    "You are a helpful assistant specializing in legal documents and contracts. "
                    "Answer questions about legal terms, obligations, rights, and contract details. "
                    "Be specific about dates, parties involved, and legal requirements. "
                    "Note: This is for informational purposes only and not legal advice."
                ),
                suggested_questions=[
                    "What are the key terms and conditions?",
                    "What are my rights and obligations?",
                    "Are there important dates or deadlines?",
                    "What are the penalties or consequences mentioned?"
                ],
                parsing_instructions="Focus on extracting legal terms, obligations, dates, parties, and key contractual elements.",
                file_types=["pdf", "docx", "txt"]
            ),
            DomainType.RESEARCH: DomainConfig(
                name="Research Documents",
                description="Academic papers, research reports, and scientific documents",
                system_prompt=(
                    "You are a helpful assistant specializing in research documents and academic papers. "
                    "Answer questions about research findings, methodologies, data, and conclusions. "
                    "Be specific about study results, statistical significance, and research implications. "
                    "Cite relevant sections and provide context for research findings."
                ),
                suggested_questions=[
                    "What are the main research findings?",
                    "What methodology was used in this study?",
                    "What are the key conclusions and implications?",
                    "Are there any limitations or future research directions?"
                ],
                parsing_instructions="Focus on extracting research objectives, methodology, results, conclusions, and statistical data.",
                file_types=["pdf", "docx", "txt", "csv"]
            ),
            DomainType.TECHNICAL: DomainConfig(
                name="Technical Documentation",
                description="Technical manuals, specifications, and documentation",
                system_prompt=(
                    "You are a helpful assistant specializing in technical documentation and specifications. "
                    "Answer questions about technical procedures, specifications, troubleshooting, and implementation details. "
                    "Be specific about technical requirements, steps, and configurations. "
                    "Provide clear, actionable technical guidance."
                ),
                suggested_questions=[
                    "How do I implement this feature?",
                    "What are the technical requirements?",
                    "How do I troubleshoot this issue?",
                    "What are the configuration options?"
                ],
                parsing_instructions="Focus on extracting technical procedures, specifications, requirements, and implementation details.",
                file_types=["pdf", "docx", "txt", "md", "json"]
            ),
            DomainType.GENERAL: DomainConfig(
                name="General Documents",
                description="General purpose document analysis and Q&A",
                system_prompt=(
                    "You are a helpful assistant that can analyze and answer questions about various types of documents. "
                    "Provide accurate, contextual answers based on the document content. "
                    "Be specific about information found in the documents and cite relevant sections when possible. "
                    "If information is not available in the documents, clearly state that."
                ),
                suggested_questions=[
                    "What are the main topics covered in these documents?",
                    "Can you summarize the key points?",
                    "What specific information can you find about [topic]?",
                    "Are there any important dates or deadlines mentioned?"
                ],
                parsing_instructions="Extract all relevant information clearly, maintaining document structure and context.",
                file_types=["pdf", "docx", "txt", "csv", "excel", "md", "json"]
            )
        }
        
        return domain_configs.get(self.domain, domain_configs[DomainType.GENERAL])
    
    def save_to_file(self, filepath: str = "rag_config.json"):
        """Save configuration to JSON file."""
        config_dict = asdict(self)
        # Convert enum to string for JSON serialization
        config_dict['domain'] = self.domain.value
        
        with open(filepath, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: str = "rag_config.json") -> 'RAGConfig':
        """Load configuration from JSON file."""
        if not Path(filepath).exists():
            return cls()  # Return default config if file doesn't exist
        
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        
        # Convert string back to enum
        if 'domain' in config_dict:
            config_dict['domain'] = DomainType(config_dict['domain'])
        
        # Reconstruct nested dataclasses
        if 'chunking' in config_dict and config_dict['chunking']:
            config_dict['chunking'] = ChunkingConfig(**config_dict['chunking'])
        if 'retrieval' in config_dict and config_dict['retrieval']:
            config_dict['retrieval'] = RetrievalConfig(**config_dict['retrieval'])
        if 'domain_config' in config_dict and config_dict['domain_config']:
            config_dict['domain_config'] = DomainConfig(**config_dict['domain_config'])
        
        return cls(**config_dict)

def create_config_for_domain(domain: DomainType, project_name: str = None) -> RAGConfig:
    """Create a configuration for a specific domain."""
    config = RAGConfig(domain=domain)
    if project_name:
        config.project_name = project_name
    return config

def get_supported_file_extensions() -> Dict[str, List[str]]:
    """Get mapping of data types to file extensions."""
    return {
        DataType.PDF.value: ['.pdf'],
        DataType.EXCEL.value: ['.xlsx', '.xls'],
        DataType.CSV.value: ['.csv'],
        DataType.TXT.value: ['.txt'],
        DataType.DOCX.value: ['.docx', '.doc'],
        DataType.JSON.value: ['.json'],
        DataType.MARKDOWN.value: ['.md', '.markdown']
    }
