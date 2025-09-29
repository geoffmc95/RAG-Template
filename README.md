# Generic RAG Template

A flexible, configurable Retrieval-Augmented Generation (RAG) system built with LlamaIndex that adapts to any domain and data type. Upload your documents (PDFs, Excel files, CSVs, etc.) and get intelligent, contextual answers tailored to your specific use case.

## Features

### Domain Adaptability

- **Pre-configured domains**: Medical, Financial, Legal, Research, Technical, and General
- **Custom domain support**: Define your own domain with custom prompts and settings
- **Intelligent suggestions**: Domain-specific suggested questions and parsing instructions

### Multi-Format Document Support

- **PDF documents**: Advanced processing with LlamaParse for complex layouts and tables
- **Excel spreadsheets**: Automatic data extraction with summary statistics (.xlsx, .xls)
- **CSV files**: Structured data processing with insights
- **Text files**: Plain text and Markdown support (.txt, .md)
- **Word documents**: DOCX processing with table extraction
- **JSON files**: Structured data parsing and analysis

### Flexible Configuration

- **Configurable chunking**: Multiple strategies (sentence, token, semantic, simple)
- **Adjustable parameters**: Chunk size, overlap, similarity thresholds, temperature
- **Model selection**: Choose your preferred LLM and embedding models
- **Storage management**: Persistent vector storage with rebuild options

### Interactive Interfaces

- **Web interface**: Streamlit-based GUI with file upload and configuration
- **Chat mode**: Conversational interface with memory and context
- **Query with sources**: Detailed responses with source citations and scores
- **Command-line interface**: Direct Python API access for automation

### Performance Optimizations

- **Persistent storage**: Vector index caching for fast startup (11+ seconds → 0.6 seconds)
- **Efficient processing**: Optimized document processing pipeline
- **Batch operations**: Handle multiple files simultaneously
- **Smart caching**: Streamlit resource caching for better performance

### Advanced Query Capabilities

- **Source attribution**: Citations with similarity scores and metadata
- **Multiple query modes**: Chat with memory vs. single query with sources
- **Configurable retrieval**: Adjustable similarity thresholds and result counts
- **Context-aware responses**: Domain-specific system prompts and formatting

## Project Structure

```
Generic RAG Template/
├── data/                      # Upload your documents here
│   ├── *.pdf                  # PDF documents
│   ├── *.xlsx, *.xls          # Excel spreadsheets
│   ├── *.csv                  # CSV files
│   ├── *.txt, *.md            # Text and Markdown files
│   ├── *.docx                 # Word documents
│   └── *.json                 # JSON files
├── storage/                   # Persistent vector index storage
├── config.py                  # Configuration system for domains and settings
├── data_processor.py          # Multi-format document processing
├── generic_rag.py             # Main RAG system (replaces query.py)
├── streamlit_app.py           # Web interface with file upload
├── chat_interface.py          # Command-line chat interface
├── enhanced_query_engine.py   # Advanced query engines (legacy)
├── advanced_chunking.py       # Chunking strategy comparison (legacy)
├── evaluation_framework.py    # Evaluation system (legacy)
├── requirements_web.txt       # Python dependencies
├── .env                       # API keys (OpenAI, LlamaCloud)
└── README.md                  # This file
```

## Setup and Installation

### First-Time Setup

1. **Clone the Repository**

   ```bash
   git clone <repo-url>
   cd "RAG Template"
   ```

2. **Create Virtual Environment** (Recommended)

   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install -r requirements_web.txt
   ```

   Or install manually:

   ```bash
   pip install llama-index==0.14.1 llama-parse==0.6.54 python-dotenv==1.0.0
   pip install llama-index-llms-openai==0.5.6 llama-index-embeddings-openai==0.5.1
   pip install streamlit==1.28.1 pandas==2.1.3 openpyxl==3.1.2 python-docx==0.8.11
   ```

4. **Configure API Keys**
   Create a `.env` file in the project root:

   ```bash
   # Create the .env file (it's gitignored for security)
   touch .env  # macOS/Linux
   # Or create manually on Windows
   ```

   Add your API keys to `.env`:

   ```
   OPENAI_API_KEY=your_openai_api_key_here
   LLAMACLOUD_API_KEY=your_llamacloud_api_key_here
   ```

   **Get API Keys:**

   - **OpenAI API Key**: [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys)
   - **LlamaCloud API Key**: [https://cloud.llamaindex.ai](https://cloud.llamaindex.ai) (for advanced PDF processing)

5. **Create Data Directory**

   ```bash
   mkdir data
   ```

   **Note**: The `data/` directory is gitignored to prevent accidentally committing large files or sensitive documents.

6. **Add Your Documents**
   Place your documents in the `data/` directory:


   **Supported formats**: PDF, Excel (.xlsx, .xls), CSV, TXT, DOCX, JSON, Markdown

### Quick Start

After completing the setup above:

1. **Run the Web Interface**

   ```bash
   streamlit run streamlit_app.py
   ```

2. **Or use Command Line**
   ```bash
   python generic_rag.py
   ```

### Important Notes for New Users

- **`.env` file**: Contains your API keys and is gitignored for security. You must create this file manually.
- **`data/` directory**: Gitignored to prevent committing large documents or sensitive files. Create and populate manually.
- **`storage/` directory**: Contains vector indexes and is gitignored. Will be created automatically when you first run the system.
- **`venv/` directory**: Virtual environment is gitignored. Create using the instructions above.
- **`__pycache__/`**: Python cache files are gitignored and created automatically.

### Troubleshooting

**Missing API Keys Error:**

- Ensure your `.env` file exists in the project root
- Check that API keys are correctly formatted (no quotes needed)
- Verify API keys are valid and have sufficient credits

**No Documents Found:**

- Ensure the `data/` directory exists and contains supported file types
- Check file permissions and that files aren't corrupted

**Import Errors:**

- Activate your virtual environment: `venv\Scripts\activate` (Windows) or `source venv/bin/activate` (macOS/Linux)
- Reinstall dependencies: `pip install -r requirements_web.txt`

## Usage

### Web Interface (Recommended)

```bash
streamlit run streamlit_app.py
```

**Features:**

- **File Upload**: Drag and drop your documents directly in the browser
- **Configuration**: Choose domain, adjust settings, and customize behavior
- **Chat Mode**: Conversational interface with memory
- **Query with Sources**: Get detailed responses with source citations
- **System Info**: Monitor configuration and performance

### Command Line Interface

```bash
python generic_rag.py
```

**Features:**

- Quick setup with default configuration
- Interactive query interface
- Domain-specific suggested questions
- Performance metrics and timing

### Custom Configuration

Create a custom configuration for your specific use case:

```python
from config import RAGConfig, DomainType, create_config_for_domain
from generic_rag import GenericRAGSystem

# Create domain-specific configuration
config = create_config_for_domain(DomainType.MEDICAL, "My Medical RAG")
config.data_directory = "./my_data"
config.chunking.chunk_size = 256  # Smaller chunks for detailed analysis

# Initialize RAG system
rag = GenericRAGSystem(config)
rag.initialize_system()

# Query the system
response = rag.query("What are the key findings in my test results?")
print(response)
```

### Domain Examples

#### Medical Documents

```python
# Configure for medical domain
config = create_config_for_domain(DomainType.MEDICAL, "Medical Records Analysis")
# Upload: test results, medical reports, lab data
# Ask: "What are my latest test results?" "Any abnormal values?"
```

#### Financial Analysis

```python
# Configure for financial domain
config = create_config_for_domain(DomainType.FINANCIAL, "Financial Analysis")
# Upload: financial statements, investment reports, expense data
# Ask: "What are the key financial metrics?" "How has performance changed?"
```

#### Research Papers

```python
# Configure for research domain
config = create_config_for_domain(DomainType.RESEARCH, "Research Analysis")
# Upload: academic papers, research reports
# Ask: "What are the main findings?" "What methodology was used?"
```

## Configuration Options

### Domain Types

| Domain        | Description                        | Best For                                  |
| ------------- | ---------------------------------- | ----------------------------------------- |
| **Medical**   | Healthcare documents, test results | Medical records, lab reports, health data |
| **Financial** | Financial statements, reports      | Investment analysis, expense tracking     |
| **Legal**     | Contracts, legal documents         | Contract analysis, compliance documents   |
| **Research**  | Academic papers, studies           | Research analysis, literature review      |
| **Technical** | Technical documentation            | API docs, manuals, specifications         |
| **General**   | Any document type                  | Mixed content, general purpose            |

### Chunking Strategies

| Strategy     | Best For          | Chunk Size | Overlap   |
| ------------ | ----------------- | ---------- | --------- |
| **Sentence** | Most documents    | 512 chars  | 50 chars  |
| **Token**    | Precise control   | 400 tokens | 40 tokens |
| **Semantic** | Coherent sections | Variable   | Automatic |
| **Simple**   | Large documents   | 1024 chars | 0 chars   |

### File Type Support

| Format    | Extensions      | Processing Method                         |
| --------- | --------------- | ----------------------------------------- |
| **PDF**   | `.pdf`          | LlamaParse (advanced) or basic extraction |
| **Excel** | `.xlsx`, `.xls` | Pandas with summary statistics            |
| **CSV**   | `.csv`          | Structured data analysis                  |
| **Text**  | `.txt`, `.md`   | Direct text processing                    |
| **Word**  | `.docx`         | Text and table extraction                 |
| **JSON**  | `.json`         | Structured data parsing                   |

### Advanced Settings

The system provides several advanced configuration options that can be adjusted through the web interface or programmatically:

#### Chunk Size

- **Range**: 256 - 1024 characters
- **Default**: 512 characters
- **Purpose**: Controls how large each text chunk is when documents are split for processing
- **Impact**:
  - **Smaller chunks (256-400)**: Better for precise, detailed answers but may lose context
  - **Larger chunks (600-1024)**: Better for comprehensive answers but may include irrelevant information
- **Best for**:
  - Medical/Legal documents: 256-400 (precise details matter)
  - Research papers: 512-700 (balance of detail and context)
  - General documents: 400-600 (good balance)

#### Chunk Overlap

- **Range**: 0 - 100 characters
- **Default**: 50 characters
- **Purpose**: How much text overlaps between adjacent chunks to maintain context
- **Impact**:
  - **No overlap (0)**: Faster processing but may lose context at chunk boundaries
  - **High overlap (75-100)**: Better context preservation but slower processing and more storage
- **Best for**:
  - Technical documents: 50-75 (maintain technical context)
  - Narrative text: 25-50 (less critical boundaries)
  - Structured data: 0-25 (clear boundaries)

#### Retrieved Chunks

- **Range**: 3 - 10 chunks
- **Default**: 5 chunks
- **Purpose**: How many relevant text chunks to retrieve for each query
- **Impact**:
  - **Fewer chunks (3-4)**: Faster, more focused answers but may miss relevant information
  - **More chunks (7-10)**: More comprehensive answers but may include less relevant information
- **Best for**:
  - Simple questions: 3-4 chunks
  - Complex analysis: 6-8 chunks
  - Research queries: 7-10 chunks

#### Temperature

- **Range**: 0.0 - 1.0
- **Default**: 0.1
- **Purpose**: Controls the creativity/randomness of the AI's responses
- **Impact**:
  - **Low temperature (0.0-0.3)**: More deterministic, factual, consistent responses
  - **Medium temperature (0.4-0.7)**: Balanced creativity and accuracy
  - **High temperature (0.8-1.0)**: More creative but potentially less accurate responses
- **Best for**:
  - Medical/Legal/Financial: 0.0-0.2 (accuracy critical)
  - Research analysis: 0.1-0.4 (some interpretation needed)
  - Creative writing: 0.5-0.8 (creativity valued)

#### Configuration Examples

```python
# High-precision medical configuration
config.chunking.chunk_size = 256
config.chunking.chunk_overlap = 75
config.retrieval.similarity_top_k = 3
config.retrieval.temperature = 0.0

# Comprehensive research configuration
config.chunking.chunk_size = 600
config.chunking.chunk_overlap = 50
config.retrieval.similarity_top_k = 8
config.retrieval.temperature = 0.2

# Balanced general-purpose configuration
config.chunking.chunk_size = 512
config.chunking.chunk_overlap = 50
config.retrieval.similarity_top_k = 5
config.retrieval.temperature = 0.1
```

## Technical Architecture

### Core Components

- **Configuration System** (`config.py`): Domain-specific settings and customization
- **Data Processor** (`data_processor.py`): Multi-format document processing pipeline
- **Generic RAG System** (`generic_rag.py`): Main RAG engine with domain adaptation
- **LlamaParse**: Advanced PDF processing with layout understanding
- **OpenAI GPT-4o-mini**: Language model for intelligent responses
- **OpenAI text-embedding-3-small**: Vector embeddings for semantic search
- **LlamaIndex**: Vector storage, retrieval, and query processing
- **Streamlit**: Web interface with file upload and configuration

### Key Features

- **Domain Adaptation**: Automatic prompt and behavior adjustment
- **Multi-Format Processing**: Unified pipeline for different file types
- **Persistent Storage**: Fast startup with vector index caching
- **Flexible Configuration**: Customizable chunking, retrieval, and model settings
- **Interactive Interfaces**: Both web and command-line access
- **Source Attribution**: Detailed citations with similarity scores

## Customization and Extension

### Adding New Domains

```python
# Add a custom domain configuration
custom_domain = DomainConfig(
    name="Legal Contracts",
    description="Contract analysis and legal document review",
    system_prompt="You are a legal document analyst...",
    suggested_questions=[
        "What are the key terms and conditions?",
        "What are the obligations of each party?",
        "Are there any important deadlines?"
    ],
    parsing_instructions="Focus on legal terms, dates, and obligations",
    file_types=["pdf", "docx", "txt"]
)
```

### Adding New File Types

Extend the `DataProcessor` class to support additional formats:

```python
def process_new_format(self, file_paths: List[Path]) -> List[Document]:
    """Process your custom file format."""
    documents = []
    for file_path in file_paths:
        # Your custom processing logic here
        content = process_custom_file(file_path)
        doc = Document(text=content, metadata={"file_type": "custom"})
        documents.append(doc)
    return documents
```

### Integration Examples

#### API Integration

```python
from generic_rag import GenericRAGSystem
from config import create_config_for_domain, DomainType

# Create RAG system for API use
config = create_config_for_domain(DomainType.TECHNICAL)
rag = GenericRAGSystem(config)
rag.initialize_system()

# Use in your application
def api_query(question: str) -> str:
    return rag.query(question)
```

#### Batch Processing

```python
# Process multiple queries
questions = ["What is the main topic?", "What are the key findings?"]
responses = [rag.query(q) for q in questions]
```

## Contributing

This template is designed to be extended and customized. Areas for contribution:

1. **New Domain Configurations**: Add pre-configured domains for specific industries
2. **File Format Support**: Extend processing for additional file types
3. **Evaluation Metrics**: Add domain-specific evaluation frameworks
4. **UI Enhancements**: Improve the Streamlit interface with new features
5. **Performance Optimizations**: Enhance processing speed and memory usage



