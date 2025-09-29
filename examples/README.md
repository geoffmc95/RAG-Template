# Generic RAG Template Examples

This directory contains example configurations and use cases for the Generic RAG Template, demonstrating how to adapt the system for different domains and data types.

## Directory Structure

```
examples/
├── medical_example.py      # Medical records and healthcare documents
├── financial_example.py    # Financial statements and analysis
├── research_example.py     # Academic papers and research documents
├── data/                   # Example data directories
│   ├── medical/           # Place medical documents here
│   ├── financial/         # Place financial documents here
│   └── research/          # Place research documents here
├── storage/               # Separate storage for each domain
│   ├── medical/          # Medical RAG index storage
│   ├── financial/        # Financial RAG index storage
│   └── research/         # Research RAG index storage
└── README.md             # This file
```

## Quick Start

### 1. Medical Documents Example

```bash
# Run the medical example
python examples/medical_example.py

# Or use the saved configuration
python -c "
from config import RAGConfig
from generic_rag import GenericRAGSystem
config = RAGConfig.load_from_file('examples/medical_config.json')
rag = GenericRAGSystem(config)
rag.initialize_system()
"
```

**Best for:** Test results, medical reports, lab data, health records

### 2. Financial Analysis Example

```bash
# Run the financial example
python examples/financial_example.py

# Or use the saved configuration
python -c "
from config import RAGConfig
from generic_rag import GenericRAGSystem
config = RAGConfig.load_from_file('examples/financial_config.json')
rag = GenericRAGSystem(config)
rag.initialize_system()
"
```

**Best for:** Financial statements, investment reports, expense data, budgets

### 3. Research Papers Example

```bash
# Run the research example
python examples/research_example.py

# Or use the saved configuration
python -c "
from config import RAGConfig
from generic_rag import GenericRAGSystem
config = RAGConfig.load_from_file('examples/research_config.json')
rag = GenericRAGSystem(config)
rag.initialize_system()
"
```

**Best for:** Academic papers, research reports, conference proceedings, survey data

## Configuration Comparison

| Domain        | Chunk Size | Overlap | Strategy | Top-K | Temperature | Best File Types       |
| ------------- | ---------- | ------- | -------- | ----- | ----------- | --------------------- |
| **Medical**   | 256        | 25      | sentence | 7     | 0.1         | PDF, Excel, CSV, TXT  |
| **Financial** | 768        | 75      | sentence | 5     | 0.05        | Excel, CSV, PDF, JSON |
| **Research**  | 512        | 50      | semantic | 6     | 0.2         | PDF, DOCX, TXT, CSV   |

## Usage Tips

### Adding Your Documents

1. **Create data directories:**

   ```bash
   mkdir -p examples/data/medical
   mkdir -p examples/data/financial
   mkdir -p examples/data/research
   ```

2. **Add your documents:**

   - Medical: Place test results, medical reports in `examples/data/medical/`
   - Financial: Place spreadsheets, financial reports in `examples/data/financial/`
   - Research: Place papers, manuscripts in `examples/data/research/`

3. **Run the appropriate example:**
   ```bash
   python examples/medical_example.py    # For medical documents
   python examples/financial_example.py  # For financial documents
   python examples/research_example.py   # For research documents
   ```

### Customizing Configurations

Each example creates a saved configuration file that you can modify:

```python
from config import RAGConfig

# Load and modify existing configuration
config = RAGConfig.load_from_file('examples/medical_config.json')
config.chunking.chunk_size = 128  # Smaller chunks
config.retrieval.similarity_top_k = 10  # More context
config.save_to_file('examples/my_custom_config.json')
```

### Creating New Domain Examples

To create a new domain example:

1. **Copy an existing example:**

   ```bash
   cp examples/medical_example.py examples/my_domain_example.py
   ```

2. **Modify the configuration:**

   ```python
   # Create custom domain configuration
   from config import DomainConfig, RAGConfig

   custom_domain = DomainConfig(
       name="My Custom Domain",
       description="Description of your domain",
       system_prompt="You are an expert in...",
       suggested_questions=["Question 1?", "Question 2?"],
       parsing_instructions="Focus on extracting...",
       file_types=["pdf", "txt", "csv"]
   )

   config = RAGConfig()
   config.domain_config = custom_domain
   ```

3. **Test and iterate:**
   ```bash
   python examples/my_domain_example.py
   ```

## Example Questions by Domain

### Medical Documents

- "What are the key findings in my test results?"
- "Are there any abnormal values or concerns?"
- "What treatments or medications are mentioned?"
- "What follow-up actions are recommended?"

### Financial Documents

- "What are the key financial metrics?"
- "How has performance changed over time?"
- "What are the major expenses or investments?"
- "Are there any concerning financial trends?"

### Research Documents

- "What are the main research findings?"
- "What methodology was used in this study?"
- "What are the key conclusions and implications?"
- "Are there any limitations or future research directions?"

## Troubleshooting

### Common Issues

1. **No documents found:**

   - Make sure documents are in the correct `examples/data/{domain}/` directory
   - Check that file formats are supported

2. **Initialization fails:**

   - Verify API keys are set in `.env` file
   - Check that all dependencies are installed

3. **Poor responses:**
   - Try adjusting chunk size and overlap
   - Increase similarity_top_k for more context
   - Adjust temperature for more/less creative responses

### Getting Help

- Check the main README.md for setup instructions
- Review the configuration options in `config.py`
- Look at the source code in `generic_rag.py` and `data_processor.py`

---

**Ready to try it out?** Pick an example that matches your use case and start exploring!
