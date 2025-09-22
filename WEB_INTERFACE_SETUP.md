# Web Interface Setup Guide

This guide will help you set up both Streamlit and Gradio web interfaces for your Medical Insurance RAG Agent.

## Prerequisites

1. **Existing RAG System**: Make sure you have already run `python query.py` to create the vector index
2. **API Keys**: Ensure your `.env` file contains valid OpenAI and LlamaCloud API keys
3. **Python Environment**: Your existing `rag_env` virtual environment

## Quick Setup

### 1. Install Web Interface Dependencies

```bash
# Activate your virtual environment
# On Windows:
rag_env\Scripts\activate
# On macOS/Linux:
source rag_env/bin/activate

# Install web interface packages
pip install streamlit
```

Or install from requirements file:

```bash
pip install -r requirements_web.txt
```

### 2. Choose Your Interface

You now have two web interface options:

## Option 1: Streamlit Interface (Recommended)

### Features:

- Professional, clean design
- Sidebar with suggested questions
- Two modes: Chat and Query with Sources
- Real-time system status
- Conversation history
- Source citations with metadata

### Launch Streamlit:

```bash
streamlit run streamlit_app.py
```

**Access:** Open your browser to `http://localhost:8501`

### Streamlit Interface Overview:

- **Left Sidebar**: Settings, suggested questions, help
- **Main Area**: Chat interface or query with sources
- **Right Panel**: System information and metrics

## 🎯 Option 2: Gradio Interface

### Features:

- ✅ Simple, intuitive design
- ✅ Tabbed interface
- ✅ Built-in suggested question buttons
- ✅ Easy sharing capabilities
- ✅ Mobile-friendly

### Launch Gradio:

```bash
python gradio_app.py
```

**Access:** Open your browser to `http://localhost:7860`

### Gradio Interface Overview:

- **Chat Mode Tab**: Conversational interface with memory
- **Query with Sources Tab**: Single queries with citations
- **System Info Tab**: System status and usage tips

## 🔧 Configuration Options

### Streamlit Configuration

Create `config.toml` in `.streamlit/` folder for custom settings:

```toml
[theme]
primaryColor = "#667eea"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[server]
port = 8501
headless = true
```

### Gradio Configuration

Modify the `interface.launch()` parameters in `gradio_app.py`:

```python
interface.launch(
    server_name="0.0.0.0",  # Allow external access
    server_port=7860,       # Custom port
    share=True,             # Create public sharing link
    debug=True,             # Enable debug mode
    auth=("username", "password")  # Add authentication
)
```

## 🌟 Interface Comparison

| Feature                 | Streamlit                    | Gradio              |
| ----------------------- | ---------------------------- | ------------------- |
| **Design**              | Professional, customizable   | Simple, clean       |
| **Chat Memory**         | ✅ Full conversation history | ✅ Session-based    |
| **Source Citations**    | ✅ Detailed with metadata    | ✅ Formatted text   |
| **Suggested Questions** | ✅ Sidebar buttons           | ✅ Inline buttons   |
| **Mobile Support**      | ✅ Responsive                | ✅ Mobile-optimized |
| **Customization**       | ✅ High (CSS, themes)        | ✅ Medium           |
| **Sharing**             | ✅ Via Streamlit Cloud       | ✅ Built-in sharing |
| **Authentication**      | ✅ Via extensions            | ✅ Built-in         |

## 🎯 Usage Tips

### For Best Results:

1. **Be Specific**: Ask detailed questions about your insurance policy
2. **Use Suggested Questions**: Start with the provided examples
3. **Try Both Modes**:
   - Use **Chat Mode** for follow-up questions
   - Use **Query with Sources** to see document references

### Example Queries:

- "What is my annual dental coverage limit?"
- "What vision benefits do I have for eye examinations?"
- "Are there any exclusions for dental coverage?"
- "How do I submit a claim for reimbursement?"

## 🔒 Security Considerations

### For Production Deployment:

1. **Environment Variables**: Never commit API keys to version control
2. **Authentication**: Add user authentication for sensitive data
3. **HTTPS**: Use SSL certificates for secure connections
4. **Access Control**: Limit access to authorized users only

### Streamlit Security:

```python
# Add to streamlit_app.py
import streamlit_authenticator as stauth

# Configure authentication
authenticator = stauth.Authenticate(
    credentials,
    'insurance_rag',
    'auth_key',
    cookie_expiry_days=30
)
```

### Gradio Security:

```python
# Add to gradio_app.py
interface.launch(
    auth=("admin", "secure_password"),
    ssl_verify=False
)
```

## 🚀 Deployment Options

### 1. Local Development

- Run on localhost for testing
- Perfect for personal use

### 2. Streamlit Cloud

```bash
# Push to GitHub and deploy via streamlit.io
git add .
git commit -m "Add web interface"
git push origin main
```

### 3. Gradio Spaces (Hugging Face)

- Upload to Hugging Face Spaces
- Automatic deployment and sharing

### 4. Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY . .
RUN pip install -r requirements_web.txt

EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

## 🐛 Troubleshooting

### Common Issues:

1. **"RAG system not loaded"**

   - Run `python query.py` first to create the index
   - Check that `./storage` directory exists

2. **API Key Errors**

   - Verify `.env` file contains valid keys
   - Check API key permissions and quotas

3. **Port Already in Use**

   - Change port in launch configuration
   - Kill existing processes: `lsof -ti:8501 | xargs kill -9`

4. **Memory Issues**
   - Reduce `similarity_top_k` parameter
   - Clear browser cache and restart

### Debug Mode:

```bash
# Streamlit with debug
streamlit run streamlit_app.py --logger.level=debug

# Gradio with debug
# Set debug=True in gradio_app.py
```

## 📞 Support

If you encounter issues:

1. Check the console output for error messages
2. Verify all dependencies are installed
3. Ensure the RAG system is properly initialized
4. Check API key validity and quotas

## 🎉 Next Steps

Once your web interface is running:

1. **Test with various queries** to ensure accuracy
2. **Customize the interface** to match your branding
3. **Add authentication** for production use
4. **Deploy to cloud** for broader access
5. **Monitor usage** and performance metrics

Your Medical Insurance RAG Agent is now ready for web-based interaction! 🚀
