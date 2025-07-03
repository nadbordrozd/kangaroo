# RAG Contact Center AI Assistant

A Retrieval-Augmented Generation (RAG) system that provides AI-powered suggestions for customer service agents. The system analyzes customer messages and generates contextual response suggestions based on a knowledge base of company documents.

## 🎯 Features

- **Three-Panel Interface**: Chat interface, AI assistant (summary + suggestions), and knowledge articles
- **Customer Service Context**: Specialized for exotic meats business with domain-specific knowledge
- **Conversation Tracking**: Maintains context across customer-agent conversations
- **Multiple Interfaces**: Both web (Flask) and notebook (Jupyter) versions available
- **Real-time Processing**: Instant AI suggestions when customers send messages

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment Setup

Create a `.env` file:
```
OPENAI_API_KEY=your_openai_api_key_here
```

### Running the Web Interface

```bash
python app.py
```
Visit `http://localhost:5000` to access the web interface.

### Running the Jupyter Notebook Interface

1. **Open in Cursor**: 
   - Open `notebook_cells.txt`
   - Copy each CELL section into separate Jupyter notebook cells
   - Run all cells

2. **Traditional Jupyter**:
   ```bash
   jupyter notebook
   ```
   - Create a new notebook
   - Copy code from `notebook_cells.txt` into cells

## 📁 Project Structure

```
agent_copilot_poc/
├── app.py                      # Flask web application
├── rag_system.py              # Core RAG system logic
├── notebook_cells.txt          # Jupyter notebook code
├── requirements.txt            # Python dependencies
├── knowledge_base/             # Knowledge base documents
│   ├── complaints_handling_procedure.txt
│   ├── customer_service_procedures.txt
│   ├── food_safety_regulations.txt
│   └── [other .txt files]
├── storage/                    # Vector embeddings cache
├── templates/
│   └── index.html             # Web interface template
└── static/
    ├── style.css              # Web interface styling
    └── script.js              # Web interface JavaScript
```

## 🎛️ How It Works

1. **Customer Message**: When a customer sends a message, the system:
   - Retrieves relevant knowledge base articles
   - Generates conversation summary
   - Creates AI-powered response suggestions

2. **Agent Response**: Agent responses are added to conversation history but don't trigger AI processing

3. **Knowledge Base**: The system indexes `.txt` files from the `knowledge_base/` directory using vector embeddings

## 🔧 Configuration

### RAG System Settings

- **Similarity Top-K**: Number of relevant documents to retrieve (default: 5)
- **Reranker**: AI-powered relevance filtering (disabled in Jupyter for compatibility)
- **Conversation Context**: Number of previous messages to include (default: 5)

### Web vs Notebook

| Feature | Web Interface | Jupyter Notebook |
|---------|--------------|------------------|
| Reranker | ✅ Enabled | ❌ Disabled (asyncio compatibility) |
| UI | Full web experience | Interactive widgets |
| Development | Production ready | Development/testing |

## 📝 Usage Examples

### Customer Messages (trigger AI processing):
- "Hello, I need help with my order"
- "I want to order kangaroo meat for a dinner party"
- "How long does shipping take?"

### Agent Responses (no AI processing):
- "Thank you for contacting us! I'll help you with that."
- "Let me check your order status."

## 🛠️ Troubleshooting

**Issue**: Jupyter asyncio errors
**Solution**: Use the notebook code from `notebook_cells.txt` which includes `nest_asyncio.apply()` and disables the reranker.

**Issue**: Empty AI responses
**Solution**: Check OpenAI API key in `.env` file and ensure knowledge base documents are present.

**Issue**: No knowledge base documents
**Solution**: Add `.txt` files to the `knowledge_base/` directory and restart the application.

## 📄 License

This project is for demonstration purposes. Ensure compliance with OpenAI's usage policies when deploying in production. 