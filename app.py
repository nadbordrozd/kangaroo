from flask import Flask, render_template, request, jsonify
from rag_system import RAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize RAG system
# similarity_top_k=5 means retrieve top 5 most relevant chunks for each query
# persist_dir stores the cached embeddings to avoid re-computing on restart
# use_reranker=True enables LLM-based relevance filtering of retrieved chunks (more API calls but better quality)
rag_system = RAGSystem(similarity_top_k=5, persist_dir='./storage', use_reranker=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    sender = data.get('sender')  # 'customer' or 'agent'
    message = data.get('message')
    
    # Only process customer messages - agents should not trigger AI processing
    if sender != 'customer':
        return jsonify({
            'success': True,
            'suggestions': [],
            'knowledge_snippets': [],
            'message': 'Agent messages are not processed by AI system'
        })

    # Get conversation context (last few messages)
    conversation_context = data.get('conversation_context', [])
    
    # Generate AI suggestions based on the conversation
    result = rag_system.get_suggestions(message, conversation_context, sender)
    
    # Extract suggestions and knowledge snippets from the result
    if isinstance(result, dict):
        suggestions = result.get('suggestions', [])
        knowledge_snippets = result.get('knowledge_snippets', [])
    else:
        # Fallback for backward compatibility
        suggestions = result if isinstance(result, list) else [str(result)]
        knowledge_snippets = []
    
    response_data = {
        'success': True,
        'suggestions': suggestions,
        'knowledge_snippets': knowledge_snippets
    }
    
    return jsonify(response_data)



@app.route('/test_simple', methods=['GET'])
def test_simple():
    """Simple test route to verify JSON responses work."""
    return jsonify({
        'success': True,
        'message': 'Flask is working correctly',
        'suggestions': [
            'Test suggestion 1',
            'Test suggestion 2', 
            'Test suggestion 3'
        ]
    })





@app.route('/system_status')
def system_status():
    """Get the system status."""
    status = rag_system.get_system_status()
    return jsonify(status)

@app.route('/clear_cache', methods=['POST'])
def clear_cache():
    """Clear the cached embeddings."""
    try:
        rag_system.clear_cache()
        return jsonify({
            'success': True,
            'message': 'Cache cleared successfully. Restart the app to rebuild embeddings.'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Contact Center AI Assistant...")
    print("Make sure to:")
    print("1. Set your OPENAI_API_KEY in a .env file")
    print("2. Add your knowledge base files to the 'knowledge_base' folder") 
    print("3. The app will be available at http://localhost:5000")
    print("Note: Now using OpenAI GPT-4 for both embeddings and text generation!")
    print(f"Configuration: Retrieving top {rag_system.similarity_top_k} most relevant chunks per query")
    print(f"Storage: Embeddings cached in '{rag_system.persist_dir}' directory")
    if rag_system.use_reranker:
        print("Reranker: ENABLED - Using GPT-4o-mini for relevance filtering (higher quality, more API calls)")
    else:
        print("Reranker: DISABLED - Using all retrieved chunks (faster, fewer API calls)")
    
    # Check system status on startup
    status = rag_system.get_system_status()
    print(f"System Status: {status}")
    
    app.run(debug=False, host='0.0.0.0', port=5000)