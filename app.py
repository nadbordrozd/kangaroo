from flask import Flask, render_template, request, jsonify
from rag_system import RAGSystem
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize RAG system
# Set use_faiss=True if you want to use FAISS (requires: pip install faiss-cpu)
# Set use_faiss=False to use the default vector store (simpler, no extra dependencies)
# similarity_top_k=5 means retrieve top 5 most relevant chunks for each query
rag_system = RAGSystem(use_faiss=False, similarity_top_k=5)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_message', methods=['POST'])
def send_message():
    data = request.json
    sender = data.get('sender')  # 'customer' or 'agent'
    message = data.get('message')
    
    print(f"DEBUG FLASK: Received request - sender: '{sender}', message: '{message}'")
    
    # Only process customer messages - agents should not trigger AI processing
    if sender != 'customer':
        print(f"DEBUG FLASK: Ignoring {sender} message - only customer messages are processed")
        return jsonify({
            'success': True,
            'suggestions': [],
            'knowledge_snippets': [],
            'message': 'Agent messages are not processed by AI system'
        })

    # Get conversation context (last few messages)
    conversation_context = data.get('conversation_context', [])
    print(f"DEBUG FLASK: Conversation context: {conversation_context}")
    
    # Generate AI suggestions based on the conversation
    result = rag_system.get_suggestions(message, conversation_context, sender)
    print(f"DEBUG FLASK: Generated result: {result}")
    
    # Extract suggestions and knowledge snippets from the result
    if isinstance(result, dict):
        suggestions = result.get('suggestions', [])
        knowledge_snippets = result.get('knowledge_snippets', [])
    else:
        # Fallback for backward compatibility
        suggestions = result if isinstance(result, list) else [str(result)]
        knowledge_snippets = []
    
    print(f"DEBUG FLASK: Suggestions: {suggestions}")
    print(f"DEBUG FLASK: Knowledge snippets: {knowledge_snippets}")
    print(f"DEBUG FLASK: Number of suggestions: {len(suggestions)}")
    print(f"DEBUG FLASK: Number of knowledge snippets: {len(knowledge_snippets)}")
    
    response_data = {
        'success': True,
        'suggestions': suggestions,
        'knowledge_snippets': knowledge_snippets
    }
    print(f"DEBUG FLASK: Response data: {response_data}")
    
    return jsonify(response_data)

@app.route('/chat', methods=['POST'])
def chat():
    """New endpoint for conversational chat with memory."""
    data = request.json
    message = data.get('message')
    
    if not message:
        return jsonify({'success': False, 'error': 'Message is required'}), 400
    
    response = rag_system.chat(message)
    
    return jsonify({
        'success': True,
        'response': response
    })

@app.route('/reset_conversation', methods=['POST'])
def reset_conversation():
    """Reset the conversation memory."""
    rag_system.reset_conversation()
    return jsonify({'success': True, 'message': 'Conversation reset'})

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

@app.route('/test_question', methods=['POST'])
def test_question():
    """Test a simple question with debug output."""
    data = request.json
    question = data.get('question', 'What is this knowledge base about?')
    
    print(f"DEBUG: Testing question: '{question}'")
    response = rag_system.answer_question(question)
    
    return jsonify({
        'success': True,
        'question': question,
        'response': response
    })

@app.route('/debug_test')
def debug_test():
    """Test the RAG system with debug output."""
    try:
        results = rag_system.debug_test_system()
        return jsonify({
            'success': True,
            'results': results
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/system_status')
def system_status():
    """Get the system status."""
    status = rag_system.get_system_status()
    return jsonify(status)

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
    
    # Check system status on startup
    status = rag_system.get_system_status()
    print(f"System Status: {status}")
    
    app.run(debug=True, host='0.0.0.0', port=5000)