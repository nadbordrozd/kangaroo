from flask import Flask, render_template, request, jsonify
from assistant import Assistant
import os
import asyncio
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Initialize Assistant
# The Assistant uses the new architecture with configurable models and async support
assistant = Assistant(
    knowledge_base_dir="knowledge_base",
    cache_dir="storage",
    embedding_model="text-embedding-3-large",
    reranker_model="gpt-4.1-mini",
    generator_model="gpt-4.1"
)

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
            'summary': 'Agent messages are not processed by AI system',
            'message': 'Agent messages are not processed by AI system'
        })

    # Get conversation context (last few messages)
    conversation_context = data.get('conversation_context', [])
    
    # Convert conversation context to the format expected by Assistant
    messages = []
    for msg in conversation_context:
        role = "user" if msg.get('sender') == 'customer' else "assistant"
        messages.append({
            "role": role,
            "content": msg.get('message', '')
        })
    
    # Add current message
    messages.append({
        "role": "user",
        "content": message
    })
    
    # Generate AI suggestions and summary using the new Assistant
    try:
        # Run async functions in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Get suggestions and summary
        suggestions, knowledge_snippets = loop.run_until_complete(assistant.get_suggestions(messages))
        summary = loop.run_until_complete(assistant.get_summary(messages))
        
        loop.close()
        
        
    except Exception as e:
        print(f"Error processing message: {e}")
        suggestions = ["I apologize, but I'm experiencing technical difficulties. Please try again."]
        knowledge_snippets = []
        summary = "Error processing request"
    
    response_data = {
        'success': True,
        'suggestions': suggestions,
        'knowledge_snippets': knowledge_snippets,  # Send full snippet objects with content and file_name
        'summary': summary
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
    try:
        # Check if knowledge base is loaded
        knowledge_store = assistant.knowledge_store
        doc_count = len(knowledge_store.document_store.filter_documents())
        
        status = {
            'status': 'healthy',
            'knowledge_base_loaded': doc_count > 0,
            'document_count': doc_count,
            'embedding_model': assistant.embedding_model,
            'reranker_model': assistant.reranker_model,
            'generator_model': assistant.generator_model,
            'cache_directory': assistant.cache_dir
        }
        return jsonify(status)
    except Exception as e:
        return jsonify({
            'status': 'error',
            'error': str(e)
        })



@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    print("Starting Contact Center AI Assistant...")
    print("Make sure to:")
    print("1. Set your OPENAI_API_KEY in a .env file")
    print("2. Add your knowledge base files to the 'knowledge_base' folder") 
    print("3. The app will be available at http://localhost:5000")
    print("Note: Now using the new Assistant architecture!")
    print(f"Configuration:")
    print(f"  - Embedding Model: {assistant.embedding_model}")
    print(f"  - Reranker Model: {assistant.reranker_model}")
    print(f"  - Generator Model: {assistant.generator_model}")
    

    
    app.run(debug=False, host='0.0.0.0', port=5000)