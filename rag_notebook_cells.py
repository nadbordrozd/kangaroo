# =============================================================================
# RAG System UI - Complete Notebook Code 
# Copy each section below into separate Jupyter notebook cells
# This keeps rag_system.py unchanged and handles asyncio issues in the notebook
# =============================================================================

# CELL 1 - Setup and Imports
# =============================================================================
import os
import json
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output
from datetime import datetime
from dotenv import load_dotenv

# Fix for Jupyter asyncio compatibility - MUST be done before importing rag_system
print("🔧 Applying Jupyter asyncio fix...")
import nest_asyncio
nest_asyncio.apply()

from rag_system import RAGSystem

# Load environment variables
load_dotenv()

# Initialize RAG System (disable reranker to avoid asyncio issues in Jupyter)
print("🚀 Initializing RAG System...")
try:
    rag_system = RAGSystem(
        similarity_top_k=5,
        persist_dir='./storage',
        use_reranker=False,  # Disabled for Jupyter compatibility - keeps rag_system.py unchanged
        conversation_context_length=5
    )
    print("✅ RAG System initialized successfully!")
    print("ℹ️  Note: Reranker disabled for Jupyter compatibility (rag_system.py unchanged)")
    status = rag_system.get_system_status()
    print(f"📚 {status['document_count']} knowledge base documents loaded")
except Exception as e:
    print(f"❌ Error initializing RAG System: {e}")
    rag_system = None

# Global conversation storage
conversation_history = []

print(f"🔑 OpenAI API Key configured: {'✅' if os.getenv('OPENAI_API_KEY') else '❌'}")


# CELL 2 - Create UI Components
# =============================================================================

# LEFT PANEL - Chat Interface
chat_display = widgets.Output(layout=widgets.Layout(height='350px', overflow='auto', border='1px solid #ccc'))

# Customer message input
customer_input = widgets.Textarea(
    placeholder="Type customer message here...",
    layout=widgets.Layout(width='100%', height='60px')
)
customer_button = widgets.Button(
    description="Send as Customer",
    button_style='primary',
    layout=widgets.Layout(width='150px')
)

# Agent message input  
agent_input = widgets.Textarea(
    placeholder="Type agent response here...",
    layout=widgets.Layout(width='100%', height='60px')
)
agent_button = widgets.Button(
    description="Send as Agent", 
    button_style='success',
    layout=widgets.Layout(width='150px')
)

clear_button = widgets.Button(
    description="Clear Chat",
    button_style='warning',
    layout=widgets.Layout(width='100px')
)

# Chat panel layout
chat_panel = widgets.VBox([
    widgets.HTML("<h3>💬 Chat Interface</h3>"),
    chat_display,
    widgets.HTML("<b>Customer Message:</b>"),
    customer_input,
    customer_button,
    widgets.HTML("<br><b>Agent Response:</b>"),
    agent_input,
    agent_button,
    widgets.HTML("<br>"),
    clear_button
], layout=widgets.Layout(width='32%', padding='10px', border='1px solid #ddd'))

# MIDDLE PANEL - AI Assistant
summary_widget = widgets.HTML(
    value="<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;'>Conversation summary will appear here...</div>",
    layout=widgets.Layout(height='120px')
)

suggestions_widget = widgets.HTML(
    value="<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #28a745;'>AI suggestions will appear here...</div>",
    layout=widgets.Layout(height='250px', overflow='auto')
)

# AI Assistant panel layout
ai_panel = widgets.VBox([
    widgets.HTML("<h3>🤖 AI Assistant</h3>"),
    widgets.HTML("<h4>📋 Summary</h4>"),
    summary_widget,
    widgets.HTML("<h4>💡 Suggestions</h4>"),
    suggestions_widget
], layout=widgets.Layout(width='36%', padding='10px', border='1px solid #ddd'))

# RIGHT PANEL - Knowledge Articles
knowledge_widget = widgets.HTML(
    value="<div style='padding: 15px; background: #f8f9fa; border-radius: 5px;'>Knowledge articles will appear here when you send a customer message...</div>",
    layout=widgets.Layout(height='400px', overflow='auto')
)

# Knowledge panel layout with green header
knowledge_panel = widgets.VBox([
    widgets.HTML("<h3 style='background: linear-gradient(135deg, #4caf50 0%, #45a049 100%); color: white; padding: 10px; margin: -10px -10px 10px -10px; border-radius: 5px 5px 0 0;'>📚 Knowledge Articles</h3>"),
    knowledge_widget
], layout=widgets.Layout(width='32%', padding='10px', border='1px solid #ddd'))

print("✅ UI Components created")


# CELL 3 - Helper Functions
# =============================================================================

def add_chat_message(sender, message):
    """Add a message to the chat display."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    
    if sender == "customer":
        bg_color = "#e3f2fd"
        align = "margin-left: 20px;"
        label = "🧑 Customer"
    else:  # agent
        bg_color = "#f3e5f5" 
        align = "margin-right: 20px;"
        label = "👨‍💼 Agent"
    
    with chat_display:
        display(HTML(f"""
        <div style='padding: 8px 12px; margin: 5px 0; background: {bg_color}; border-radius: 8px; {align}'>
            <strong>{label}</strong> <small>[{timestamp}]</small><br>
            {message.replace(chr(10), '<br>')}
        </div>
        """))

def update_summary(summary_text):
    """Update the summary widget."""
    if summary_text:
        summary_widget.value = f"<div style='padding: 15px; background: #e8f5e8; border-radius: 5px; border-left: 4px solid #4caf50;'>{summary_text}</div>"
    else:
        summary_widget.value = "<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #007bff;'>Conversation summary will appear here...</div>"

def update_suggestions(suggestions_list):
    """Update the suggestions widget."""
    if suggestions_list:
        suggestions_html = ""
        for i, suggestion in enumerate(suggestions_list, 1):
            suggestions_html += f"""
            <div style='padding: 10px; margin: 8px 0; background: #e8f4fd; border-radius: 5px; border-left: 4px solid #2196f3;'>
                <strong>Suggestion {i}:</strong><br>
                {suggestion}
            </div>
            """
        suggestions_widget.value = suggestions_html
    else:
        suggestions_widget.value = "<div style='padding: 15px; background: #f8f9fa; border-radius: 5px; border-left: 4px solid #28a745;'>AI suggestions will appear here...</div>"

def update_knowledge_articles(knowledge_snippets):
    """Update the knowledge articles widget."""
    if knowledge_snippets:
        knowledge_html = ""
        for i, snippet in enumerate(knowledge_snippets, 1):
            # Truncate long snippets
            display_snippet = snippet[:200] + "..." if len(snippet) > 200 else snippet
            knowledge_html += f"""
            <div style='padding: 10px; margin: 8px 0; background: #fff8e1; border-radius: 5px; border-left: 4px solid #ff9800;'>
                <strong>Article {i}:</strong><br>
                <small>{display_snippet}</small>
            </div>
            """
        knowledge_widget.value = knowledge_html
    else:
        knowledge_widget.value = "<div style='padding: 15px; background: #f8f9fa; border-radius: 5px;'>Knowledge articles will appear here when you send a customer message...</div>"

print("✅ Helper functions created")


# CELL 4 - Event Handlers
# =============================================================================

def on_customer_message(b):
    """Handle customer message - triggers RAG system."""
    message = customer_input.value.strip()
    if not message:
        return
    
    # Add customer message to chat
    add_chat_message("customer", message)
    conversation_history.append({
        'sender': 'customer',
        'message': message,
        'timestamp': datetime.now().isoformat()
    })
    
    # Clear input
    customer_input.value = ""
    
    # Process with RAG system (only for customer messages)
    if rag_system:
        try:
            # Show processing indicator
            update_summary("🔄 Processing customer message...")
            update_suggestions(["🔄 Generating AI suggestions..."])
            update_knowledge_articles(["🔄 Retrieving relevant knowledge articles..."])
            
            # Get AI response
            result = rag_system.get_suggestions(message, conversation_history[:-1], 'customer')
            
            # Update UI with results
            summary = result.get('summary', '')
            suggestions = result.get('suggestions', [])
            knowledge_snippets = result.get('knowledge_snippets', [])
            
            update_summary(summary)
            update_suggestions(suggestions)
            update_knowledge_articles(knowledge_snippets)
            
        except Exception as e:
            update_summary(f"❌ Error: {str(e)}")
            update_suggestions([f"❌ Error generating suggestions: {str(e)}"])
            update_knowledge_articles([f"❌ Error retrieving knowledge: {str(e)}"])

def on_agent_message(b):
    """Handle agent message - no RAG processing."""
    message = agent_input.value.strip()
    if not message:
        return
    
    # Add agent message to chat
    add_chat_message("agent", message)
    conversation_history.append({
        'sender': 'agent',
        'message': message,
        'timestamp': datetime.now().isoformat()
    })
    
    # Clear input
    agent_input.value = ""

def on_clear_chat(b):
    """Clear the chat and reset all widgets."""
    global conversation_history
    conversation_history = []
    
    # Clear chat display
    with chat_display:
        clear_output()
    
    # Reset widgets
    update_summary("")
    update_suggestions([])
    update_knowledge_articles([])
    
    # Clear inputs
    customer_input.value = ""
    agent_input.value = ""

# Attach event handlers
customer_button.on_click(on_customer_message)
agent_button.on_click(on_agent_message)
clear_button.on_click(on_clear_chat)

print("✅ Event handlers attached")


# CELL 5 - Display the UI
# =============================================================================

# Create the main three-column layout
main_ui = widgets.HBox([
    chat_panel,      # Left: Chat Interface  
    ai_panel,        # Middle: AI Assistant
    knowledge_panel  # Right: Knowledge Articles
], layout=widgets.Layout(width='100%'))

# Add welcome message
add_chat_message("customer", "Hello, I need help with my order")

# Show initial knowledge base status
if rag_system:
    status = rag_system.get_system_status()
    initial_kb_info = f"""
    <div style='padding: 15px; background: #e8f5e8; border-radius: 5px;'>
        <strong>📊 Knowledge Base Status:</strong><br>
        • {status['document_count']} documents loaded<br>
        • Vector Store: {status['vector_store_type']}<br>
        • Reranker: {'Enabled' if status['use_reranker'] else 'Disabled (Jupyter mode)'}<br>
        • Ready to assist with customer inquiries
    </div>
    """
    knowledge_widget.value = initial_kb_info

# Display the interface
display(main_ui)

print("🎉 RAG System UI is ready!")
print("💬 Type a customer message and click 'Send as Customer' to see AI suggestions")
print("👨‍💼 Type agent responses and click 'Send as Agent' to continue the conversation")
print("📚 Knowledge articles will appear when processing customer messages")


# CELL 6 - Optional Quick Test Functions
# =============================================================================

def quick_test(scenario):
    """Load a test scenario into the customer input."""
    scenarios = {
        "greeting": "Hello, I need help with ordering exotic meats",
        "product": "I'm interested in kangaroo steaks for a dinner party", 
        "shipping": "How long does shipping take and what are the costs?",
        "complaint": "I received the wrong order and need to return it",
        "food_safety": "What are your food safety certifications?"
    }
    
    if scenario in scenarios:
        customer_input.value = scenarios[scenario]
        print(f"✅ Loaded scenario: {scenarios[scenario]}")
        print("Click 'Send as Customer' to process this message")
    else:
        print(f"❌ Unknown scenario. Available: {list(scenarios.keys())}")

# Test buttons
test_greeting = widgets.Button(description="Test: Greeting", button_style='info')
test_product = widgets.Button(description="Test: Product", button_style='info') 
test_shipping = widgets.Button(description="Test: Shipping", button_style='info')

test_greeting.on_click(lambda b: quick_test("greeting"))
test_product.on_click(lambda b: quick_test("product"))
test_shipping.on_click(lambda b: quick_test("shipping"))

print("🧪 Quick Test Utilities:")
display(widgets.HBox([test_greeting, test_product, test_shipping])) 