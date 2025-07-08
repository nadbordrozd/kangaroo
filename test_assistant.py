"""
Test script for the Assistant class.
Tests the assistant's response to a customer asking about kangaroo meat.
"""

import asyncio
from assistant import Assistant


async def test_assistant():
    """Test the assistant with a kangaroo meat inquiry."""
    
    print("Initializing Assistant...")
    print("-" * 50)
    
    # Initialize the assistant
    assistant = Assistant(
        knowledge_base_dir="knowledge_base",
        cache_dir="storage",
        embedding_model="text-embedding-3-small",
        reranker_model="gpt-4.1",
        generator_model="gpt-4.1"
    )
    
    print("Assistant initialized successfully!")
    print()
    
    # Create a conversation where customer asks about kangaroo
    messages = [
        {
            "role": "customer", 
            "content": "Hi, I'm looking for something exotic for a special dinner. Do you carry kangaroo meat?"
        },
        {
            "role": "agent", 
            "content": "Hello! Let me check what exotic meats we have available for you."
        },
        {
            "role": "customer", 
            "content": "do you carry kangaroo"
        }
    ]
    
    print("Customer Conversation:")
    print("-" * 50)
    for msg in messages:
        print(f"{msg['role'].title()}: {msg['content']}")
    print()
    
    print("Processing request...")
    print("-" * 50)
    
    # Get assistant suggestions
    try:
        suggestion, knowledge_snippets = await assistant.get_suggestions(messages)
        
        print("ASSISTANT SUGGESTION:")
        print("-" * 30)
        print(suggestion)
        print()
        
        print("KNOWLEDGE SNIPPETS USED:")
        print("-" * 30)
        if knowledge_snippets:
            for i, snippet in enumerate(knowledge_snippets, 1):
                print(f"Snippet {i}:")
                print(f"  File: {snippet.get('file_name', 'unknown')}")
                print(f"  Score: {snippet.get('score', 'N/A')}")
                print(f"  Content: {snippet['content'][:200]}{'...' if len(snippet['content']) > 200 else ''}")
                print()
        else:
            print("No relevant knowledge snippets found.")
        
        print()
        
        # Also test summary
        print("CONVERSATION SUMMARY:")
        print("-" * 30)
        summary = await assistant.get_summary(messages)
        print(summary)
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()


async def main():
    """Main test function."""
    print("=== ASSISTANT TEST ===")
    print("Testing kangaroo meat inquiry")
    print("=" * 50)
    print()
    
    await test_assistant()
    
    print()
    print("=" * 50)
    print("Test completed!")


if __name__ == "__main__":
    asyncio.run(main()) 