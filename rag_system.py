import os
from typing import List, Dict, Optional
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, knowledge_base_path: str = 'knowledge_base', use_faiss: bool = False, similarity_top_k: int = 5):
        self.knowledge_base_path = knowledge_base_path
        self.use_faiss = use_faiss
        self.similarity_top_k = similarity_top_k
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        self.chat_engine = None
        self._setup_system()

    def _setup_system(self):
        """Setup the RAG system with OpenAI models and LlamaIndex components."""
        # Configure global settings
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        # Set up models using Settings (LlamaIndex's global configuration)
        Settings.llm = OpenAI(model="gpt-4", api_key=openai_api_key, temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large", 
            api_key=openai_api_key
        )
        
        # Set up document processing
        Settings.node_parser = SentenceSplitter(
            chunk_size=1024, 
            chunk_overlap=100
        )
        
        self._load_knowledge_base()

    def _setup_vector_store(self):
        """Setup vector store based on configuration."""
        if self.use_faiss:
            try:
                import faiss
                from llama_index.vector_stores.faiss import FaissVectorStore
                
                # Get embedding dimension (OpenAI text-embedding-3-large is 3072 dimensions)
                embedding_dim = 3072
                faiss_index = faiss.IndexFlatL2(embedding_dim)
                vector_store = FaissVectorStore(faiss_index=faiss_index)
                print("Using FAISS vector store")
                return vector_store
            except ImportError:
                print("FAISS not available, falling back to default vector store")
                return None
        else:
            # Use default vector store (SimpleVectorStore)
            print("Using default vector store")
            return None

    def _load_knowledge_base(self):
        """Load and index the knowledge base documents."""
        # Create directory if it doesn't exist
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path)
            print(f"Created {self.knowledge_base_path} directory. Please add your text files there.")
            return
        
        # Check for documents
        txt_files = [f for f in os.listdir(self.knowledge_base_path) if f.endswith('.txt')]
        if not txt_files:
            print("No .txt files found in knowledge_base directory.")
            return
        
        try:
            # Load documents using SimpleDirectoryReader
            documents = SimpleDirectoryReader(
                input_dir=self.knowledge_base_path,
                required_exts=['.txt']
            ).load_data()
            
            if not documents:
                print("No documents loaded from knowledge base.")
                return
            
            # Setup vector store
            vector_store = self._setup_vector_store()
            
            # Create index
            if vector_store:
                self.index = VectorStoreIndex.from_documents(
                    documents,
                    vector_store=vector_store
                )
            else:
                # Use default vector store
                self.index = VectorStoreIndex.from_documents(documents)
            
            # Set up query engine
            self.query_engine = self.index.as_query_engine(
                similarity_top_k=self.similarity_top_k,
                streaming=False
            )
            print(f"DEBUG: Query engine created with similarity_top_k={self.similarity_top_k}")
            
            # Set up chat engine with memory for conversation context
            memory = ChatMemoryBuffer.from_defaults(token_limit=3000)
            self.chat_engine = CondenseQuestionChatEngine.from_defaults(
                query_engine=self.query_engine,
                memory=memory
            )
            print(f"DEBUG: Chat engine created with memory buffer")
            
            print(f"Knowledge base indexed successfully from {len(txt_files)} files.")
            print(f"DEBUG: Total documents processed: {len(documents)}")
            
            # Test query engine
            print("DEBUG: Testing query engine with simple question...")
            try:
                test_response = self.query_engine.query("What is this knowledge base about?")
                print(f"DEBUG: Test query response: '{test_response}'")
                print(f"DEBUG: Test query response type: {type(test_response)}")
            except Exception as e:
                print(f"DEBUG: Test query failed: {e}")
            
        except Exception as e:
            print(f"Error loading knowledge base: {e}")
            print(f"DEBUG: Full exception details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()

    def answer_question(self, question: str) -> str:
        """Answer a standalone question using the knowledge base."""
        print(f"DEBUG: answer_question called with: {question}")
        
        if not self.query_engine:
            print("DEBUG: Query engine not available")
            return "Knowledge base is not loaded or indexed."
        
        try:
            print("DEBUG: Sending query to query engine...")
            response = self.query_engine.query(question)
            print(f"DEBUG: Raw response received: {response}")
            print(f"DEBUG: Response type: {type(response)}")
            
            response_str = str(response)
            print(f"DEBUG: Response as string: '{response_str}'")
            print(f"DEBUG: Response length: {len(response_str)}")
            
            return response_str
        except Exception as e:
            print(f"DEBUG: Exception in answer_question: {e}")
            return f"Error answering question: {e}"

    def get_suggestions(self, message: str, conversation_context: List[Dict], sender: str) -> Dict[str, List[str]]:
        """Generate AI suggestions and retrieve relevant knowledge base snippets."""
        print(f"DEBUG: get_suggestions called")
        print(f"DEBUG: message: '{message}'")
        print(f"DEBUG: sender: '{sender}'")
        print(f"DEBUG: conversation_context: {conversation_context}")
        
        if not self.index:
            print("DEBUG: Index not available")
            return {
                "suggestions": ["Knowledge base is not loaded. Please add documents to the knowledge_base folder."],
                "knowledge_snippets": []
            }
        
        try:
            # Create context-aware suggestions based on sender type
            if sender == 'customer':
                print("DEBUG: Customer message - Using RAG to generate agent response suggestions")
                result = self._generate_customer_suggestions(message, conversation_context)
            elif sender == 'agent':
                print("DEBUG: Agent message - Using static suggestions (NO RAG)")
                result = self._generate_agent_suggestions(message, conversation_context)
            else:
                print(f"DEBUG: Unknown sender type: {sender}")
                result = {
                    "suggestions": ["Please specify sender type (customer or agent)"],
                    "knowledge_snippets": []
                }
            
            print(f"DEBUG: Final result: {result}")
            return result
            
        except Exception as e:
            print(f"DEBUG: Exception in get_suggestions: {e}")
            return {
                "suggestions": [f"Error generating suggestions: {e}"],
                "knowledge_snippets": []
            }

    def _generate_customer_suggestions(self, message: str, conversation_context: List[Dict]) -> Dict[str, List[str]]:
        """Generate suggestions for customer messages using RAG to retrieve relevant knowledge snippets."""
        print("DEBUG: _generate_customer_suggestions called (WITH RAG)")
        
        # Build context from conversation history
        context = self._build_conversation_context(conversation_context)
        print(f"DEBUG: Built context: '{context}'")
        
        # Create a prompt for customer assistance
        prompt = f"""
        Based on the customer's message: "{message}"
        And conversation context: {context}
        
        Generate 3 helpful suggestions for how an agent might respond to assist this customer.
        Focus on being helpful, professional, and solution-oriented.
        """
        print(f"DEBUG: Customer prompt: '{prompt}'")
        
        try:
            print("DEBUG: Sending query to query engine for customer suggestions...")
            response = self.query_engine.query(prompt)
            print(f"DEBUG: Raw customer response: {response}")
            print(f"DEBUG: Customer response type: {type(response)}")
            
            # Extract suggestions
            response_str = str(response)
            print(f"DEBUG: Customer response as string: '{response_str}'")
            
            suggestions = self._parse_suggestions(response_str)
            final_suggestions = suggestions[:3]  # Return max 3 suggestions (but up to similarity_top_k knowledge snippets)
            
            # Extract knowledge snippets from source nodes
            knowledge_snippets = self._extract_knowledge_snippets(response)
            
            print(f"DEBUG: Final customer suggestions: {final_suggestions}")
            print(f"DEBUG: Knowledge snippets: {knowledge_snippets}")
            
            return {
                "suggestions": final_suggestions,
                "knowledge_snippets": knowledge_snippets
            }
            
        except Exception as e:
            print(f"DEBUG: Exception in _generate_customer_suggestions: {e}")
            return {
                "suggestions": [f"Error generating customer suggestions: {e}"],
                "knowledge_snippets": []
            }

    def _generate_agent_suggestions(self, message: str, conversation_context: List[Dict]) -> Dict[str, List[str]]:
        """Generate suggestions for agent messages without using RAG (no knowledge base retrieval)."""
        print("DEBUG: _generate_agent_suggestions called (NO RAG)")
        
        # Build context from conversation history
        context = self._build_conversation_context(conversation_context)
        print(f"DEBUG: Built context: '{context}'")
        
        # Generate static suggestions based on common best practices
        # No RAG retrieval for agent messages
        try:
            print("DEBUG: Generating static agent suggestions without RAG...")
            
            # Analyze the agent message and context to provide relevant suggestions
            agent_message_lower = message.lower()
            
            suggestions = []
            
            # Provide suggestions based on message content and best practices
            if any(word in agent_message_lower for word in ['sorry', 'apologize', 'apologies']):
                suggestions = [
                    "Consider offering a specific solution or next step after the apology.",
                    "Acknowledge the customer's frustration and provide a timeline for resolution.",
                    "Follow up with 'What can I do to make this right for you?'"
                ]
            elif any(word in agent_message_lower for word in ['understand', 'hear', 'see']):
                suggestions = [
                    "Build on this empathy by asking clarifying questions to better understand their needs.",
                    "Provide specific examples of how you can help resolve their concern.",
                    "Offer multiple options or solutions when possible."
                ]
            elif any(word in agent_message_lower for word in ['help', 'assist', 'support']):
                suggestions = [
                    "Be specific about what help you can provide and set clear expectations.",
                    "Ask if there are any other concerns while you have them on the line.",
                    "Provide your direct contact information for future assistance."
                ]
            elif any(word in agent_message_lower for word in ['thank', 'thanks']):
                suggestions = [
                    "Express genuine appreciation and ask if there's anything else you can help with.",
                    "Reinforce the value of their business and your commitment to service.",
                    "Invite them to reach out again if they have future questions."
                ]
            else:
                # Default suggestions for general agent communication
                suggestions = [
                    "Ensure you're being clear and specific in your communication.",
                    "Ask open-ended questions to better understand the customer's needs.",
                    "Provide a clear next step or timeline for any actions you'll take."
                ]
            
            print(f"DEBUG: Generated static agent suggestions: {suggestions}")
            
            return {
                "suggestions": suggestions[:3],  # Return max 3 suggestions
                "knowledge_snippets": []  # No knowledge snippets for agent messages
            }
            
        except Exception as e:
            print(f"DEBUG: Exception in _generate_agent_suggestions: {e}")
            return {
                "suggestions": [f"Error generating agent suggestions: {e}"],
                "knowledge_snippets": []
            }

    def _extract_knowledge_snippets(self, response) -> List[str]:
        """Extract knowledge base snippets from LlamaIndex response source nodes."""
        print("DEBUG: _extract_knowledge_snippets called")
        
        snippets = []
        
        try:
            # Access source nodes from the response
            if hasattr(response, 'source_nodes') and response.source_nodes:
                print(f"DEBUG: Found {len(response.source_nodes)} source nodes")
                
                for i, node in enumerate(response.source_nodes[:self.similarity_top_k]):  # Max similarity_top_k snippets
                    print(f"DEBUG: Processing source node {i}")
                    
                    # Get the text content
                    text = node.text if hasattr(node, 'text') else str(node)
                    
                    # Get the source file name
                    file_name = "Unknown"
                    if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                        file_name = node.metadata['file_name']
                    elif hasattr(node, 'node') and hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
                        file_name = node.node.metadata['file_name']
                    
                    # Create a formatted snippet with full text (no truncation)
                    snippet = f"📄 **{file_name}**\n{text}"
                    snippets.append(snippet)
                    print(f"DEBUG: Added snippet from {file_name}: {len(text)} characters")
                    
            else:
                print("DEBUG: No source nodes found in response")
                
        except Exception as e:
            print(f"DEBUG: Error extracting knowledge snippets: {e}")
            # Fallback: create a generic snippet
            snippets = ["📄 **Knowledge Base**: Information retrieved from knowledge base documents"]
        
        print(f"DEBUG: Final snippets: {len(snippets)} items")
        return snippets

    def _build_conversation_context(self, conversation_context: List[Dict]) -> str:
        """Build a string representation of the conversation context."""
        print(f"DEBUG: _build_conversation_context called with {len(conversation_context)} messages")
        
        if not conversation_context:
            print("DEBUG: No conversation context provided")
            return "No previous conversation context."
        
        context_parts = []
        for i, msg in enumerate(conversation_context[-5:]):  # Last 5 messages for context
            sender = msg.get('sender', 'Unknown')
            message = msg.get('message', '')
            context_part = f"{sender}: {message}"
            context_parts.append(context_part)
            print(f"DEBUG: Context part {i}: '{context_part}'")
        
        final_context = " | ".join(context_parts)
        print(f"DEBUG: Final context string: '{final_context}'")
        return final_context

    def _parse_suggestions(self, response: str) -> List[str]:
        """Parse AI response into a list of suggestions."""
        print(f"DEBUG: _parse_suggestions called with response: '{response}'")
        print(f"DEBUG: Response length: {len(response)}")
        
        # Split response into lines and clean up
        lines = response.strip().split('\n')
        print(f"DEBUG: Split into {len(lines)} lines")
        
        suggestions = []
        
        for i, line in enumerate(lines):
            print(f"DEBUG: Processing line {i}: '{line}'")
            line = line.strip()
            
            if line and not line.startswith('Based on') and len(line) > 10:
                print(f"DEBUG: Line passes filters: '{line}'")
                # Remove numbering and bullet points
                cleaned_line = line.lstrip('123456789. -•*').strip()
                print(f"DEBUG: Cleaned line: '{cleaned_line}'")
                
                if cleaned_line:
                    suggestions.append(cleaned_line)
                    print(f"DEBUG: Added suggestion: '{cleaned_line}'")
        
        print(f"DEBUG: Parsed {len(suggestions)} suggestions: {suggestions}")
        
        # If no structured suggestions found, return the whole response as one suggestion
        if not suggestions and response.strip():
            print("DEBUG: No structured suggestions found, using whole response")
            suggestions = [response.strip()]
        
        print(f"DEBUG: Final parsed suggestions: {suggestions}")
        return suggestions

    def chat(self, message: str) -> str:
        """Chat with the system using conversation memory."""
        print(f"DEBUG: chat method called with message: '{message}'")
        
        if not self.chat_engine:
            print("DEBUG: Chat engine not available")
            return "Chat engine is not available. Please ensure the knowledge base is loaded."
        
        try:
            print("DEBUG: Sending message to chat engine...")
            response = self.chat_engine.chat(message)
            print(f"DEBUG: Raw chat response: {response}")
            print(f"DEBUG: Chat response type: {type(response)}")
            
            response_str = str(response)
            print(f"DEBUG: Chat response as string: '{response_str}'")
            print(f"DEBUG: Chat response length: {len(response_str)}")
            
            return response_str
        except Exception as e:
            print(f"DEBUG: Exception in chat: {e}")
            import traceback
            traceback.print_exc()
            return f"Error in chat: {e}"

    def reset_conversation(self):
        """Reset the conversation memory."""
        if self.chat_engine:
            self.chat_engine.reset()

    def get_system_status(self) -> Dict[str, bool]:
        """Get the status of system components."""
        return {
            'knowledge_base_loaded': self.index is not None,
            'query_engine_ready': self.query_engine is not None,
            'chat_engine_ready': self.chat_engine is not None,
            'documents_found': len([f for f in os.listdir(self.knowledge_base_path) 
                                  if f.endswith('.txt')]) > 0 if os.path.exists(self.knowledge_base_path) else False,
            'vector_store_type': 'FAISS' if self.use_faiss else 'Default',
            'similarity_top_k': self.similarity_top_k
        }

    def debug_test_system(self):
        """Test system with debug output."""
        print("=== DEBUG: Testing RAG System ===")
        
        # Test basic question
        print("\n1. Testing basic question...")
        result = self.answer_question("What is this knowledge base about?")
        print(f"Result: '{result}'")
        
        # Test suggestions
        print("\n2. Testing customer suggestions (WITH RAG)...")
        customer_result = self.get_suggestions(
            message="I need help with my account",
            conversation_context=[],
            sender="customer"
        )
        print(f"Customer result (with knowledge retrieval): {customer_result}")
        
        # Test agent suggestions  
        print("\n3. Testing agent suggestions (WITHOUT RAG)...")
        agent_result = self.get_suggestions(
            message="I understand your concern",
            conversation_context=[{"sender": "customer", "message": "I need help with my account"}],
            sender="agent"
        )
        print(f"Agent result (static suggestions only): {agent_result}")
        
        print("\n=== DEBUG: Test Complete ===")
        
        return {
            'basic_question': result,
            'customer_suggestions': customer_result.get('suggestions', []) if isinstance(customer_result, dict) else customer_result,
            'customer_knowledge_snippets': customer_result.get('knowledge_snippets', []) if isinstance(customer_result, dict) else [],
            'agent_suggestions': agent_result.get('suggestions', []) if isinstance(agent_result, dict) else agent_result,
            'agent_knowledge_snippets': agent_result.get('knowledge_snippets', []) if isinstance(agent_result, dict) else []
        }