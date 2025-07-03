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
    def __init__(self, knowledge_base_path: str = 'knowledge_base', use_faiss: bool = False):
        self.knowledge_base_path = knowledge_base_path
        self.use_faiss = use_faiss
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
                similarity_top_k=3,
                streaming=False
            )
            print(f"DEBUG: Query engine created with similarity_top_k=3")
            
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

    def get_suggestions(self, message: str, conversation_context: List[Dict], sender: str) -> List[str]:
        """Generate AI suggestions based on conversation context and sender type."""
        print(f"DEBUG: get_suggestions called")
        print(f"DEBUG: message: '{message}'")
        print(f"DEBUG: sender: '{sender}'")
        print(f"DEBUG: conversation_context: {conversation_context}")
        
        if not self.index:
            print("DEBUG: Index not available")
            return ["Knowledge base is not loaded. Please add documents to the knowledge_base folder."]
        
        try:
            # Create context-aware suggestions based on sender type
            if sender == 'customer':
                print("DEBUG: Generating customer suggestions")
                suggestions = self._generate_customer_suggestions(message, conversation_context)
            elif sender == 'agent':
                print("DEBUG: Generating agent suggestions")
                suggestions = self._generate_agent_suggestions(message, conversation_context)
            else:
                print(f"DEBUG: Unknown sender type: {sender}")
                suggestions = ["Please specify sender type (customer or agent)"]
            
            print(f"DEBUG: Final suggestions: {suggestions}")
            return suggestions
            
        except Exception as e:
            print(f"DEBUG: Exception in get_suggestions: {e}")
            return [f"Error generating suggestions: {e}"]

    def _generate_customer_suggestions(self, message: str, conversation_context: List[Dict]) -> List[str]:
        """Generate suggestions for customer messages."""
        print("DEBUG: _generate_customer_suggestions called")
        
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
            
            response_str = str(response)
            print(f"DEBUG: Customer response as string: '{response_str}'")
            print(f"DEBUG: Customer response length: {len(response_str)}")
            
            # Parse the response into a list of suggestions
            suggestions = self._parse_suggestions(response_str)
            print(f"DEBUG: Parsed customer suggestions: {suggestions}")
            
            final_suggestions = suggestions[:3]  # Return max 3 suggestions
            print(f"DEBUG: Final customer suggestions: {final_suggestions}")
            
            return final_suggestions
            
        except Exception as e:
            print(f"DEBUG: Exception in _generate_customer_suggestions: {e}")
            return [f"Error generating customer suggestions: {e}"]

    def _generate_agent_suggestions(self, message: str, conversation_context: List[Dict]) -> List[str]:
        """Generate suggestions for agent messages."""
        print("DEBUG: _generate_agent_suggestions called")
        
        # Build context from conversation history
        context = self._build_conversation_context(conversation_context)
        print(f"DEBUG: Built context: '{context}'")
        
        # Create a prompt for agent assistance
        prompt = f"""
        Based on the agent's message: "{message}"
        And conversation context: {context}
        
        Generate 3 suggestions for improving the agent's response or providing additional information.
        Focus on knowledge base information, best practices, and professional communication.
        """
        print(f"DEBUG: Agent prompt: '{prompt}'")
        
        try:
            print("DEBUG: Sending query to query engine for agent suggestions...")
            response = self.query_engine.query(prompt)
            print(f"DEBUG: Raw agent response: {response}")
            print(f"DEBUG: Agent response type: {type(response)}")
            
            response_str = str(response)
            print(f"DEBUG: Agent response as string: '{response_str}'")
            print(f"DEBUG: Agent response length: {len(response_str)}")
            
            # Parse the response into a list of suggestions
            suggestions = self._parse_suggestions(response_str)
            print(f"DEBUG: Parsed agent suggestions: {suggestions}")
            
            final_suggestions = suggestions[:3]  # Return max 3 suggestions
            print(f"DEBUG: Final agent suggestions: {final_suggestions}")
            
            return final_suggestions
            
        except Exception as e:
            print(f"DEBUG: Exception in _generate_agent_suggestions: {e}")
            return [f"Error generating agent suggestions: {e}"]

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
            'vector_store_type': 'FAISS' if self.use_faiss else 'Default'
        }

    def debug_test_system(self):
        """Test system with debug output."""
        print("=== DEBUG: Testing RAG System ===")
        
        # Test basic question
        print("\n1. Testing basic question...")
        result = self.answer_question("What is this knowledge base about?")
        print(f"Result: '{result}'")
        
        # Test suggestions
        print("\n2. Testing customer suggestions...")
        suggestions = self.get_suggestions(
            message="I need help with my account",
            conversation_context=[],
            sender="customer"
        )
        print(f"Suggestions: {suggestions}")
        
        # Test agent suggestions  
        print("\n3. Testing agent suggestions...")
        suggestions = self.get_suggestions(
            message="I understand your concern",
            conversation_context=[{"sender": "customer", "message": "I need help with my account"}],
            sender="agent"
        )
        print(f"Suggestions: {suggestions}")
        
        print("\n=== DEBUG: Test Complete ===")
        
        return {
            'basic_question': result,
            'customer_suggestions': suggestions
        }