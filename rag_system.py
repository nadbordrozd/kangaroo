import os
import json
import time
import asyncio
from typing import List, Dict, Optional
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import CondenseQuestionChatEngine

# Load environment variables
load_dotenv()

class RAGSystem:
    def __init__(self, knowledge_base_path: str = 'knowledge_base', use_faiss: bool = False, similarity_top_k: int = 5, persist_dir: str = './storage', use_reranker: bool = False):
        self.knowledge_base_path = knowledge_base_path
        self.use_faiss = use_faiss
        self.similarity_top_k = similarity_top_k
        self.persist_dir = persist_dir
        self.use_reranker = use_reranker
        self.index: Optional[VectorStoreIndex] = None
        self.query_engine = None
        self.chat_engine = None
        self.reranker_llm = None
        self._setup_system()

    def _setup_system(self):
        """Setup the RAG system with OpenAI models and LlamaIndex components."""
        # Configure global settings
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        # Set up models using Settings (LlamaIndex's global configuration)git s
        Settings.llm = OpenAI(model="gpt-4", api_key=openai_api_key, temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large", 
            api_key=openai_api_key
        )
        
        # Set up reranker LLM if enabled
        if self.use_reranker:
            self.reranker_llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.0)
            print("DEBUG: Reranker LLM (GPT-4o-mini) configured")
        else:
            print("DEBUG: Reranker disabled")
        
        # Set up document processing
        Settings.node_parser = SentenceSplitter(
            chunk_size=1024, 
            chunk_overlap=100
        )
        
        self._load_knowledge_base()

    def _get_knowledge_base_files(self) -> List[str]:
        """Get list of .txt files in knowledge base directory."""
        if not os.path.exists(self.knowledge_base_path):
            return []
        return [f for f in os.listdir(self.knowledge_base_path) if f.endswith('.txt')]

    def _get_files_modification_time(self, files: List[str]) -> float:
        """Get the latest modification time of knowledge base files."""
        if not files:
            return 0
        
        latest_time = 0
        for file in files:
            file_path = os.path.join(self.knowledge_base_path, file)
            if os.path.exists(file_path):
                file_time = os.path.getmtime(file_path)
                latest_time = max(latest_time, file_time)
        return latest_time

    def _should_rebuild_index(self) -> bool:
        """Check if the index should be rebuilt based on file changes."""
        # Check if storage directory exists
        if not os.path.exists(self.persist_dir):
            print("DEBUG: Storage directory doesn't exist, will create new index")
            return True
        
        # Check if index files exist
        index_files = ['docstore.json', 'index_store.json', 'vector_store.json']
        for index_file in index_files:
            if not os.path.exists(os.path.join(self.persist_dir, index_file)):
                print(f"DEBUG: Index file {index_file} missing, will rebuild index")
                return True
        
        # Check if metadata file exists
        metadata_file = os.path.join(self.persist_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            print("DEBUG: Metadata file missing, will rebuild index")
            return True
        
        # Load metadata and compare file modification times
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            
            stored_files = set(metadata.get('files', []))
            stored_mod_time = metadata.get('modification_time', 0)
            stored_similarity_top_k = metadata.get('similarity_top_k', 3)
            stored_use_reranker = metadata.get('use_reranker', False)
            
            current_files = set(self._get_knowledge_base_files())
            current_mod_time = self._get_files_modification_time(list(current_files))
            
            # Rebuild if files changed or modification time is newer
            if stored_files != current_files:
                print("DEBUG: Knowledge base files changed, will rebuild index")
                return True
            
            if current_mod_time > stored_mod_time:
                print("DEBUG: Knowledge base files modified, will rebuild index")
                return True
            
            # Rebuild if configuration changed (though this doesn't affect cached embeddings)
            if stored_similarity_top_k != self.similarity_top_k:
                print("DEBUG: similarity_top_k changed, will rebuild index")
                return True
            
            if stored_use_reranker != self.use_reranker:
                print("DEBUG: reranker configuration changed, will rebuild index")
                return True
            
            print("DEBUG: Index is up to date, will load from storage")
            return False
            
        except (json.JSONDecodeError, KeyError) as e:
            print(f"DEBUG: Error reading metadata: {e}, will rebuild index")
            return True

    def _save_metadata(self, files: List[str]):
        """Save metadata about the current index."""
        os.makedirs(self.persist_dir, exist_ok=True)
        metadata = {
            'files': files,
            'modification_time': self._get_files_modification_time(files),
            'created_at': time.time(),
            'similarity_top_k': self.similarity_top_k,
            'use_faiss': self.use_faiss,
            'use_reranker': self.use_reranker
        }
        
        metadata_file = os.path.join(self.persist_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"DEBUG: Saved metadata to {metadata_file}")

    async def _check_chunk_relevance(self, chunk_text: str, conversation_context: str, customer_message: str) -> bool:
        """Check if a chunk is relevant to the conversation using GPT-4o-mini."""
        if not self.use_reranker or not self.reranker_llm:
            return True  # If reranker disabled, consider all chunks relevant
        
        prompt = f"""
You are a relevance judge for a customer service knowledge base system.

CUSTOMER MESSAGE: "{customer_message}"
CONVERSATION CONTEXT: {conversation_context}

KNOWLEDGE BASE CHUNK:
{chunk_text}

TASK: Determine if this knowledge base chunk is relevant to helping answer the customer's message within the conversation context.

CRITERIA:
- Is the chunk directly related to the customer's question or concern?
- Does the chunk contain information that would help a customer service agent respond appropriately?
- Would this information be useful for addressing the customer's needs?

RESPOND: Answer only with "true" or "false" (no explanation needed).
"""
        
        try:
            response = await self.reranker_llm.acomplete(prompt)
            result = str(response).strip().lower()
            return result == "true"
        except Exception as e:
            print(f"DEBUG: Error in relevance check: {e}, defaulting to relevant")
            return True  # Default to relevant if error occurs

    async def _rerank_chunks(self, chunks: List, conversation_context: str, customer_message: str) -> List:
        """Filter chunks based on relevance using parallel LLM calls."""
        if not self.use_reranker or not chunks:
            return chunks
        
        print(f"DEBUG: Reranking {len(chunks)} chunks for relevance...")
        
        # Create tasks for parallel relevance checking
        tasks = []
        for chunk in chunks:
            chunk_text = chunk.text if hasattr(chunk, 'text') else str(chunk)
            task = self._check_chunk_relevance(chunk_text, conversation_context, customer_message)
            tasks.append(task)
        
        # Execute all relevance checks in parallel
        relevance_results = await asyncio.gather(*tasks)
        
        # Filter chunks based on relevance
        relevant_chunks = []
        for chunk, is_relevant in zip(chunks, relevance_results):
            if is_relevant:
                relevant_chunks.append(chunk)
                print(f"DEBUG: Chunk relevant - keeping")
            else:
                print(f"DEBUG: Chunk not relevant - filtering out")
        
        print(f"DEBUG: Reranking complete: {len(relevant_chunks)}/{len(chunks)} chunks kept")
        return relevant_chunks

    def clear_cache(self):
        """Clear the cached embeddings and force rebuild on next load."""
        import shutil
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            print(f"DEBUG: Cleared cache directory: {self.persist_dir}")
        else:
            print(f"DEBUG: Cache directory doesn't exist: {self.persist_dir}")

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
        """Load and index the knowledge base documents with disk caching."""
        # Create directory if it doesn't exist
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path)
            print(f"Created {self.knowledge_base_path} directory. Please add your text files there.")
            return
        
        # Check for documents
        txt_files = self._get_knowledge_base_files()
        if not txt_files:
            print("No .txt files found in knowledge_base directory.")
            return
        
        try:
            # Check if we can load from existing storage
            if not self._should_rebuild_index():
                print("DEBUG: Loading index from storage...")
                try:
                    # Load from storage
                    storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                    self.index = load_index_from_storage(storage_context)
                    print(f"DEBUG: Successfully loaded index from {self.persist_dir}")
                    
                except Exception as e:
                    print(f"DEBUG: Failed to load from storage: {e}, will rebuild")
                    # Fall through to rebuild
                    pass
            
            # If we don't have an index, create it
            if self.index is None:
                print("DEBUG: Creating new index...")
                
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
                
                # Persist the index to storage
                print(f"DEBUG: Saving index to {self.persist_dir}...")
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                
                # Save metadata
                self._save_metadata(txt_files)
                
                print(f"DEBUG: Index saved to storage successfully")
                print(f"Knowledge base indexed successfully from {len(txt_files)} files.")
                print(f"DEBUG: Total documents processed: {len(documents)}")
            else:
                print(f"DEBUG: Using cached index for {len(txt_files)} files")
            
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
        """Generate AI suggestions and retrieve relevant knowledge base snippets for customer messages."""
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
            # Only process customer messages - agent messages are handled at Flask level
            if sender == 'customer':
                print("DEBUG: Customer message - Using RAG to generate agent response suggestions")
                result = self._generate_customer_suggestions(message, conversation_context)
            else:
                print(f"DEBUG: Non-customer message blocked (sender: {sender})")
                result = {
                    "suggestions": ["Only customer messages are processed by the AI system"],
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
        
        Generate one helpful suggestion for how an agent might respond to assist this customer.
        Focus on being helpful, professional, and solution-oriented.
        """
        print(f"DEBUG: Customer prompt: '{prompt}'")
        
        try:
            if self.use_reranker:
                # Use custom retrieval with reranking
                print("DEBUG: Using custom retrieval with reranking...")
                return self._generate_with_reranking(prompt, message, context)
            else:
                # Use standard query engine
                print("DEBUG: Using standard query engine...")
                response = self.query_engine.query(prompt)
                print(f"DEBUG: Raw customer response: {response}")
                print(f"DEBUG: Customer response type: {type(response)}")
                
                # Use response directly as single suggestion
                response_str = str(response).strip()
                print(f"DEBUG: Customer response as string: '{response_str}'")
                
                final_suggestions = [response_str]  # Single suggestion
                
                # Extract knowledge snippets from source nodes
                knowledge_snippets = self._extract_knowledge_snippets(response)
                
                print(f"DEBUG: Final customer suggestion: {final_suggestions}")
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

    def _generate_with_reranking(self, prompt: str, customer_message: str, conversation_context: str) -> Dict[str, List[str]]:
        """Generate suggestions using custom retrieval with reranking."""
        try:
            # Step 1: Retrieve raw chunks from vector store
            print("DEBUG: Retrieving chunks from vector store...")
            retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
            retrieved_nodes = retriever.retrieve(prompt)
            print(f"DEBUG: Retrieved {len(retrieved_nodes)} initial chunks")
            
            # Step 2: Apply reranking in an async context
            print("DEBUG: Applying reranking...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                filtered_nodes = loop.run_until_complete(
                    self._rerank_chunks(retrieved_nodes, conversation_context, customer_message)
                )
            finally:
                loop.close()
            
            if not filtered_nodes:
                print("DEBUG: No relevant chunks found after reranking")
                return {
                    "suggestions": ["No relevant information found in knowledge base for this query."],
                    "knowledge_snippets": []
                }
            
            # Step 3: Create context from filtered chunks
            context_parts = []
            for node in filtered_nodes:
                chunk_text = node.text if hasattr(node, 'text') else str(node)
                context_parts.append(chunk_text)
            
            combined_context = "\n\n".join(context_parts)
            
            # Step 4: Generate response using filtered context
            enhanced_prompt = f"""
            {prompt}
            
            Use the following relevant information from the knowledge base to inform your suggestions:
            
            {combined_context}
            """
            
            print("DEBUG: Generating suggestions with reranked context...")
            response = Settings.llm.complete(enhanced_prompt)
            response_str = str(response)
            
            # Step 5: Use response directly as single suggestion
            final_suggestions = [response_str.strip()]
            
            # Create knowledge snippets from filtered nodes
            knowledge_snippets = []
            for node in filtered_nodes:
                chunk_text = node.text if hasattr(node, 'text') else str(node)
                file_name = "Unknown"
                if hasattr(node, 'metadata') and 'file_name' in node.metadata:
                    file_name = node.metadata['file_name']
                elif hasattr(node, 'node') and hasattr(node.node, 'metadata') and 'file_name' in node.node.metadata:
                    file_name = node.node.metadata['file_name']
                
                snippet = f"📄 **{file_name}**\n{chunk_text}"
                knowledge_snippets.append(snippet)
            
            print(f"DEBUG: Generated 1 suggestion with {len(knowledge_snippets)} reranked snippets")
            
            return {
                "suggestions": final_suggestions,
                "knowledge_snippets": knowledge_snippets
            }
            
        except Exception as e:
            print(f"DEBUG: Exception in _generate_with_reranking: {e}")
            import traceback
            traceback.print_exc()
            return {
                "suggestions": [f"Error in reranked generation: {e}"],
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

    def get_system_status(self) -> Dict:
        """Get the status of system components."""
        txt_files = self._get_knowledge_base_files()
        
        # Check if storage exists
        storage_exists = os.path.exists(self.persist_dir)
        metadata_file = os.path.join(self.persist_dir, 'metadata.json')
        metadata_exists = os.path.exists(metadata_file)
        
        # Get storage info
        storage_info = {}
        if metadata_exists:
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                storage_info = {
                    'cached_files': metadata.get('files', []),
                    'cache_created_at': metadata.get('created_at', 0),
                    'cache_mod_time': metadata.get('modification_time', 0)
                }
            except Exception:
                storage_info = {'error': 'Failed to read metadata'}
        
        return {
            'knowledge_base_loaded': self.index is not None,
            'query_engine_ready': self.query_engine is not None,
            'chat_engine_ready': self.chat_engine is not None,
            'documents_found': len(txt_files) > 0,
            'document_count': len(txt_files),
            'document_files': txt_files,
            'vector_store_type': 'FAISS' if self.use_faiss else 'Default',
            'similarity_top_k': self.similarity_top_k,
            'persist_dir': self.persist_dir,
            'storage_exists': storage_exists,
            'cache_available': metadata_exists,
            'storage_info': storage_info,
            'use_reranker': self.use_reranker,
            'reranker_model': 'gpt-4o-mini' if self.use_reranker else None
        }
