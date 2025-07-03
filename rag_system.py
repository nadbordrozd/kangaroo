import os
import json
import time
import asyncio
import logging
from typing import List, Dict, Optional
from dotenv import load_dotenv
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SentenceSplitter


# Load environment variables
load_dotenv()

# Configure logger
logger = logging.getLogger('RAGSystem')
logger.setLevel(logging.DEBUG)

# Create console handler for stdout output
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)

# Add handler to logger (only if not already added to avoid duplicates)
if not logger.handlers:
    logger.addHandler(console_handler)

class RAGSystem:
    def __init__(self, knowledge_base_path: str = 'knowledge_base', similarity_top_k: int = 5, persist_dir: str = './storage', use_reranker: bool = False, conversation_context_length: int = 5):
        self.knowledge_base_path = knowledge_base_path
        self.similarity_top_k = similarity_top_k
        self.persist_dir = persist_dir
        self.use_reranker = use_reranker
        self.conversation_context_length = conversation_context_length
        self.index: Optional[VectorStoreIndex] = None
        self.reranker_llm = None
        self._setup_system()

    def _setup_system(self):
        """Setup the RAG system with OpenAI models and LlamaIndex components."""
        # Configure global settings
        openai_api_key = os.getenv('OPENAI_API_KEY')
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables.")
        
        # Set up models using Settings (LlamaIndex's global configuration)git s
        Settings.llm = OpenAI(model="gpt-4.1", api_key=openai_api_key, temperature=0.1)
        Settings.embed_model = OpenAIEmbedding(
            model="text-embedding-3-large", 
            api_key=openai_api_key
        )
        
        # Set up reranker LLM if enabled
        if self.use_reranker:
            self.reranker_llm = OpenAI(model="gpt-4o-mini", api_key=openai_api_key, temperature=0.0)
            logger.debug("Reranker LLM (GPT-4o-mini) configured")
        else:
            logger.debug("Reranker disabled")
        
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
        logger.debug(f"Checking if index should be rebuilt: {self.persist_dir}")
        if not os.path.exists(self.persist_dir):
            logger.debug("Storage directory doesn't exist, will create new index")
            return True
        
        # Check if index files exist
        index_files = ['docstore.json', 'index_store.json', 'default__vector_store.json']
        for index_file in index_files:
            if not os.path.exists(os.path.join(self.persist_dir, index_file)):
                logger.debug(f"Index file {index_file} missing, will rebuild index")
                return True
        
        # Check if metadata file exists
        metadata_file = os.path.join(self.persist_dir, 'metadata.json')
        if not os.path.exists(metadata_file):
            logger.debug("Metadata file missing, will rebuild index")
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
                logger.debug("Knowledge base files changed, will rebuild index")
                return True
            
            if current_mod_time > stored_mod_time:
                logger.debug("Knowledge base files modified, will rebuild index")
                return True
            
            # Rebuild if configuration changed (though this doesn't affect cached embeddings)
            if stored_similarity_top_k != self.similarity_top_k:
                logger.debug("similarity_top_k changed, will rebuild index")
                return True
            
            if stored_use_reranker != self.use_reranker:
                logger.debug("reranker configuration changed, will rebuild index")
                return True
            
            logger.debug("Index is up to date, will load from storage")
            return False
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.debug(f"Error reading metadata: {e}, will rebuild index")
            return True
        
        logger.warning('Index is up to date, will load from storage')
        return False

    def _save_metadata(self, files: List[str]):
        """Save metadata about the current index."""
        os.makedirs(self.persist_dir, exist_ok=True)
        metadata = {
            'files': files,
            'modification_time': self._get_files_modification_time(files),
            'created_at': time.time(),
            'similarity_top_k': self.similarity_top_k,
            'use_reranker': self.use_reranker,
            'conversation_context_length': self.conversation_context_length
        }
        
        metadata_file = os.path.join(self.persist_dir, 'metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.debug(f"Saved metadata to {metadata_file}")

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
            logger.debug(f"Error in relevance check: {e}, defaulting to relevant")
            return True  # Default to relevant if error occurs

    async def _rerank_chunks(self, chunks: List, conversation_context: str, customer_message: str) -> List:
        """Filter chunks based on relevance using parallel LLM calls."""
        if not self.use_reranker or not chunks:
            return chunks
        
        logger.debug(f"Reranking {len(chunks)} chunks for relevance...")
        
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
                logger.debug("Chunk relevant - keeping")
            else:
                logger.debug("Chunk not relevant - filtering out")
        
        logger.debug(f"Reranking complete: {len(relevant_chunks)}/{len(chunks)} chunks kept")
        return relevant_chunks

    def clear_cache(self):
        """Clear the cached embeddings and force rebuild on next load."""
        import shutil
        if os.path.exists(self.persist_dir):
            shutil.rmtree(self.persist_dir)
            logger.debug(f"Cleared cache directory: {self.persist_dir}")
        else:
            logger.debug(f"Cache directory doesn't exist: {self.persist_dir}")



    def _load_knowledge_base(self):
        """Load and index the knowledge base documents with disk caching."""
        logger.debug('loading knowledge base')
        # Create directory if it doesn't exist
        if not os.path.exists(self.knowledge_base_path):
            os.makedirs(self.knowledge_base_path)
            logger.info(f"Created {self.knowledge_base_path} directory. Please add your text files there.")
            return
        
        # Check for documents
        txt_files = self._get_knowledge_base_files()
        if not txt_files:
            logger.info("No .txt files found in knowledge_base directory.")
            return
        try:
            # Check if we can load from existing storage
            if not self._should_rebuild_index():
                logger.debug("Loading index from storage...")
                try:
                    # Load from storage
                    storage_context = StorageContext.from_defaults(persist_dir=self.persist_dir)
                    self.index = load_index_from_storage(storage_context)
                    logger.debug(f"Successfully loaded index from {self.persist_dir}")
                    
                except Exception as e:
                    logger.debug(f"Failed to load from storage: {e}, will rebuild")
                    # Fall through to rebuild
                    pass
            # If we don't have an index, create it
            if self.index is None:
                logger.debug("Creating new index...")
                
                # Load documents using SimpleDirectoryReader
                documents = SimpleDirectoryReader(
                    input_dir=self.knowledge_base_path,
                    required_exts=['.txt']
                ).load_data()
                
                if not documents:
                    logger.info("No documents loaded from knowledge base.")
                    return
                
                # Create index with default vector store
                logger.debug("Creating index with default vector store")
                self.index = VectorStoreIndex.from_documents(documents)
                
                # Persist the index to storage
                logger.debug(f"Saving index to {self.persist_dir}...")
                self.index.storage_context.persist(persist_dir=self.persist_dir)
                
                # Save metadata
                self._save_metadata(txt_files)
                
                logger.debug("Index saved to storage successfully")
                logger.info(f"Knowledge base indexed successfully from {len(txt_files)} files.")
                logger.debug(f"Total documents processed: {len(documents)}")
            else:
                logger.debug(f"Using cached index for {len(txt_files)} files")
            
            logger.debug(f"Index ready for retrieval with similarity_top_k={self.similarity_top_k}")
            
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            logger.debug(f"Full exception details: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()



    def get_suggestions(self, message: str, conversation_context: List[Dict], sender: str) -> Dict[str, List[str]]:
        """Generate AI suggestions and retrieve relevant knowledge base snippets for customer messages."""
        logger.debug("get_suggestions called")
        logger.debug(f"message: '{message}'")
        logger.debug(f"sender: '{sender}'")
        logger.debug(f"conversation_context: {conversation_context}")
        
        if not self.index:
            logger.debug("Index not available")
            return {
                "suggestions": ["Knowledge base is not loaded. Please add documents to the knowledge_base folder."],
                "knowledge_snippets": []
            }
        
        try:
            # Only process customer messages - agent messages are handled at Flask level
            if sender == 'customer':
                logger.debug("Customer message - Using RAG to generate agent response suggestions")
                result = self._generate_customer_suggestions(message, conversation_context)
            else:
                logger.debug(f"Non-customer message blocked (sender: {sender})")
                result = {
                    "suggestions": ["Only customer messages are processed by the AI system"],
                    "knowledge_snippets": []
                }
            
            logger.debug(f"Final result: {result}")
            return result
            
        except Exception as e:
            logger.debug(f"Exception in get_suggestions: {e}")
            return {
                "suggestions": [f"Error generating suggestions: {e}"],
                "knowledge_snippets": []
            }

    def _generate_customer_suggestions(self, message: str, conversation_context: List[Dict]) -> Dict[str, List[str]]:
        """Generate suggestions for customer messages using retriever + optional reranking + manual LLM."""
        logger.debug("_generate_customer_suggestions called - using retriever approach")
        
        # Build context from conversation history
        conversation_context = self._build_conversation_context(conversation_context)
        logger.debug(f"Built context: '{conversation_context}'")
        
        try:
            # Step 1: Retrieve raw chunks from vector store
            logger.debug("Retrieving chunks from vector store...")
            retriever = self.index.as_retriever(similarity_top_k=self.similarity_top_k)
            retrieved_nodes = retriever.retrieve(conversation_context)
            logger.debug(f"Retrieved {len(retrieved_nodes)} initial chunks")
            
            # Step 2: Apply optional reranking
            if self.use_reranker:
                logger.debug("Applying reranking...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    filtered_nodes = loop.run_until_complete(
                        self._rerank_chunks(retrieved_nodes, conversation_context, message)
                    )
                finally:
                    loop.close()
                
                if not filtered_nodes:
                    logger.debug("No relevant chunks found after reranking")
                    return {
                        "suggestions": ["No relevant information found in knowledge base for this query."],
                        "knowledge_snippets": []
                    }
                final_nodes = filtered_nodes
                logger.debug(f"After reranking: {len(final_nodes)} chunks")
            else:
                logger.debug("Skipping reranking - using all retrieved chunks")
                final_nodes = retrieved_nodes
            
            # Step 3: Create context from final chunks
            context_parts = []
            for node in final_nodes:
                chunk_text = node.text if hasattr(node, 'text') else str(node)
                context_parts.append(chunk_text)
            
            combined_context = "\n\n".join(context_parts)
            

            # Step 4: Generate response using context
            enhanced_prompt = f"""You are a customer service agent chatting to a customer. 
Write a helpful response to the customer's last message:

{conversation_context}

based on the following sources of information:
            
{combined_context}

------------------------------------
If there is no information in the sources of information that is relevant to the customer's message, 
just say "I'm sorry, I don't have any information on that topic."

------------------------------------
            """
            response = Settings.llm.complete(enhanced_prompt)
            response_str = str(response)
            
            # Step 5: Use response directly as single suggestion
            final_suggestions = [response_str.strip()]
            
            # Step 6: Create knowledge snippets from final nodes
            
            knowledge_snippets = [f"📄 **{node.metadata['file_name']}**\n{node.text}" for node in final_nodes]
            
            return {
                "suggestions": final_suggestions,
                "knowledge_snippets": knowledge_snippets
            }
            
        except Exception as e:
            logger.debug(f"Exception in _generate_customer_suggestions: {e}")
            import traceback
            traceback.print_exc()
            return {
                "suggestions": [f"Error generating customer suggestions: {e}"],
                "knowledge_snippets": []
            }

    def _build_conversation_context(self, conversation_context: List[Dict]) -> str:
        """Build a string representation of the conversation context using the last N messages."""
        logger.debug(f"_build_conversation_context called with {len(conversation_context)} messages, using last {self.conversation_context_length} for context")
        
        if not conversation_context:
            logger.debug("No conversation context provided")
            return "No previous conversation context."
        
        context_parts = []
        for i, msg in enumerate(conversation_context[-self.conversation_context_length:]):
            sender = msg.get('sender', 'Unknown')
            message = msg.get('message', '')
            context_part = f"{sender}: {message}"
            context_parts.append(context_part)
            logger.debug(f"Context part {i}: '{context_part}'")
        
        final_context = " | ".join(context_parts)
        logger.debug(f"Final context string: '{final_context}'")
        return final_context



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
            'retriever_ready': self.index is not None,
            'documents_found': len(txt_files) > 0,
            'document_count': len(txt_files),
            'document_files': txt_files,
            'vector_store_type': 'Default',
            'similarity_top_k': self.similarity_top_k,
            'persist_dir': self.persist_dir,
            'storage_exists': storage_exists,
            'cache_available': metadata_exists,
            'storage_info': storage_info,
            'use_reranker': self.use_reranker,
            'reranker_model': 'gpt-4o-mini' if self.use_reranker else None,
            'conversation_context_length': self.conversation_context_length
        }
