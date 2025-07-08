"""
Knowledge Base Store using Haystack for vector storage and retrieval.
"""

import os
import json
import hashlib
import pickle
from pathlib import Path
from typing import List, Dict, Any
from haystack import Document
from haystack.components.retrievers.in_memory import InMemoryEmbeddingRetriever
from haystack.document_stores.in_memory import InMemoryDocumentStore
from llm_client import get_embedding


class KnowledgeBaseStore:
    """
    Knowledge base store with vector indexing, caching, and retrieval using Haystack.
    """
    
    def __init__(
        self,
        knowledge_base_dir: str,
        cache_dir: str,
        model_name: str = "text-embedding-3-small"
    ):
        """
        Initialize the knowledge base store.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base text files
            cache_dir: Directory for storing cached embeddings and index
            model_name: Embedding model to use
        """
        self.knowledge_base_dir = Path(knowledge_base_dir)
        self.cache_dir = Path(cache_dir) / model_name
        self.model_name = model_name
        
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # File paths for caching
        self.hash_file = self.cache_dir / "files_hash.json"
        self.document_store_file = self.cache_dir / "document_store.pkl"
        
        # Initialize Haystack components
        self.document_store = InMemoryDocumentStore()
        self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
        
        # Initialize the knowledge base
        self._setup_knowledge_base()
    
    def _setup_knowledge_base(self):
        """
        Setup knowledge base by loading from cache or creating new embeddings.
        """
        if not self.knowledge_base_dir.exists():
            raise ValueError(f"Knowledge base directory does not exist: {self.knowledge_base_dir}")
        
        # Calculate current files hash
        current_hash = self._get_files_hash()
        cached_hash = self._get_cached_hash()
        
        # Try to load from cache if hash matches
        if current_hash == cached_hash and self._load_from_cache():
            print("Loaded knowledge base from cache")
            return
        
        # Need to recreate embeddings
        print("Cache miss or files changed, creating embeddings...")
        self._create_embeddings()
        self._save_cache()
        self._save_hash(current_hash)
        print("Knowledge base setup complete")
    
    def _get_files_hash(self) -> str:
        """Calculate hash of all files in knowledge base directory."""
        hash_md5 = hashlib.md5()
        text_files = sorted(self.knowledge_base_dir.glob("*.txt"))
        
        for file_path in text_files:
            hash_md5.update(str(file_path.name).encode())
            with open(file_path, 'rb') as f:
                hash_md5.update(f.read())
        
        return hash_md5.hexdigest()
    
    def _get_cached_hash(self) -> str:
        """Load cached hash or return empty string."""
        if self.hash_file.exists():
            try:
                with open(self.hash_file, 'r') as f:
                    data = json.load(f)
                    return data.get('hash', '')
            except (json.JSONDecodeError, KeyError):
                pass
        return ''
    
    def _save_hash(self, files_hash: str):
        """Save current files hash."""
        with open(self.hash_file, 'w') as f:
            json.dump({'hash': files_hash, 'model': self.model_name}, f)
    
    def _load_from_cache(self) -> bool:
        """Try to load documents from cache and recreate document store."""
        if not self.document_store_file.exists():
            return False
        
        try:
            with open(self.document_store_file, 'rb') as f:
                cached_data = pickle.load(f)
                
            # Recreate document store and load cached documents
            self.document_store = InMemoryDocumentStore()
            cached_documents = cached_data['documents']
            
            # Recreate Document objects from cached data
            documents = []
            for doc_data in cached_documents:
                doc = Document(
                    content=doc_data['content'],
                    meta=doc_data['meta']
                )
                doc.embedding = doc_data['embedding']
                documents.append(doc)
            
            # Write documents to store
            self.document_store.write_documents(documents)
            self.retriever = InMemoryEmbeddingRetriever(document_store=self.document_store)
            
            print(f"Loaded {len(documents)} documents from cache")
            return True
            
        except (pickle.PickleError, KeyError, FileNotFoundError) as e:
            print(f"Cache loading failed: {e}")
            return False
    
    def _save_cache(self):
        """Save document data to cache."""
        try:
            # Extract serializable data from documents
            documents_data = []
            for doc in self.document_store.filter_documents():
                doc_data = {
                    'content': doc.content,
                    'meta': doc.meta,
                    'embedding': doc.embedding
                }
                documents_data.append(doc_data)
            
            cached_data = {
                'documents': documents_data,
                'model': self.model_name
            }
            
            with open(self.document_store_file, 'wb') as f:
                pickle.dump(cached_data, f)
                
        except pickle.PickleError as e:
            print(f"Failed to save cache: {e}")
    
    def _create_embeddings(self):
        """Load documents, create embeddings, and store in document store."""
        # Load all text files
        documents = []
        text_files = list(self.knowledge_base_dir.glob("*.txt"))
        
        print(f"Loading {len(text_files)} documents from {self.knowledge_base_dir}")
        
        for file_path in text_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                
                if content:  # Only add non-empty files
                    document = Document(
                        content=content,
                        meta={
                            "file_name": file_path.name,
                            "file_path": str(file_path)
                        }
                    )
                    documents.append(document)
                    
            except (UnicodeDecodeError, IOError) as e:
                print(f"Error reading {file_path}: {e}")
        
        if not documents:
            print("No documents found in knowledge base directory")
            return
        
        # Create embeddings for documents
        print(f"Creating embeddings for {len(documents)} documents...")
        
        # Get embeddings for all document contents
        doc_contents = [doc.content for doc in documents]
        embeddings = get_embedding(doc_contents, model=self.model_name)
        
        # Set embeddings on documents
        for doc, embedding in zip(documents, embeddings):
            doc.embedding = embedding
        
        # Clear existing documents and add new ones
        self.document_store.delete_documents([])
        self.document_store.write_documents(documents)
        
        print(f"Created embeddings for {len(documents)} documents")
    
    def retrieve_snippets(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant snippets for a given query.
        
        Args:
            query: Query string to search for
            top_k: Number of top snippets to return
            
        Returns:
            List of snippet dictionaries with content and metadata
        """
        # Create query embedding
        query_embedding = get_embedding(query, model=self.model_name)
        
        # Retrieve documents
        result = self.retriever.run(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        # Format and return results
        snippets = []
        for doc in result["documents"]:
            snippet = {
                "content": doc.content,
                "score": getattr(doc, 'score', None),
                "file_name": doc.meta.get("file_name", "unknown"),
                "file_path": doc.meta.get("file_path", "unknown")
            }
            snippets.append(snippet)
        
        return snippets 