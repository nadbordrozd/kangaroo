"""
Contact Center Assistant using RAG (Retrieval-Augmented Generation)

This module contains the Assistant class that provides LLM-powered suggestions
and summaries for contact center agents based on a knowledge base.
"""

from typing import List, Dict, Any, Tuple, Optional
import asyncio
from llm_client import get_embedding, get_completion, get_completion_async
from knowledge_base_store import KnowledgeBaseStore


async def rerank_snippet(snippet: str, conversation: str, model: str) -> bool:
    messages = [
        {
            "role": "system", 
            "content": "You are a relevance judge for a customer service knowledge base system. Respond with only 'yes' or 'no'."
        },
        {
            "role": "user", 
            "content": f"Conversation:\n{conversation}\n\nKnowledge snippet:\n{snippet}\n\nDoes this snippet contain information that would help a customer service agent respond appropriately to the conversation?"
        }
    ]
    
    response = await get_completion_async(messages, model=model, temperature=0.1)
    return response.lower().strip().startswith('yes')


async def get_summary(conversation: str, model: str) -> str:
    summary_messages = [
        {
            "role": "system", 
            "content": """You are a helpful assistant that creates concise summaries of customer service conversations. 
            List customer's issues and whether or not they have been resolved.
            Return only the summary, no other text.
            """
        },
        {
            "role": "user", 
            "content": f"{conversation}"
        }
    ]
    
    return await get_completion_async(summary_messages, model=model, temperature=0.3)


async def get_suggestions(knowledge_snippets: str, conversation: str, model: str) -> str:
    suggestion_messages = [
        {
            "role": "system", 
            "content": "You are a helpful assistant that suggests an appropriate response for a customer service agent. Provide one practical, helpful suggestion based on the conversation and available knowledge."
        },
        {
            "role": "user", 
            "content": f"Conversation:\n{conversation}{knowledge_snippets}\n\nPlease suggest one appropriate response for the agent."
        }
    ]
    
    suggestion = await get_completion_async(suggestion_messages, model=model, temperature=0.7)
    return suggestion.strip()


class Assistant:
    """
    LLM-powered contact center assistant that provides agent suggestions
    and conversation summaries using RAG.
    """
    
    def __init__(
        self, 
        knowledge_base_dir: str = "knowledge_base",
        cache_dir: str = "storage",
        embedding_model: str = "text-embedding-3-small",
        reranker_model: str = "gpt-4.1",
        generator_model: str = "gpt-4.1",
        max_conversation_messages: int = 10,
        max_retrieved_snippets: int = 5
    ):
        """
        Initialize the Assistant with configurable parameters.
        
        Args:
            knowledge_base_dir: Directory containing knowledge base text files
            cache_dir: Directory for caching embeddings and vector database
            embedding_model: Embedding model to use
            reranker_model: Model to use for snippet reranking
            generator_model: Model to use for suggestions and summaries
            max_conversation_messages: Number of recent messages to consider
            max_retrieved_snippets: Number of top snippets to retrieve
        """
        self.max_conversation_messages = max_conversation_messages
        self.max_retrieved_snippets = max_retrieved_snippets
        self.reranker_model = reranker_model
        self.generator_model = generator_model
        self.embedding_model = embedding_model
        
        # Initialize knowledge base store
        self.knowledge_store = KnowledgeBaseStore(
            knowledge_base_dir=knowledge_base_dir,
            cache_dir=cache_dir,
            model_name=embedding_model
        )
    
    async def _filter_snippets_async(
        self, 
        snippets: List[Dict[str, Any]], 
        conversation: List[Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """
        Filter snippets using async LLM reranker calls.
        
        Args:
            snippets: Retrieved snippets
            conversation: Full conversation history
            
        Returns:
            Filtered relevant snippets
        """
        # Prepare conversation context
        conversation_context = self._concatenate_recent_messages(conversation)
        
        # Run reranking tasks concurrently
        rerank_tasks = [
            rerank_snippet(snippet["content"], conversation_context, self.reranker_model) 
            for snippet in snippets
        ]
        
        # Wait for all reranking to complete
        relevance_results = await asyncio.gather(*rerank_tasks)
        
        # Filter snippets based on relevance
        relevant_snippets = []
        for snippet, is_relevant in zip(snippets, relevance_results):
            if is_relevant:
                relevant_snippets.append(snippet)
        
        return relevant_snippets
    
    async def get_suggestions(
        self, 
        messages: List[Dict[str, str]]
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Get agent suggestion based on conversation history.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
                     (e.g., [{'role': 'customer', 'content': '...'}, ...])
        
        Returns:
            Tuple of (suggestion_text, knowledge_base_snippets_used)
        """
        print(f"Getting suggestions for {messages}")
        # Concatenate recent messages for context
        recent_context = self._concatenate_recent_messages(messages)
        
        # Retrieve relevant snippets from knowledge base
        retrieved_snippets = self.knowledge_store.retrieve_snippets(
            query=recent_context, 
            top_k=self.max_retrieved_snippets
        )
        
        # Filter snippets using async LLM reranker
        relevant_snippets = await self._filter_snippets_async(retrieved_snippets, messages)
        
        # Prepare knowledge base context string
        knowledge_snippets = ""
        if relevant_snippets:
            knowledge_snippets = "\n\nRelevant knowledge base information:\n"
            for snippet in relevant_snippets:
                knowledge_snippets += f"- {snippet['content']}\n"
        
        # Use standalone suggestions function
        suggestion = await get_suggestions(knowledge_snippets, recent_context, self.generator_model)
        print(f"Suggestion: {suggestion}")
        return [suggestion], relevant_snippets
    
    async def get_summary(self, messages: List[Dict[str, str]]) -> str:
        """
        Generate a summary of the conversation so far.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
        
        Returns:
            Summary of the conversation
        """
        # Prepare conversation context
        context = self._concatenate_recent_messages(messages)
        
        # Use standalone summary function
        return await get_summary(context, self.generator_model)
    
    def _concatenate_recent_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Concatenate the last N messages for context.
        
        Args:
            messages: Full message history
            
        Returns:
            Concatenated recent messages as string
        """
        # Get the last N messages
        recent_messages = messages[-self.max_conversation_messages:]
        
        # Concatenate messages with role information
        context_parts = []
        for msg in recent_messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            context_parts.append(f"{role}: {content}")
        
        return "\n".join(context_parts)
    
 