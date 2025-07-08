"""
LLM Client for embedding and completion operations using OpenAI API.
"""

from typing import List, Dict
import os
from openai import OpenAI, AsyncOpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize OpenAI clients (both sync and async)
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

async_client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def get_embedding(
    input: str | List[str], 
    model: str = "text-embedding-3-small"
) -> List[float]:
    """
    Get embedding for the given text using OpenAI embedding API.
    
    Args:
        input: Text string or list of text strings to embed
        model: Embedding model to use
        
    Returns:
        Embedding vector for single input, or list of embedding vectors for multiple inputs
    """
    response = client.embeddings.create(
        input=input,
        model=model
    )
    
    # Return single embedding if input was a string, otherwise return list of embeddings
    if isinstance(input, str):
        return response.data[0].embedding
    else:
        return [item.embedding for item in response.data]


def get_completion(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> str:
    """
    Get completion from OpenAI chat completions API (synchronous).
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use for completion
        temperature: Sampling temperature (0-2)
        
    Returns:
        Generated completion text
    """
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content


async def get_completion_async(
    messages: List[Dict[str, str]],
    model: str = "gpt-4o",
    temperature: float = 0.7
) -> str:
    """
    Get completion from OpenAI chat completions API (asynchronous).
    
    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        model: Model to use for completion
        temperature: Sampling temperature (0-2)
        
    Returns:
        Generated completion text
    """
    response = await async_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature
    )
    
    return response.choices[0].message.content 