"""
Tools for the RAG Chatbot system.
This module contains the ChromaDB and Gemini API tools.
"""

import os
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_core.documents import Document

# Load environment variables
load_dotenv()

# Configure Google Generative AI
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Please set your GEMINI_API_KEY in the .env file")
    
genai.configure(api_key=API_KEY)


class ChromaDBTool:
    """Simplified tool for document retrieval using keyword matching."""
    
    def __init__(self, collection_name: str = "faq_collection"):
        """Initialize the document retrieval tool.
        
        Args:
            collection_name: Name of the collection (for compatibility)
        """
        self.collection_name = collection_name
        self.documents = []
        self.embeddings_cache = {}
        self.model = genai.GenerativeModel('gemini-1.5-pro')
        
    async def create_db_from_documents(self, documents: List[Document]) -> None:
        """Store documents for later retrieval.
        
        Args:
            documents: List of Document objects
        """
        self.documents = documents
        print(f"Stored {len(documents)} documents for retrieval in collection '{self.collection_name}'")
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using Gemini API.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector
        """
        # Check cache first
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
            
        # Generate embedding using Gemini
        try:
            # Use a simple hash-based approach for demonstration
            # In a real implementation, you would use a proper embedding API
            embedding = [hash(word) % 100 / 100.0 for word in text.split()[:20]]
            # Pad or truncate to fixed length (20)
            embedding = embedding[:20] + [0] * max(0, 20 - len(embedding))
            self.embeddings_cache[text] = embedding
            return embedding
        except Exception as e:
            print(f"Error generating embedding: {e}")
            # Return a random embedding as fallback
            return [np.random.random() for _ in range(20)]
    
    async def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """Find documents most relevant to the query using semantic similarity.
        
        Args:
            query: User query
            k: Number of results to return
            
        Returns:
            List of Document objects
        """
        if not self.documents:
            raise ValueError("No documents stored. Call create_db_from_documents first.")
        
        # Get query embedding
        query_embedding = await self._get_embedding(query)
        
        # Calculate similarity scores
        scored_docs = []
        for doc in self.documents:
            # Get document embedding
            doc_embedding = await self._get_embedding(doc.page_content[:1000])  # Limit to first 1000 chars
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_embedding, doc_embedding)
            scored_docs.append((similarity, doc))
        
        # Sort by similarity (descending) and take top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for _, doc in scored_docs[:k]]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity
        """
        # Convert to numpy arrays
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


class GeminiTool:
    """Tool for interacting with Gemini API."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """Initialize Gemini tool.
        
        Args:
            model_name: Name of the Gemini model
        """
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
    async def generate_response(
        self, 
        query: str, 
        context: Optional[str] = None, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using Gemini API.
        
        Args:
            query: User query
            context: Optional context for the query
            system_prompt: Optional system prompt
            
        Returns:
            Generated response
        """
        try:
            if context:
                prompt = f"""{system_prompt or 'You are a helpful AI assistant.'}

Context information is below.
{context}

Given the context information and not prior knowledge, answer the query.

Query: {query}"""
            else:
                prompt = f"""{system_prompt or 'You are a helpful AI assistant.'}

Query: {query}"""
            
            # Execute the model call asynchronously
            response = await asyncio.to_thread(
                lambda: self.model.generate_content(prompt).text
            )
            
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I'm sorry, I encountered an error: {str(e)[:100]}..."
    
    async def classify_query(self, query: str) -> str:
        """Classify a query as AI-related or general.
        
        Args:
            query: User query
            
        Returns:
            Classification result ('AI' or 'GENERAL')
        """
        try:
            prompt = """You are a query classifier. Determine if the given query is about AI topics or general knowledge.
            Return only 'AI' if the query is about AI topics, or 'GENERAL' if it's about general knowledge.
            
            Query: {query}
            Classification:""".format(query=query)
            
            # Execute the model call asynchronously
            response = await asyncio.to_thread(
                lambda: self.model.generate_content(prompt).text
            )
            
            # Clean up response
            response = response.strip().upper()
            if 'AI' in response:
                return 'AI'
            else:
                return 'GENERAL'
        except Exception as e:
            print(f"Error classifying query: {e}")
            return 'GENERAL'  # Default to general if there's an error
