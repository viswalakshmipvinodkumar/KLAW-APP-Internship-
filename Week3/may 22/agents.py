"""
Agents for the RAG Chatbot system.
This module contains the Query Handler and RAG Retriever agents.
"""

import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain.agents import AgentExecutor
from langchain.agents.format_scratchpad import format_to_openai_function_messages
from langchain.agents.output_parsers import OpenAIFunctionsAgentOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool, StructuredTool, tool
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from tools import ChromaDBTool, GeminiTool


class RAGRetrieverAgent:
    """Agent for retrieving relevant information using RAG."""
    
    def __init__(self, chroma_tool: ChromaDBTool, gemini_tool: GeminiTool):
        """Initialize RAG Retriever agent.
        
        Args:
            chroma_tool: ChromaDB tool
            gemini_tool: Gemini tool
        """
        self.chroma_tool = chroma_tool
        self.gemini_tool = gemini_tool
        
    async def retrieve_and_generate(self, query: str) -> str:
        """Retrieve relevant documents and generate a response.
        
        Args:
            query: User query
            
        Returns:
            Generated response
        """
        # Retrieve relevant documents
        docs = await self.chroma_tool.similarity_search(query)
        
        # Extract and combine document content
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Generate response using Gemini with the retrieved context
        system_prompt = """You are a helpful AI assistant specialized in answering questions about AI topics.
        Use the provided context to answer the user's question accurately.
        If the answer is not in the context, say that you don't have enough information to answer the question.
        Always provide comprehensive, accurate, and helpful responses."""
        
        response = await self.gemini_tool.generate_response(
            query=query,
            context=context,
            system_prompt=system_prompt
        )
        
        return response


class QueryHandlerAgent:
    """Agent for handling user queries."""
    
    def __init__(self, rag_retriever: RAGRetrieverAgent, gemini_tool: GeminiTool):
        """Initialize Query Handler agent.
        
        Args:
            rag_retriever: RAG Retriever agent
            gemini_tool: Gemini tool
        """
        self.rag_retriever = rag_retriever
        self.gemini_tool = gemini_tool
        
    async def process_query(self, query: str) -> str:
        """Process a user query.
        
        Args:
            query: User query
            
        Returns:
            Response to the query
        """
        # First, determine if the query is about AI topics or general
        classification = await self.gemini_tool.classify_query(query)
        
        # Use RAG for AI-related queries, direct Gemini for general queries
        if classification == "AI":
            return await self.rag_retriever.retrieve_and_generate(query)
        else:
            system_prompt = """You are a helpful AI assistant. Answer the user's question to the best of your ability.
            If you don't know the answer, say so honestly."""
            
            return await self.gemini_tool.generate_response(
                query=query,
                system_prompt=system_prompt
            )
