"""
Data loader for the RAG Chatbot system.
This module contains functions to load and process data for the RAG system.
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


async def load_text_file(file_path: str) -> str:
    """Load text from a file asynchronously.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Text content
    """
    async def _read_file():
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    return await asyncio.to_thread(_read_file)


async def load_documents_from_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
    """Load documents from text.
    
    Args:
        text: Text content
        chunk_size: Size of each chunk
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of Document objects
    """
    # Create a text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False
    )
    
    # Split the text into chunks synchronously to avoid coroutine issues
    documents = text_splitter.create_documents([text])
    return documents


async def load_documents_from_directory(directory_path: str, extensions: List[str] = ['.txt']) -> List[Document]:
    """Load documents from a directory.
    
    Args:
        directory_path: Path to the directory
        extensions: List of file extensions to include
        
    Returns:
        List of Document objects
    """
    documents = []
    
    # Get all files in the directory with the specified extensions
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # Load the text from the file
                text = await load_text_file(file_path)
                
                # Split the text into documents
                docs = await load_documents_from_text(text)
                
                # Add metadata to the documents
                for doc in docs:
                    doc.metadata['source'] = file_path
                
                documents.extend(docs)
    
    return documents
