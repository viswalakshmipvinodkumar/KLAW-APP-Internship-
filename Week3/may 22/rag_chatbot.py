"""
FAQ Chatbot with RAG (Retrieval-Augmented Generation)
This is a simplified implementation that uses Gemini API for both retrieval and generation.
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

# Configure Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("Please set your GEMINI_API_KEY in the .env file")
    
genai.configure(api_key=API_KEY)

# Initialize Gemini model
model = genai.GenerativeModel('gemini-1.5-pro')

# Document class for storing text content
class Document:
    """Simple document class to store text content and metadata."""
    
    def __init__(self, content: str, metadata: Dict[str, Any] = None):
        self.page_content = content
        self.metadata = metadata or {}


# ChromaDB Tool (Simplified version using embeddings)
class ChromaDBTool:
    """Simplified tool for document retrieval using semantic similarity."""
    
    def __init__(self, collection_name: str = "faq_collection"):
        """Initialize the document retrieval tool."""
        self.collection_name = collection_name
        self.documents = []
        self.embeddings_cache = {}
        
    async def create_db_from_documents(self, documents: List[Document]) -> None:
        """Store documents for later retrieval."""
        self.documents = documents
        print(f"Stored {len(documents)} documents for retrieval in collection '{self.collection_name}'")
        
    async def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using a simple hash-based approach."""
        # Check cache first
        if text in self.embeddings_cache:
            return self.embeddings_cache[text]
            
        # Generate embedding using a simple hash-based approach
        try:
            # Use a simple hash-based approach for demonstration
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
        """Find documents most relevant to the query using semantic similarity."""
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
        """Calculate cosine similarity between two vectors."""
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


# Gemini Tool
class GeminiTool:
    """Tool for interacting with Gemini API."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """Initialize Gemini tool."""
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
    async def generate_response(
        self, 
        query: str, 
        context: Optional[str] = None, 
        system_prompt: Optional[str] = None
    ) -> str:
        """Generate a response using Gemini API."""
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
        """Classify a query as AI-related or general."""
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


# RAG Retriever Agent
class RAGRetrieverAgent:
    """Agent for retrieving relevant information using RAG."""
    
    def __init__(self, chroma_tool: ChromaDBTool, gemini_tool: GeminiTool):
        """Initialize RAG Retriever agent."""
        self.chroma_tool = chroma_tool
        self.gemini_tool = gemini_tool
        
    async def retrieve_and_generate(self, query: str) -> str:
        """Retrieve relevant documents and generate a response."""
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


# Query Handler Agent
class QueryHandlerAgent:
    """Agent for handling user queries."""
    
    def __init__(self, rag_retriever: RAGRetrieverAgent, gemini_tool: GeminiTool):
        """Initialize Query Handler agent."""
        self.rag_retriever = rag_retriever
        self.gemini_tool = gemini_tool
        
    async def process_query(self, query: str) -> str:
        """Process a user query."""
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


# SelectorGroupChat
class SelectorGroupChat:
    """Selector Group Chat implementation."""
    
    def __init__(self, agents: Dict[str, Any], gemini_tool: GeminiTool):
        """Initialize Selector Group Chat."""
        self.agents = agents
        self.gemini_tool = gemini_tool
        self.conversation_history = []
        
    async def select_agent(self, query: str) -> str:
        """Select an agent to handle the query."""
        agent_names = list(self.agents.keys())
        agent_descriptions = {
            "query_handler": "Handles general queries and decides how to route them",
            "rag_retriever": "Retrieves information from a knowledge base and generates responses"
        }
        
        system_prompt = f"""You are a selector agent. Your job is to select the most appropriate agent to handle the user's query.
        Available agents:
        {', '.join([f"{name}: {agent_descriptions.get(name, 'No description')}" for name in agent_names])}
        
        Return only the name of the agent that should handle the query. Choose from: {', '.join(agent_names)}.
        """
        
        selected_agent = await self.gemini_tool.generate_response(
            query=query,
            system_prompt=system_prompt
        )
        
        # Clean up the response to match an agent name
        for agent_name in agent_names:
            if agent_name.lower() in selected_agent.lower():
                return agent_name
        
        # Default to query_handler if no match is found
        return "query_handler"
    
    async def process_query(self, query: str) -> str:
        """Process a query through the group chat."""
        # Add the user query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Select an agent to handle the query
        selected_agent_name = await self.select_agent(query)
        
        # Get the selected agent
        selected_agent = self.agents.get(selected_agent_name)
        
        if not selected_agent:
            response = "No suitable agent found to handle the query."
        else:
            # Process the query with the selected agent
            if isinstance(selected_agent, QueryHandlerAgent):
                response = await selected_agent.process_query(query)
            elif isinstance(selected_agent, RAGRetrieverAgent):
                response = await selected_agent.retrieve_and_generate(query)
            else:
                # For any other type of agent
                if hasattr(selected_agent, 'generate_response'):
                    response = await selected_agent.generate_response(
                        query=query,
                        system_prompt="You are a helpful AI assistant. Respond to the query."
                    )
                else:
                    response = f"Agent {selected_agent_name} cannot process the query."
        
        # Add the response to the conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        return response


# RoundRobinGroupChat
class RoundRobinGroupChat:
    """Round Robin Group Chat implementation."""
    
    def __init__(self, agents: List[Any], max_turns: int = 5):
        """Initialize Round Robin Group Chat."""
        self.agents = agents
        self.max_turns = max_turns
        self.conversation_history = []
        
    async def process_query(self, query: str) -> str:
        """Process a query through the group chat."""
        # Add the user query to the conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Initialize the current message
        current_message = query
        
        # Process the query through each agent in a round-robin fashion
        for turn in range(self.max_turns):
            # Get the current agent (cycling through the list)
            agent_index = turn % len(self.agents)
            current_agent = self.agents[agent_index]
            
            # Process the current message with the current agent
            if isinstance(current_agent, QueryHandlerAgent):
                response = await current_agent.process_query(current_message)
            elif isinstance(current_agent, RAGRetrieverAgent):
                response = await current_agent.retrieve_and_generate(current_message)
            elif isinstance(current_agent, GeminiTool):
                response = await current_agent.generate_response(
                    query=current_message,
                    system_prompt="You are a helpful AI assistant in a group chat. Respond to the query."
                )
            else:
                response = f"Agent {agent_index} cannot process the query."
            
            # Add the agent's response to the conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
            
            # Update the current message for the next turn
            current_message = response
            
            # If this is the last turn, return the final response
            if turn == self.max_turns - 1:
                return response
        
        # This should not be reached, but just in case
        return "No response generated."


# Data loading functions
async def load_text_file(file_path: str) -> str:
    """Load text from a file asynchronously."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


async def load_documents_from_directory(directory_path: str, extensions: List[str] = ['.txt']) -> List[Document]:
    """Load documents from a directory."""
    documents = []
    
    # Get all files in the directory with the specified extensions
    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in extensions):
                file_path = os.path.join(root, file)
                
                # Load the text from the file
                text = await load_text_file(file_path)
                
                # Create a document with metadata
                doc = Document(
                    content=text,
                    metadata={"source": file_path}
                )
                
                documents.append(doc)
                print(f"Loaded document from {file_path}")
    
    return documents


# Main application
async def initialize_system():
    """Initialize the RAG Chatbot system."""
    print("Initializing the RAG Chatbot system...")
    
    # Initialize tools
    print("Initializing tools...")
    chroma_tool = ChromaDBTool()
    gemini_tool = GeminiTool()
    
    # Load documents
    print("Loading documents...")
    documents = await load_documents_from_directory("./data")
    
    # Create the vector database
    print("Creating vector database...")
    await chroma_tool.create_db_from_documents(documents)
    
    # Initialize agents
    print("Initializing agents...")
    rag_retriever = RAGRetrieverAgent(chroma_tool, gemini_tool)
    query_handler = QueryHandlerAgent(rag_retriever, gemini_tool)
    
    # Choose the group chat implementation (SelectorGroupChat for more intelligent routing)
    print("Initializing SelectorGroupChat...")
    group_chat = SelectorGroupChat(
        agents={
            "query_handler": query_handler,
            "rag_retriever": rag_retriever
        },
        gemini_tool=gemini_tool
    )
    
    print("RAG Chatbot system initialized!")
    return group_chat


async def main():
    """Main function."""
    # Initialize the system
    group_chat = await initialize_system()
    
    print("\nWelcome to the FAQ Chatbot with RAG!")
    print("Type 'exit' to quit.")
    
    # Main interaction loop
    while True:
        # Get user input
        user_query = input("\nYou: ")
        
        # Check if the user wants to exit
        if user_query.lower() == 'exit':
            print("Goodbye!")
            break
        
        # Process the query
        print("\nProcessing your query...")
        try:
            response = await group_chat.process_query(user_query)
            print(f"\nChatbot: {response}")
        except Exception as e:
            print(f"\nError: {str(e)}")


if __name__ == "__main__":
    # Run the main function
    asyncio.run(main())
