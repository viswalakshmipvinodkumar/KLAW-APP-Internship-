"""
Group Chat implementation for the RAG Chatbot system.
This module contains both RoundRobinGroupChat and SelectorGroupChat implementations.
"""

import asyncio
from typing import List, Dict, Any, Optional, Callable, Awaitable
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents import QueryHandlerAgent, RAGRetrieverAgent
from tools import ChromaDBTool, GeminiTool


class RoundRobinGroupChat:
    """Round Robin Group Chat implementation.
    
    In this implementation, each agent takes turns responding to the query.
    """
    
    def __init__(self, agents: List[Any], max_turns: int = 5):
        """Initialize Round Robin Group Chat.
        
        Args:
            agents: List of agents
            max_turns: Maximum number of turns
        """
        self.agents = agents
        self.max_turns = max_turns
        self.conversation_history = []
        
    async def process_query(self, query: str) -> str:
        """Process a query through the group chat.
        
        Args:
            query: User query
            
        Returns:
            Final response
        """
        # Add the user query to the conversation history
        self.conversation_history.append(HumanMessage(content=query))
        
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
            else:
                # For any other type of agent (e.g., GeminiTool)
                if hasattr(current_agent, 'generate_response'):
                    response = await current_agent.generate_response(
                        query=current_message,
                        system_prompt="You are a helpful AI assistant in a group chat. Respond to the query."
                    )
                else:
                    response = f"Agent {agent_index} cannot process the query."
            
            # Add the agent's response to the conversation history
            self.conversation_history.append(AIMessage(content=response))
            
            # Update the current message for the next turn
            current_message = response
            
            # If this is the last turn, return the final response
            if turn == self.max_turns - 1:
                return response
        
        # This should not be reached, but just in case
        return "No response generated."


class SelectorGroupChat:
    """Selector Group Chat implementation.
    
    In this implementation, a selector agent chooses which agent should respond to the query.
    """
    
    def __init__(self, agents: Dict[str, Any], gemini_tool: GeminiTool):
        """Initialize Selector Group Chat.
        
        Args:
            agents: Dictionary of agent name to agent
            gemini_tool: Gemini tool for the selector
        """
        self.agents = agents
        self.gemini_tool = gemini_tool
        self.conversation_history = []
        
    async def select_agent(self, query: str) -> str:
        """Select an agent to handle the query.
        
        Args:
            query: User query
            
        Returns:
            Name of the selected agent
        """
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
        """Process a query through the group chat.
        
        Args:
            query: User query
            
        Returns:
            Response
        """
        # Add the user query to the conversation history
        self.conversation_history.append(HumanMessage(content=query))
        
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
        self.conversation_history.append(AIMessage(content=response))
        
        return response
