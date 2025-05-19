"""
Group chat implementation for coordinating agents.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union, Callable, Any

class Agent:
    """Base class for agents in the group chat."""
    
    def __init__(self, name: str):
        """
        Initialize an agent.
        
        Args:
            name: The name of the agent.
        """
        self.name = name
    
    async def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message.
        
        Args:
            message: The message to process.
            
        Returns:
            Dict[str, Any]: The response message.
        """
        raise NotImplementedError("Subclasses must implement process method")

class RoundRobinGroupChat:
    """
    A group chat that processes messages in a round-robin fashion.
    Each agent gets a turn to process the message in sequence.
    """
    
    def __init__(self, agents: List[Agent]):
        """
        Initialize a RoundRobinGroupChat.
        
        Args:
            agents: List of agents to participate in the chat.
        """
        self.agents = agents
        self.messages = []
    
    async def process(self, input_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an input message through all agents in sequence.
        
        Args:
            input_message: The input message to process.
            
        Returns:
            Dict[str, Any]: The final response after all agents have processed the message.
        """
        current_message = input_message
        self.messages.append({"role": "user", "content": current_message})
        
        for agent in self.agents:
            logging.info(f"Agent {agent.name} processing message")
            current_message = await agent.process(current_message)
            self.messages.append({"role": agent.name, "content": current_message})
        
        return current_message

class SelectorGroupChat:
    """
    A group chat that selects which agent should process each message.
    """
    
    def __init__(self, agents: List[Agent], selector: Callable[[Dict[str, Any], List[Agent]], Agent]):
        """
        Initialize a SelectorGroupChat.
        
        Args:
            agents: List of agents to participate in the chat.
            selector: Function that selects which agent should process a message.
        """
        self.agents = agents
        self.selector = selector
        self.messages = []
    
    async def process(self, input_message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an input message by selecting an appropriate agent.
        
        Args:
            input_message: The input message to process.
            
        Returns:
            Dict[str, Any]: The response from the selected agent.
        """
        self.messages.append({"role": "user", "content": input_message})
        
        # Select an agent to process the message
        agent = self.selector(input_message, self.agents)
        logging.info(f"Selected agent {agent.name} to process message")
        
        # Process the message with the selected agent
        response = await agent.process(input_message)
        self.messages.append({"role": agent.name, "content": response})
        
        return response

# Example selector function
def topic_based_selector(message: Dict[str, Any], agents: List[Agent]) -> Agent:
    """
    Select an agent based on the topic of the message.
    
    Args:
        message: The message to process.
        agents: List of available agents.
        
    Returns:
        Agent: The selected agent.
    """
    # This is a simple example. In a real implementation, you would use more
    # sophisticated logic to determine which agent should handle the message.
    topic = message.get("topic", "").lower()
    
    if "research" in topic or "search" in topic:
        for agent in agents:
            if agent.name == "Researcher":
                return agent
    
    if "summarize" in topic or "summary" in topic:
        for agent in agents:
            if agent.name == "Summarizer":
                return agent
    
    # Default to the first agent if no match is found
    return agents[0]
