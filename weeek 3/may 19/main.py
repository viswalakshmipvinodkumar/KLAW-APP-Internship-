"""
Web Research Assistant - Main Application

This application uses Selenium for web browsing and Gemini for text summarization
to research topics and generate concise summaries.
"""
import asyncio
import logging
import os
from typing import Dict, List, Optional, Union, Any

from dotenv import load_dotenv

from agents.researcher import Researcher
from agents.summarizer import Summarizer
from tools.web_browser import WebBrowser
from tools.text_summarizer import TextSummarizer
from utils.logging_config import setup_logging
from utils.output_formatter import OutputFormatter
from utils.group_chat import RoundRobinGroupChat, SelectorGroupChat, Agent, topic_based_selector

# Load environment variables
load_dotenv()

# Check for API key
if not os.getenv("GEMINI_API_KEY"):
    print("ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Please create a .env file with your Gemini API key:")
    print("GEMINI_API_KEY=your_api_key_here")
    exit(1)

# Setup logging
setup_logging()

class ResearcherAgent(Agent):
    """Wrapper for the Researcher agent to fit into the group chat framework."""
    
    def __init__(self, researcher: Researcher):
        """
        Initialize the ResearcherAgent.
        
        Args:
            researcher: The Researcher instance.
        """
        super().__init__("Researcher")
        self.researcher = researcher
    
    async def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message by researching the topic.
        
        Args:
            message: The message to process.
            
        Returns:
            Dict[str, Any]: The research results.
        """
        topic = message.get("topic", "")
        num_sources = message.get("num_sources", 3)
        
        logging.info(f"ResearcherAgent processing topic: {topic}")
        sources = await self.researcher.research_topic(topic, num_sources=num_sources)
        
        return {
            "topic": topic,
            "sources": sources
        }

class SummarizerAgent(Agent):
    """Wrapper for the Summarizer agent to fit into the group chat framework."""
    
    def __init__(self, summarizer: Summarizer):
        """
        Initialize the SummarizerAgent.
        
        Args:
            summarizer: The Summarizer instance.
        """
        super().__init__("Summarizer")
        self.summarizer = summarizer
    
    async def process(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a message by summarizing the sources.
        
        Args:
            message: The message to process.
            
        Returns:
            Dict[str, Any]: The summarization results.
        """
        topic = message.get("topic", "")
        sources = message.get("sources", [])
        
        logging.info(f"SummarizerAgent processing {len(sources)} sources")
        summaries = await self.summarizer.summarize_sources(sources)
        
        # Generate a comprehensive topic summary
        topic_summary = await self.summarizer.generate_topic_summary(topic, summaries)
        
        return {
            "topic": topic,
            "summary": topic_summary,
            "key_points": summaries.get("key_points", []),
            "sentiment": summaries.get("sentiment", {}),
            "source_summaries": summaries.get("source_summaries", [])
        }

async def main():
    """Main function to run the Web Research Assistant."""
    try:
        # Print welcome message
        print("\n" + "=" * 80)
        print("Welcome to the Web Research Assistant!")
        print("This tool will help you research topics and generate concise summaries.")
        print("=" * 80 + "\n")
        
        # Get user input
        topic = input("Enter a research topic: ")
        num_sources = int(input("Enter the number of sources to research (default: 3): ") or "3")
        
        print(f"\nResearching '{topic}' using {num_sources} sources...\n")
        
        # Initialize tools
        browser = WebBrowser(headless=True)
        summarizer_tool = TextSummarizer()
        
        # Initialize agents
        researcher = Researcher(browser)
        summarizer = Summarizer(summarizer_tool)
        
        # Create agent wrappers for the group chat
        researcher_agent = ResearcherAgent(researcher)
        summarizer_agent = SummarizerAgent(summarizer)
        
        # Choose which group chat implementation to use
        # Uncomment the one you want to use
        
        # Option 1: RoundRobinGroupChat - processes messages through all agents in sequence
        group_chat = RoundRobinGroupChat([researcher_agent, summarizer_agent])
        
        # Option 2: SelectorGroupChat - selects which agent should process each message
        # group_chat = SelectorGroupChat([researcher_agent, summarizer_agent], topic_based_selector)
        
        # Process the research request
        input_message = {
            "topic": topic,
            "num_sources": num_sources
        }
        
        results = await group_chat.process(input_message)
        
        # Format and save the results
        formatter = OutputFormatter()
        formatted_results = formatter.format_research_results(topic, results)
        output_path = formatter.save_research_results(topic, results)
        
        print("\n" + "=" * 80)
        print(f"Research complete! Results saved to: {output_path}")
        print("=" * 80 + "\n")
        
        # Print a preview of the results
        print("Research Summary Preview:")
        print("-" * 40)
        
        # Print the first 500 characters of the summary
        summary = results.get("summary", "No summary available.")
        print(f"{summary[:500]}...")
        
        print("\nKey Points:")
        for i, point in enumerate(results.get("key_points", []), 1):
            print(f"{i}. {point}")
        
        print("\nSources:")
        for i, source in enumerate(results.get("source_summaries", []), 1):
            print(f"{i}. {source.get('title', 'Untitled')}: {source.get('url', '')}")
        
        print("\nFor the full research results, please check the saved file.")
        
    except KeyboardInterrupt:
        print("\nResearch interrupted by user.")
    except Exception as e:
        logging.error(f"Error in main function: {e}")
        print(f"\nAn error occurred: {e}")
    finally:
        # Clean up resources
        if 'browser' in locals():
            browser.close()
        
        print("\nThank you for using the Web Research Assistant!")

if __name__ == "__main__":
    asyncio.run(main())
