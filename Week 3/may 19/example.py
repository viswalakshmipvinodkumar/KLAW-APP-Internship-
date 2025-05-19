"""
Example usage of the Web Research Assistant.

This example demonstrates how to use the Web Research Assistant to research
a specific topic and generate a summary.
"""
import asyncio
import logging
import os
from typing import Dict, List, Any

from dotenv import load_dotenv

from agents.researcher import Researcher
from agents.summarizer import Summarizer
from tools.web_browser import WebBrowser
from tools.text_summarizer import TextSummarizer
from utils.logging_config import setup_logging
from utils.output_formatter import OutputFormatter

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

async def research_example():
    """Example function demonstrating the Web Research Assistant."""
    try:
        # Define a research topic
        topic = "Asynchronous programming in Python"
        print(f"Researching topic: {topic}")
        
        # Initialize tools
        browser = WebBrowser(headless=True)
        summarizer_tool = TextSummarizer()
        
        # Initialize agents
        researcher = Researcher(browser)
        summarizer = Summarizer(summarizer_tool)
        
        # Step 1: Research the topic
        print("Step 1: Researching the topic...")
        sources = await researcher.research_topic(topic, num_sources=3)
        print(f"Found {len(sources)} sources")
        
        # Step 2: Summarize the sources
        print("Step 2: Summarizing the sources...")
        summaries = await summarizer.summarize_sources(sources)
        
        # Step 3: Generate a comprehensive topic summary
        print("Step 3: Generating comprehensive summary...")
        topic_summary = await summarizer.generate_topic_summary(topic, summaries)
        
        # Combine results
        results = {
            "topic": topic,
            "summary": topic_summary,
            "key_points": summaries.get("key_points", []),
            "sentiment": summaries.get("sentiment", {}),
            "source_summaries": summaries.get("source_summaries", [])
        }
        
        # Format and save the results
        formatter = OutputFormatter()
        output_path = formatter.save_research_results(topic, results)
        
        print(f"Research complete! Results saved to: {output_path}")
        
        # Print a preview of the results
        print("\nResearch Summary Preview:")
        print("-" * 40)
        print(f"{topic_summary[:500]}...")
        
        print("\nKey Points:")
        for i, point in enumerate(summaries.get("key_points", []), 1):
            print(f"{i}. {point}")
        
    except Exception as e:
        logging.error(f"Error in research_example: {e}")
        print(f"An error occurred: {e}")
    finally:
        # Clean up resources
        if 'browser' in locals():
            browser.close()

if __name__ == "__main__":
    asyncio.run(research_example())
