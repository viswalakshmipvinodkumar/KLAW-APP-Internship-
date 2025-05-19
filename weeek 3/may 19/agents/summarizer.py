"""
Summarizer Agent for processing and summarizing text data.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union

from tools.text_summarizer import TextSummarizer

class Summarizer:
    """
    Agent responsible for processing and summarizing text data.
    """
    
    def __init__(self, summarizer: Optional[TextSummarizer] = None):
        """
        Initialize the Summarizer agent.
        
        Args:
            summarizer: An instance of TextSummarizer tool. If None, a new instance will be created.
        """
        self.summarizer = summarizer or TextSummarizer()
    
    async def summarize_sources(self, sources: List[Dict[str, str]]) -> Dict[str, Union[str, List[Dict[str, str]]]]:
        """
        Summarize multiple sources of content.
        
        Args:
            sources: List of dictionaries with title, url, and content.
            
        Returns:
            Dict[str, Union[str, List[Dict[str, str]]]]: Dictionary with overall summary and individual source summaries.
        """
        if not sources:
            return {"summary": "No sources provided for summarization.", "source_summaries": []}
        
        logging.info(f"Summarizing {len(sources)} sources")
        
        # Process each source concurrently
        tasks = []
        for source in sources:
            tasks.append(self.summarize_source(source))
        
        source_summaries = await asyncio.gather(*tasks)
        
        # Filter out empty summaries
        source_summaries = [s for s in source_summaries if s]
        
        if not source_summaries:
            return {"summary": "Failed to generate summaries from the provided sources.", "source_summaries": []}
        
        # Create an overall summary from all source summaries
        combined_text = "\n\n".join([s.get("summary", "") for s in source_summaries])
        overall_summary = await self.summarizer.summarize(
            combined_text, 
            max_length=2000
        )
        
        # Extract key points from the overall summary
        key_points = await self.summarizer.extract_key_points(overall_summary, num_points=5)
        
        # Analyze sentiment of the overall summary
        sentiment = await self.summarizer.analyze_sentiment(overall_summary)
        
        return {
            "summary": overall_summary,
            "key_points": key_points,
            "sentiment": sentiment,
            "source_summaries": source_summaries
        }
    
    async def summarize_source(self, source: Dict[str, str]) -> Dict[str, str]:
        """
        Summarize a single source.
        
        Args:
            source: Dictionary with title, url, and content.
            
        Returns:
            Dict[str, str]: Dictionary with title, url, and summary.
        """
        title = source.get("title", "Untitled")
        url = source.get("url", "")
        content = source.get("content", "")
        
        if not content:
            logging.warning(f"No content to summarize for source: {url}")
            return {}
        
        try:
            logging.info(f"Summarizing content from: {url}")
            
            # Generate a summary of the content
            summary = await self.summarizer.summarize(content, max_length=500)
            
            # Extract key points
            key_points = await self.summarizer.extract_key_points(content, num_points=3)
            
            return {
                "title": title,
                "url": url,
                "summary": summary,
                "key_points": key_points
            }
        except Exception as e:
            logging.error(f"Error summarizing content from {url}: {e}")
            return {}
    
    async def generate_topic_summary(self, topic: str, summaries: Dict[str, Union[str, List[Dict[str, str]]]]) -> str:
        """
        Generate a comprehensive summary of a topic based on the summarized sources.
        
        Args:
            topic: The research topic.
            summaries: Dictionary with overall summary and individual source summaries.
            
        Returns:
            str: A comprehensive summary of the topic.
        """
        try:
            overall_summary = summaries.get("summary", "")
            key_points = summaries.get("key_points", [])
            sentiment = summaries.get("sentiment", {})
            source_summaries = summaries.get("source_summaries", [])
            
            # Create a prompt for the LLM to generate a comprehensive summary
            prompt = f"""
            Generate a comprehensive research summary on the topic: "{topic}"
            
            Overall Summary:
            {overall_summary}
            
            Key Points:
            {' '.join(['- ' + point for point in key_points])}
            
            Sources:
            {' '.join([f"- {s.get('title', 'Untitled')}: {s.get('summary', '')[:100]}..." for s in source_summaries])}
            
            The overall sentiment of the research is {sentiment.get('sentiment', 'neutral')}.
            
            Please provide a well-structured, informative summary that synthesizes all this information.
            """
            
            # Use the summarizer to generate the final summary
            final_summary = await self.summarizer.summarize(prompt, max_length=2000)
            
            return final_summary
        except Exception as e:
            logging.error(f"Error generating topic summary: {e}")
            return "Failed to generate a comprehensive summary for the topic."
