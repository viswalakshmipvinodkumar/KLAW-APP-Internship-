"""
Output formatter for the Web Research Assistant.
"""
import json
import os
from typing import Dict, List, Union, Any
from datetime import datetime

class OutputFormatter:
    """Formats and saves research results."""
    
    def __init__(self, output_dir: str = "research_results"):
        """
        Initialize the OutputFormatter.
        
        Args:
            output_dir: Directory to save research results.
        """
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def format_research_results(self, topic: str, results: Dict[str, Any]) -> str:
        """
        Format research results as a readable text.
        
        Args:
            topic: The research topic.
            results: The research results.
            
        Returns:
            str: Formatted research results.
        """
        summary = results.get("summary", "No summary available.")
        key_points = results.get("key_points", [])
        sentiment = results.get("sentiment", {})
        source_summaries = results.get("source_summaries", [])
        
        # Format the output
        output = f"""
# Research Summary: {topic}

## Overall Summary
{summary}

## Key Points
"""
        
        for i, point in enumerate(key_points, 1):
            output += f"{i}. {point}\n"
        
        output += f"""
## Overall Sentiment
{sentiment.get('sentiment', 'neutral')} (Confidence: {sentiment.get('confidence', 0.0):.2f})

## Sources
"""
        
        for i, source in enumerate(source_summaries, 1):
            title = source.get("title", "Untitled")
            url = source.get("url", "")
            source_summary = source.get("summary", "No summary available.")
            source_key_points = source.get("key_points", [])
            
            output += f"""
### {i}. {title}
**URL**: {url}

**Summary**:
{source_summary}

**Key Points**:
"""
            
            for j, point in enumerate(source_key_points, 1):
                output += f"- {point}\n"
        
        return output
    
    def save_research_results(self, topic: str, results: Dict[str, Any]) -> str:
        """
        Save research results to files.
        
        Args:
            topic: The research topic.
            results: The research results.
            
        Returns:
            str: Path to the saved markdown file.
        """
        # Create a sanitized filename from the topic
        filename_base = "".join(c if c.isalnum() else "_" for c in topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{filename_base}_{timestamp}"
        
        # Save as markdown
        md_path = os.path.join(self.output_dir, f"{filename}.md")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self.format_research_results(topic, results))
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        return md_path
