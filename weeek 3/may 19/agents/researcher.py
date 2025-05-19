"""
Researcher Agent for fetching and extracting web content.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union

from tools.web_browser import WebBrowser

class Researcher:
    """
    Agent responsible for researching topics on the web and extracting relevant content.
    """
    
    def __init__(self, browser: Optional[WebBrowser] = None):
        """
        Initialize the Researcher agent.
        
        Args:
            browser: An instance of WebBrowser tool. If None, a new instance will be created.
        """
        self.browser = browser or WebBrowser(headless=True)
        self.search_results = []
        
    async def research_topic(self, topic: str, num_sources: int = 3) -> List[Dict[str, str]]:
        """
        Research a topic by searching the web and extracting content from multiple sources.
        
        Args:
            topic: The topic to research.
            num_sources: Number of sources to extract content from.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with title, url, and content.
        """
        logging.info(f"Researching topic: {topic}")
        
        # Search for the topic
        self.search_results = await self.browser.search_google(topic, num_results=num_sources+2)
        
        if not self.search_results:
            logging.warning(f"No search results found for topic: {topic}")
            return []
        
        # Extract content from each source
        results = []
        tasks = []
        
        for result in self.search_results[:num_sources+2]:
            tasks.append(self.extract_source_content(result))
        
        # Process all sources concurrently
        sources = await asyncio.gather(*tasks)
        
        # Filter out empty results
        return [source for source in sources if source and source.get("content")]
    
    async def extract_source_content(self, search_result: Dict[str, str]) -> Dict[str, str]:
        """
        Extract content from a single source.
        
        Args:
            search_result: Dictionary with title and url.
            
        Returns:
            Dict[str, str]: Dictionary with title, url, and content.
        """
        url = search_result.get("url")
        title = search_result.get("title")
        
        if not url:
            return {}
        
        try:
            logging.info(f"Extracting content from: {url}")
            content = await self.browser.extract_article_content(url)
            
            if not content or len(content) < 200:
                logging.warning(f"Insufficient content extracted from: {url}")
                return {}
            
            return {
                "title": title,
                "url": url,
                "content": content
            }
        except Exception as e:
            logging.error(f"Error extracting content from {url}: {e}")
            return {}
    
    async def get_additional_details(self, url: str, selectors: List[str]) -> Dict[str, str]:
        """
        Get additional details from a webpage using specific CSS selectors.
        
        Args:
            url: The URL to extract details from.
            selectors: Dictionary of CSS selectors to extract.
            
        Returns:
            Dict[str, str]: Dictionary with extracted details.
        """
        try:
            await self.browser.navigate(url)
            
            results = {}
            for selector in selectors:
                content = await self.browser.extract_content(selector, wait_time=2)
                if content:
                    results[selector] = content
            
            return results
        except Exception as e:
            logging.error(f"Error getting additional details from {url}: {e}")
            return {}
    
    def close(self):
        """Close the browser."""
        if self.browser:
            self.browser.close()
    
    def __del__(self):
        """Destructor to ensure the browser is closed."""
        self.close()
