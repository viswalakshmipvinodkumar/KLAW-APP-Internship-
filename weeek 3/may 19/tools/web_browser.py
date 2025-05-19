"""
Web Browser Tool using Selenium for web browsing and data extraction.
"""
import asyncio
import logging
from typing import Dict, List, Optional, Union

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

class WebBrowser:
    """A tool for web browsing and data extraction using Selenium."""
    
    def __init__(self, headless: bool = True):
        """
        Initialize the WebBrowser tool.
        
        Args:
            headless: Whether to run the browser in headless mode.
        """
        self.headless = headless
        self.driver = None
        self.setup_driver()
        
    def setup_driver(self):
        """Set up the Selenium WebDriver."""
        chrome_options = Options()
        if self.headless:
            chrome_options.add_argument("--headless")
        
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        
        try:
            service = Service(ChromeDriverManager().install())
            self.driver = webdriver.Chrome(service=service, options=chrome_options)
            logging.info("WebDriver initialized successfully")
        except Exception as e:
            logging.error(f"Error initializing WebDriver: {e}")
            raise
    
    async def navigate(self, url: str) -> bool:
        """
        Navigate to a URL.
        
        Args:
            url: The URL to navigate to.
            
        Returns:
            bool: True if navigation was successful, False otherwise.
        """
        try:
            # Use asyncio to run the blocking Selenium code in a separate thread
            return await asyncio.to_thread(self._navigate_sync, url)
        except Exception as e:
            logging.error(f"Error navigating to {url}: {e}")
            return False
    
    def _navigate_sync(self, url: str) -> bool:
        """Synchronous version of navigate method."""
        try:
            self.driver.get(url)
            return True
        except Exception as e:
            logging.error(f"Error in _navigate_sync to {url}: {e}")
            return False
    
    async def extract_content(self, selector: str = "body", wait_time: int = 5) -> str:
        """
        Extract content from the current page.
        
        Args:
            selector: CSS selector to extract content from.
            wait_time: Time to wait for the element to be present.
            
        Returns:
            str: The extracted content.
        """
        try:
            return await asyncio.to_thread(self._extract_content_sync, selector, wait_time)
        except Exception as e:
            logging.error(f"Error extracting content: {e}")
            return ""
    
    def _extract_content_sync(self, selector: str, wait_time: int) -> str:
        """Synchronous version of extract_content method."""
        try:
            WebDriverWait(self.driver, wait_time).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, selector))
            )
            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.extract()
            
            # Get text
            text = soup.get_text()
            
            # Break into lines and remove leading and trailing space on each
            lines = (line.strip() for line in text.splitlines())
            # Break multi-headlines into a line each
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            # Remove blank lines
            text = '\n'.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            logging.error(f"Error in _extract_content_sync: {e}")
            return ""
    
    async def search_google(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """
        Search Google for a query and return the top results.
        
        Args:
            query: The search query.
            num_results: Number of results to return.
            
        Returns:
            List[Dict[str, str]]: List of dictionaries with title and url.
        """
        search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}"
        
        try:
            await self.navigate(search_url)
            return await asyncio.to_thread(self._extract_search_results, num_results)
        except Exception as e:
            logging.error(f"Error searching Google: {e}")
            return []
    
    def _extract_search_results(self, num_results: int) -> List[Dict[str, str]]:
        """Extract search results from Google search page."""
        try:
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "div.g"))
            )
            
            results = []
            elements = self.driver.find_elements(By.CSS_SELECTOR, "div.g")
            
            for i, element in enumerate(elements):
                if i >= num_results:
                    break
                
                try:
                    title_element = element.find_element(By.CSS_SELECTOR, "h3")
                    title = title_element.text
                    
                    link_element = element.find_element(By.CSS_SELECTOR, "a")
                    url = link_element.get_attribute("href")
                    
                    if title and url:
                        results.append({"title": title, "url": url})
                except Exception:
                    continue
            
            return results
        except Exception as e:
            logging.error(f"Error in _extract_search_results: {e}")
            return []
    
    async def extract_article_content(self, url: str) -> str:
        """
        Extract the main content from an article.
        
        Args:
            url: The URL of the article.
            
        Returns:
            str: The extracted article content.
        """
        try:
            await self.navigate(url)
            
            # Try to find the main content
            selectors = [
                "article", 
                "main", 
                ".article-content", 
                ".post-content", 
                ".entry-content",
                "#content",
                ".content"
            ]
            
            for selector in selectors:
                try:
                    content = await self.extract_content(selector, wait_time=2)
                    if content and len(content) > 200:  # Ensure we have meaningful content
                        return content
                except Exception:
                    continue
            
            # If no specific content container found, extract from body
            return await self.extract_content("body")
        except Exception as e:
            logging.error(f"Error extracting article content from {url}: {e}")
            return ""
    
    def close(self):
        """Close the WebDriver."""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logging.info("WebDriver closed")
    
    def __del__(self):
        """Destructor to ensure the WebDriver is closed."""
        self.close()
