"""
Text Summarizer Tool using Google's Gemini for efficient text processing and summarization.
"""
import os
import asyncio
import logging
from typing import Dict, List, Optional, Union

import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is not set")

# Configure the API key for Google Generative AI
genai.configure(api_key=GEMINI_API_KEY)

# Set environment variable for LangChain
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

class TextSummarizer:
    """A tool for processing and summarizing text data using Google's Gemini."""
    
    def __init__(self, model_name: str = "gemini-1.5-pro"):
        """
        Initialize the TextSummarizer tool.
        
        Args:
            model_name: The Gemini model to use.
        """
        self.model_name = model_name
        self.llm = ChatGoogleGenerativeAI(model=model_name, temperature=0.3)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=10000,
            chunk_overlap=1000,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    async def summarize(self, text: str, max_length: int = 1000) -> str:
        """
        Summarize the given text.
        
        Args:
            text: The text to summarize.
            max_length: Maximum length of the summary.
            
        Returns:
            str: The summarized text.
        """
        if not text or len(text) < 100:
            return text
        
        try:
            # Use asyncio to run the blocking LLM code in a separate thread
            return await asyncio.to_thread(self._summarize_sync, text, max_length)
        except Exception as e:
            logging.error(f"Error summarizing text: {e}")
            return text[:max_length] + "... [Error: Could not generate complete summary]"
    
    def _summarize_sync(self, text: str, max_length: int) -> str:
        """Synchronous version of summarize method."""
        try:
            # Split text into chunks if it's too long
            docs = self._split_text(text)
            
            # Use LangChain's summarize chain
            chain = load_summarize_chain(
                self.llm,
                chain_type="map_reduce",
                verbose=False
            )
            
            summary = chain.run(docs)
            
            # Ensure the summary is not too long
            if len(summary) > max_length:
                summary = summary[:max_length] + "..."
                
            return summary
        except Exception as e:
            logging.error(f"Error in _summarize_sync: {e}")
            return text[:max_length] + "... [Error: Could not generate complete summary]"
    
    def _split_text(self, text: str) -> List[Document]:
        """
        Split text into chunks for processing.
        
        Args:
            text: The text to split.
            
        Returns:
            List[Document]: List of Document objects.
        """
        texts = self.text_splitter.split_text(text)
        return [Document(page_content=t) for t in texts]
    
    async def extract_key_points(self, text: str, num_points: int = 5) -> List[str]:
        """
        Extract key points from the text.
        
        Args:
            text: The text to extract key points from.
            num_points: Number of key points to extract.
            
        Returns:
            List[str]: List of key points.
        """
        try:
            prompt = f"""
            Extract exactly {num_points} key points from the following text. 
            Format each point as a concise, informative statement.
            
            TEXT:
            {text}
            
            KEY POINTS:
            """
            
            # Use direct Gemini API for this task
            model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            # Process the response
            content = response.text
            
            # Extract points (assuming they're numbered or bulleted)
            points = []
            for line in content.split('\n'):
                line = line.strip()
                if line and (line.startswith('-') or line.startswith('•') or 
                            (len(line) > 2 and line[0].isdigit() and line[1] == '.')):
                    # Remove the bullet or number
                    point = line[2:].strip() if line[1] in ['.', ' '] else line[1:].strip()
                    points.append(point)
            
            # If no points were extracted with formatting, just split by newlines
            if not points:
                points = [line.strip() for line in content.split('\n') if line.strip()]
            
            # Limit to requested number
            return points[:num_points]
        except Exception as e:
            logging.error(f"Error extracting key points: {e}")
            return ["Error extracting key points"]
    
    async def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze the sentiment of the text.
        
        Args:
            text: The text to analyze.
            
        Returns:
            Dict[str, Union[str, float]]: Sentiment analysis results.
        """
        try:
            prompt = f"""
            Analyze the sentiment of the following text. 
            Provide a sentiment label (positive, negative, or neutral) and a confidence score (0-1).
            Format your response exactly as JSON with keys "sentiment" and "confidence".
            
            TEXT:
            {text}
            """
            
            model = genai.GenerativeModel(self.model_name)
            response = await asyncio.to_thread(model.generate_content, prompt)
            
            # Extract JSON from response
            import json
            import re
            
            content = response.text
            # Look for JSON pattern
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                try:
                    result = json.loads(json_match.group(0))
                    # Ensure expected keys exist
                    if "sentiment" not in result or "confidence" not in result:
                        raise ValueError("Missing expected keys in result")
                    return result
                except json.JSONDecodeError:
                    pass
            
            # Fallback if JSON parsing fails
            if "positive" in content.lower():
                sentiment = "positive"
            elif "negative" in content.lower():
                sentiment = "negative"
            else:
                sentiment = "neutral"
                
            return {
                "sentiment": sentiment,
                "confidence": 0.7  # Default confidence
            }
        except Exception as e:
            logging.error(f"Error analyzing sentiment: {e}")
            return {"sentiment": "neutral", "confidence": 0.5}
