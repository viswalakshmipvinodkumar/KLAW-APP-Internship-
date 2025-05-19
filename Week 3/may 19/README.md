# Web Research Assistant

A powerful web research tool that uses Selenium for web browsing and Gemini for text summarization.

## Features

- **Web Browser Tool**: Uses Selenium to navigate and extract content from websites
- **Text Summarizer Tool**: Leverages Gemini to generate concise summaries of web content
- **Researcher Agent**: Searches for and extracts relevant information from websites
- **Summarizer Agent**: Processes and summarizes the extracted content
- **Asynchronous Processing**: Handles multiple web requests smoothly

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Gemini API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

3. Run the main script:
   ```
   python main.py
   ```

## Usage

The system will prompt you to enter a research topic. It will then:
1. Search for relevant information on the web
2. Extract content from multiple sources
3. Generate a comprehensive summary of the findings

## Components

- `main.py`: Entry point for the application
- `tools/`: Contains the web browser and text summarizer tools
- `agents/`: Contains the researcher and summarizer agents
- `utils/`: Helper functions and utilities
