# FAQ Chatbot with RAG (Retrieval-Augmented Generation)

This project implements a FAQ chatbot using Retrieval-Augmented Generation (RAG) with Gemini API and ChromaDB. The system enhances responses by retrieving relevant information from a knowledge base before generating answers.

## Features

- **Retrieval-Augmented Generation (RAG)**: Combines information retrieval with text generation
- **Vector Database**: Uses ChromaDB for efficient similarity search
- **Asynchronous Processing**: Handles real-time user queries with async functions
- **Group Chat Architecture**: Implements both RoundRobinGroupChat and SelectorGroupChat
- **Modular Design**: Separate components for tools, agents, and data loading

## Components

1. **Tools**:
   - `ChromaDBTool`: Interface to ChromaDB vector database
   - `GeminiTool`: Interface to Google's Gemini API

2. **Agents**:
   - `QueryHandlerAgent`: Processes user queries and determines how to route them
   - `RAGRetrieverAgent`: Retrieves relevant information and generates responses

3. **Group Chat**:
   - `RoundRobinGroupChat`: Each agent takes turns responding
   - `SelectorGroupChat`: A selector agent chooses which agent should respond

4. **Data Loader**: Loads and processes text data for the RAG system

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Create a `.env` file with your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```

3. Add your knowledge base text files to the `data` directory

## Usage

Run the main application:
```
python main.py
```

The chatbot will start and prompt you for questions. Type 'exit' to quit.

## Customization

- Modify the `ai_faq.txt` file in the `data` directory to add more information
- Adjust chunk size and overlap in `data_loader.py` for different document splitting behavior
- Switch between RoundRobinGroupChat and SelectorGroupChat in `main.py`
- Modify system prompts in the agents for different response styles

## Architecture

```
┌─────────────┐     ┌───────────────┐     ┌─────────────┐
│ User Query  │────▶│ Group Chat    │────▶│  Response   │
└─────────────┘     └───────┬───────┘     └─────────────┘
                           │
                ┌──────────┴──────────┐
                ▼                     ▼
        ┌───────────────┐    ┌────────────────┐
        │ Query Handler │    │ RAG Retriever  │
        └───────┬───────┘    └────────┬───────┘
                │                     │
                ▼                     ▼
        ┌───────────────┐    ┌────────────────┐
        │  Gemini API   │    │   ChromaDB     │
        └───────────────┘    └────────────────┘
```

## License

MIT
