# AI-Powered Executive Team

This project implements an AI-powered executive team that can assist a CEO in running their company. The system consists of AI agents with different roles (Sales, Marketing, Finance, Customer Service, Technical Support) that communicate via Slack and have access to a shared knowledge base.

## Project Structure

- `main.py`: Main application entry point
- `agents/`: Contains the specialized agent implementations
- `brain_data/`: Vector database storage for the knowledge base
- `data/`: Company documents and other data sources
- `config/`: Configuration files
- `logs/`: Log files
- `rasa/`: Rasa NLU configuration and training data

## Setup Instructions

1. Clone this repository
2. Create a virtual environment: `python -m venv venv`
3. Activate the virtual environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
4. Install dependencies: `pip install -r requirements.txt`
5. Copy `.env.example` to `.env` and fill in your credentials
6. Set up the knowledge base by adding documents to the `data/` directory
7. Run the application: `python main.py`

## Agent Types

- **Director Agent**: Orchestrates the team and delegates tasks
- **Sales Agent**: Handles sales-related queries and tasks
- **Marketing Agent**: Manages marketing campaigns and content
- **Finance Agent**: Deals with financial matters and reporting
- **Customer Service Agent**: Provides customer support
- **Technical Support Agent**: Offers technical assistance

## Knowledge Base ("Brain")

The knowledge base stores company documents, procedures, and best practices. It uses vector embeddings to enable semantic search and retrieval of relevant information.

## Technology Stack

- Slack API: For communication with users
- Activepieces: For workflow orchestration
- Rasa: For natural language understanding
- ChromaDB: For vector storage and retrieval
- SentenceTransformers: For document embeddings
- LangChain: For agent framework and utilities
