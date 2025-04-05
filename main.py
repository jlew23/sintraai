# AI-Powered Executive Team
# Main project structure and components

import os
from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
import chromadb
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader
import rasa
import activepieces
import logging

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Slack app
slack_app = App(token=os.environ["SLACK_BOT_TOKEN"])

# Initialize ChromaDB for the knowledge base (Brain)
class KnowledgeBase:
    def __init__(self, persist_directory="./brain_data"):
        self.embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
        self.persist_directory = persist_directory
        self.vector_db = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
    def add_documents(self, documents_path):
        """Add documents to the knowledge base from a directory"""
        loader = DirectoryLoader(documents_path, glob="**/*.txt", loader_cls=TextLoader)
        documents = loader.load()
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        self.vector_db.add_documents(splits)
        self.vector_db.persist()
        logger.info(f"Added {len(splits)} document chunks to the knowledge base")
        
    def query(self, query, k=5):
        """Query the knowledge base for relevant information"""
        results = self.vector_db.similarity_search(query, k=k)
        return results

# Base Agent class
class Agent:
    def __init__(self, name, role, knowledge_base):
        self.name = name
        self.role = role
        self.knowledge_base = knowledge_base
        
    def process_message(self, message):
        """Process incoming message and generate a response"""
        # This is where you would integrate with Rasa or another NLU system
        # For now, we'll just use a placeholder
        context = self.get_context(message)
        return f"Response from {self.name} ({self.role}) with context: {context}"
    
    def get_context(self, message, k=3):
        """Get relevant context from the knowledge base"""
        docs = self.knowledge_base.query(message, k=k)
        return [doc.page_content for doc in docs]

# Director Agent - Orchestrates other agents
class DirectorAgent(Agent):
    def __init__(self, knowledge_base, agents=None):
        super().__init__("Director", "Executive Director", knowledge_base)
        self.agents = agents or {}
        
    def add_agent(self, agent_name, agent):
        """Add an agent to the team"""
        self.agents[agent_name] = agent
        
    def delegate_task(self, message):
        """Delegate a task to the appropriate agent"""
        # This would normally use NLU to determine which agent should handle the task
        # For now, we'll use a simple keyword matching approach
        
        for agent_name, agent in self.agents.items():
            if agent_name.lower() in message.lower():
                return agent.process_message(message)
        
        # If no specific agent matches, handle it at the director level
        return self.process_message(message)

# Specialized agents
class SalesAgent(Agent):
    def __init__(self, knowledge_base):
        super().__init__("Sales", "Sales Director", knowledge_base)

class MarketingAgent(Agent):
    def __init__(self, knowledge_base):
        super().__init__("Marketing", "Marketing Director", knowledge_base)

class FinanceAgent(Agent):
    def __init__(self, knowledge_base):
        super().__init__("Finance", "Finance Director", knowledge_base)

class CustomerServiceAgent(Agent):
    def __init__(self, knowledge_base):
        super().__init__("Support", "Customer Service Manager", knowledge_base)

class TechnicalSupportAgent(Agent):
    def __init__(self, knowledge_base):
        super().__init__("Tech", "Technical Support Manager", knowledge_base)

# Slack event handlers
@slack_app.event("app_mention")
def handle_mention(event, say):
    """Handle mentions in Slack"""
    message = event["text"]
    user = event["user"]
    channel = event["channel"]
    
    # This is where you would route the message to the Director Agent
    # For now, we'll just acknowledge it
    say(f"Hello <@{user}>! I received your message and I'm processing it.")

@slack_app.event("message")
def handle_message(event, say):
    """Handle direct messages in Slack"""
    if "channel_type" in event and event["channel_type"] == "im":
        message = event["text"]
        user = event["user"]
        
        # Process the message through the Director Agent
        # For now, we'll just acknowledge it
        say(f"Hello <@{user}>! I received your direct message and I'm processing it.")

# Main application
def main():
    # Initialize knowledge base
    brain = KnowledgeBase()
    
    # Initialize agents
    sales_agent = SalesAgent(brain)
    marketing_agent = MarketingAgent(brain)
    finance_agent = FinanceAgent(brain)
    customer_service_agent = CustomerServiceAgent(brain)
    technical_support_agent = TechnicalSupportAgent(brain)
    
    # Initialize director agent and add team members
    director = DirectorAgent(brain)
    director.add_agent("sales", sales_agent)
    director.add_agent("marketing", marketing_agent)
    director.add_agent("finance", finance_agent)
    director.add_agent("customer", customer_service_agent)
    director.add_agent("technical", technical_support_agent)
    
    # Start Slack app
    SocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"]).start()

if __name__ == "__main__":
    main()
