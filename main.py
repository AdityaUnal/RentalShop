import re
import os
import uuid
import chromadb
import torch
import sqlite3
from datetime import datetime
from typing import Annotated, Literal, TypedDict
from dotenv import load_dotenv


from langchain.tools import tool
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages    
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# Load environment variables
load_dotenv()
TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
HF_API_KEY = os.getenv('HUGGINGFACE_API_KEY')

class ModelConfig:
    def __init__(self):
        self.embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.llm_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
        
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=self.embedding_model_name, 
            model_kwargs={'device': self.device}
        )
        self.max_length = self.embedding_model._client.tokenizer.model_max_length - 50
        
        self.llm = HuggingFaceEndpoint(
            repo_id=self.llm_model_name,
            task="conversational",
            huggingfacehub_api_token=HF_API_KEY
        )
        self.chat_llm = ChatHuggingFace(llm=self.llm)

class DatabaseConfig:
    def __init__(self):
        self.local_file = "./db/vehicledb.sqlite"
        self.db_path = os.path.join(os.getcwd(), "chroma_db")
        self.setup_chroma_db()
        
    def setup_chroma_db(self):
        try:
            chroma_client = chromadb.PersistentClient(path=self.db_path)
            chroma_client.delete_collection(name='lookup_ext')
        except Exception as e:
            print(f"DB deletion error (ignored): {e}")

class VectorStore:
    def __init__(self, model_config: ModelConfig, db_config: DatabaseConfig):
        self.model_config = model_config
        self.db_config = db_config
        self.setup_vector_store()
        
    def setup_vector_store(self):
        faq_text = 'db/vehicle_lookup.txt'
        with open(faq_text, "r", encoding="utf-8") as file:
            faq_text = file.read()
            
        raw_chunks = re.split(r"(?=\n\d+\.\s)", faq_text)
        docs = [Document(page_content=txt.strip()) for txt in raw_chunks]
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=30, length_function=len
        )
        split_docs = text_splitter.split_documents(docs)
        
        self.vector_store = Chroma.from_documents(
            split_docs,
            embedding=self.model_config.embedding_model,
            persist_directory=self.db_config.db_path,
            collection_name='lookup_text',
        )
        self.retriever = self.vector_store.as_retriever()

class Tools:
    def __init__(self, model_config: ModelConfig, db_config: DatabaseConfig, vector_store: VectorStore):
        self.model_config = model_config
        self.db_config = db_config
        self.vector_store = vector_store
        self.MINOR_ISSUES = ["flat tire", "won't start", "headlight", "oil change", "brake squeak"]
        
    @tool
    def lookup_recommendations(self, query: str) -> str:
        """See some suggestions of how to recommend."""
        docs = self.vector_store.retriever.invoke(query, k=1)
        return "\n\n".join([doc.page_content for doc in docs])
    
    def read_sql_query(self, sql: str, db: str):
        conn = sqlite3.connect(db)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        conn.commit()
        conn.close()
        for row in rows:
            print(row)
        return rows
    
    @tool
    def query_database(self, input_str: str) -> str:
        """Converts natural language questions to SQL queries and executes them."""
        question, start_date, end_date = input_str.split(";")
        
        prompt = f"""
        You are an expert in converting English questions to SQL queries!
        The SQL database is named 'vehicles' and has a table called 'bookings' with the following columns:

        vehicle_id, vehicle_name, vehicle_type, vehicle_wheel_count, vehicle_gear_type, 
        vehicle_price_per_day, vehicle_price_per_hour, vehicle_description, vehicle_mileage, 
        vehicle_fuel_type, vehicle_transmission, vehicle_seating_capacity, vehicle_user_rating, 
        vehicle_usage, vehicle_user_feedback

        Examples:

        Question: How many 4-wheelers are available?
        SQL: SELECT COUNT(*) FROM bookings WHERE vehicle_wheel_count = 4;

        Question: Show me all bikes under 1000 rupees per day
        SQL: SELECT * FROM bookings WHERE vehicle_type = 'Bike' AND vehicle_price_per_day < 1000;

        Now, based on this question, write only the SQL query. Do not include any explanation or formatting.
        Question: {question}
        """
        
        sql_query = self.model_config.chat_llm.invoke(prompt)
        print(f"Generated SQL: {sql_query}")
        print(sql_query.content)
        return sql_query
    
    def classify_issue(self, description: str) -> str:
        for issue in self.MINOR_ISSUES:
            if issue in description.lower():
                return "minor"
        return "major"
    
    @tool
    def vehicle_repair_assist(self, input_str: str) -> str:
        """Processes vehicle repair requests."""
        try:
            description, destination, vehicle_name = input_str.split(";")
        except ValueError:
            return "Invalid format. Use: 'issue;destination;vehicle_name'"
        
        issue_type = self.classify_issue(description)
        
        if issue_type == "minor":
            return (
                f"This looks like a minor issue {description}.\n"
                "You can try checking this guide:\n"
                f"{TavilySearchResults(k=1,).invoke(f'How to fix {description} . The vehicle is {vehicle_name}')}"
            )
        else:
            return (
                f"This might be a major issue : {description}.\n"
                "You can call us at +9199988822221, or find help nearby:\n"
                f"{TavilySearchResults(k=3).invoke(f'Mechanics between Delhi and {destination} for  fix')}"
            )
    
    def generate_updated_review(self, previous_review: str, new_review: str):
        prompt = ChatPromptTemplate.from_template(
        """
            You are a helpful assistant that combines and refines customer reviews. 
            The goal is to generate a new, well-written review that reflects both the previous and the updated feedback. 
            Preserve the tone, address both positive and negative points, and make it sound natural and honest.Try to fit in maximum 2 lines.

            Previous review: "{previous_review}"
            New review: "{new_review}"
            """
        )
        chain = prompt | self.model_config.chat_llm
        return chain.invoke({
            "previous_review": previous_review,
            "new_review": new_review
        }).content
    
    @tool
    def give_feedback(self, input_str: str):
        """Updates the rating and review of a vehicle."""
        try:
            rating_str, review, vehicleId = input_str.split(";", 2)
            rating = int(rating_str)
        except ValueError:
            return "Invalid input format. Use: 'rating;review;vehicleId'"
            
        conn = sqlite3.connect(self.db_config.local_file)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT vehicle_user_rating, vehicle_usage,vehicle_user_feedback FROM vehicles WHERE vehicle_id = ?",
            (vehicleId,)
        )
        previous_data = cursor.fetchone()
        old_rating, n, old_feedback = previous_data
        
        new_rating = (rating + old_rating*n)/(n+1)
        new_feedback = self.generate_updated_review(old_feedback, review)
        
        cursor.execute(
            """
            UPDATE vehicles
            SET vehicle_user_rating = ?, vehicle_usage = ?,vehicle_user_feedback = ?
            WHERE vehicle_id = ?
            """,
            (new_rating, n + 1, new_feedback, vehicleId)
        )
        
        conn.commit()
        conn.close()
    
    @tool
    def store_chatbot_feedback(self, input_str: str) -> str:
        """Stores user feedback about the chatbot's performance."""
        try:
            parts = input_str.split(';', 2)
            conversation_id = parts[0]
            rating = float(parts[1])
            feedback = parts[2] if len(parts) > 2 else ''
        except (IndexError, ValueError):
            return "Invalid input format. Use: 'conversation_id;rating;feedback' with rating as a float."
            
        conn = sqlite3.connect(self.db_config.local_file)
        cursor = conn.cursor()
        
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS chatbot_feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id TEXT NOT NULL,
                rating REAL CHECK(rating >= 1 AND rating <= 5),
                feedback TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            """
        )
        
        cursor.execute(
            """
            INSERT INTO chatbot_feedback (conversation_id, rating, feedback)
            VALUES (?, ?, ?)
            """,
            (conversation_id, rating, feedback)
        )
        conn.commit()
        conn.close()
        
        return "Thanks for rating our chatbot! Your feedback helps us improve."

class Agent:
    def __init__(self, model_config: ModelConfig, tools: Tools):
        self.model_config = model_config
        self.tools = tools
        self.MAX_TOOL_CALLS = 3
        


        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """
                You are a helpful and conversational customer support assistant for a Vehicle Rental Company located in Kashmiri Gate, Delhi.

                Your job is to assist users with:
                - Recommending rental vehicles.
                - Answering queries using available tools.
                - Helping with repair-related issues or feedback.

                Tool Usage Guidelines:
                - Use tools like `lookup_recommendations`, `query_database`, `vehicle_repair_assist`, or `give_feedback` **only when the user asks a specific question that requires information or action.**
                - Do **not** use any tools for casual greetings or small talk (e.g., "hello", "how are you").
                - If a tool returns no results, try expanding the search intelligently before replying.
                - If the user asks to speak to the rental company, give them the number `+9112233445566` and politely end the chat.

                Conversational Style:
                - Be friendly, concise, and human-like.
                - Responses should be around **30–40 tokens**.
                - Remember and refer to earlier context naturally in replies.

                Always close with:  
                "How would you like to rate the chatbot today?"

                Current conversation:"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        self.limited_prompt = ChatPromptTemplate.from_messages([
            ("ai", """You are a helpful AI assistant. You have reached your tool usage limit for this conversation.
                Please answer the question using your existing knowledge without using any tools.Be conversational and provide the best answer you can from your training data.

Current conversation:"""),
            MessagesPlaceholder(variable_name="messages"),
        ])
        
        self.tools_list = [
            self.tools.lookup_recommendations,
            self.tools.query_database,
            self.tools.vehicle_repair_assist,
            self.tools.give_feedback,
            self.tools.store_chatbot_feedback
        ]
        
        self.chat_llm_with_tools = self.model_config.chat_llm.bind_tools(self.tools_list)
        sel
        
    def filter_chat_messages(self, messages):
        """Filters out only HumanMessage and AIMessage objects from the full message list."""
        return [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    
    def agent_node(self, state: State):
        """The main agent that decides what to do next"""
        print(" Thinking...")
        
        current_count = state.get('tool_call_count', 0)
        
        if current_count >= self.MAX_TOOL_CALLS:
            print("🚫 Tool limit reached - using limited prompt")
            formatted_messages = self.limited_prompt.format_messages(messages=state["messages"])
            response = self.model_config.chat_llm.invoke(formatted_messages)
        else:
            print("✅ Tools available - using normal prompt")
            formatted_messages = self.prompt.format_messages(messages=state["messages"])
            response = self.chat_llm_with_tools.invoke(formatted_messages)
        
        print(f"🤖 AGENT RESPONSE: {response.content[:100]}...")
        
        new_count = current_count
        if hasattr(response, 'tool_calls') and response.tool_calls:
            new_count = current_count + 1
            print(f"🔧 Tool call detected! Incrementing count to: {new_count}")
        
        return {
            "messages": [response],
            "tool_call_count": new_count
        }
    
    def should_continue(self, state: State) -> Literal["tools", "end"]:
        """Determine if we should use tools or end the conversation"""
        last_message = state["messages"][-1]
        current_count = state.get('tool_call_count', 0)
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            if current_count <= self.MAX_TOOL_CALLS:
                print("🔧 ROUTING: Going to tools")
                return "tools"
            else:
                print("🚫 ROUTING: Tool limit exceeded, ending conversation")
                return "end"
        else:
            print("🏁 ROUTING: Ending conversation")
            return "end"
    
    def create_custom_agent(self):
        """Build the custom graph"""
        workflow = StateGraph(State)
        
        workflow.add_node("agent", self.agent_node)
        workflow.add_node("tools", ToolNode(self.tools_list))
        
        workflow.add_edge(START, "agent")
        
        workflow.add_conditional_edges(
            "agent",
            self.should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        workflow.add_edge("tools", "agent")
        
        return workflow


import streamlit as st
from main import ModelConfig, DatabaseConfig, VectorStore, Tools, Agent
from langchain_core.messages import HumanMessage, AIMessage
import uuid

def initialize_session_state():
    """Initialize session state variables if they don't exist."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'agent_graph' not in st.session_state:
        # Initialize configurations
        model_config = ModelConfig()
        db_config = DatabaseConfig()
        
        # Setup vector store
        vector_store = VectorStore(model_config, db_config)
        
        # Initialize tools
        tools = Tools(model_config, db_config, vector_store)
        
        # Create and setup agent
        agent = Agent(model_config, tools)
        workflow = agent.create_custom_agent()
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        st.session_state.agent_graph = workflow.compile(checkpointer=memory)
    if 'config' not in st.session_state:
        st.session_state.config = {"configurable": {"thread_id": str(uuid.uuid4())}}

def display_chat_history():
    """Display the chat history in a nice format."""
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            with st.chat_message("user"):
                st.write(message.content)
        elif isinstance(message, AIMessage):
            with st.chat_message("assistant"):
                st.write(message.content)

def process_user_input(user_input):
    """Process user input and get response from the agent."""
    # Add user message to chat history
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Get response from agent
    for step in st.session_state.agent_graph.stream(
        {"messages": [HumanMessage(content=user_input)]},
        st.session_state.config,
        stream_mode="values",
    ):
        if step["messages"]:
            ai_message = step["messages"][-1]
            st.session_state.messages.append(ai_message)
            with st.chat_message("assistant"):
                st.write(ai_message.content)

def main():
    st.title("Vehicle Rental Assistant")
    st.write("Welcome to the Vehicle Rental Assistant! How can I help you today?")
    
    # Initialize session state
    initialize_session_state()
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if user_input := st.chat_input("Type your message here..."):
        process_user_input(user_input)
        
        # Add rating section after each response
        with st.expander("Rate this response"):
            rating = st.slider("How helpful was this response?", 1, 5, 3)
            feedback = st.text_area("Additional feedback (optional)")
            if st.button("Submit Rating"):
                # Store feedback
                feedback_str = f"{st.session_state.config['configurable']['thread_id']};{rating};{feedback}"
                tools = Tools(ModelConfig(), DatabaseConfig(), VectorStore(ModelConfig(), DatabaseConfig()))
                tools.store_chatbot_feedback(feedback_str)
                st.success("Thank you for your feedback!")

if __name__ == "__main__":
    main()
