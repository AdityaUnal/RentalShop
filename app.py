from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from langchain_huggingface import HuggingFaceEmbeddings
import torch
from langchain_chroma import Chroma

import sqlite3
import re

from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages    
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
import uuid
import streamlit as st

from dotenv import load_dotenv
import os
load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
load_dotenv()
hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
llm_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
local_file = "./db/vehicledb.sqlite"


## Initiliazes a chat llm model and retriever
class llm_config:
    def __init__(self,llm_model_name:str):
        llm = HuggingFaceEndpoint(
            repo_id=llm_model_name,
            task="conversational",
            huggingfacehub_api_token = hf_api_key
        )
        self.chat_llm = ChatHuggingFace(llm=llm)
        
class retriever_config:
    def __init__(self, doc_location:str,embedding_model_name:str):
        with open(doc_location, "r", encoding="utf-8") as file:
            faq_text = file.read()

        # Split on lines that begin with a number followed by a dot (e.g., "1.")
        raw_chunks = re.split(r"(?=\n\d+\.\s)", faq_text)
        docs = [Document(page_content=txt.strip()) for txt in raw_chunks]


        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=100, chunk_overlap=30, length_function=len
        )
        split_docs = text_splitter.split_documents(docs)

        
        device = "cuda" if torch.cuda.is_available() else "cpu"

        embedding_model = HuggingFaceEmbeddings(
        model_name=embedding_model_name, model_kwargs={'device': device})
        # Create Chroma vector store
        vector_store = Chroma.from_documents(
            split_docs,
            embedding=embedding_model,
            persist_directory=os.path.join(os.getcwd(),"chroma_db"),
            collection_name='lookup_text',
        )
        self.retriever = vector_store.as_retriever()

    
llm_chat = llm_config(llm_model_name=llm_model_name).chat_llm
retriever = retriever_config(doc_location=os.path.join(os.getcwd(),'db\\vehicle_lookup.txt'),embedding_model_name=embedding_model_name).retriever

@tool
def lookup_recommendations(query: str) -> str:
    """See some suggestions of how to recommend ."""
    docs = retriever.invoke(query, k = 1)

    return "\n\n".join([doc.page_content for doc in docs])

@tool
def store_chatbot_feedback(input_str: str) -> str:
    """
    Stores user feedback about the chatbot's performance.

    Args:
        input_str (str): Input string in the format 'conversation_id;rating;feedback'

    Returns:
        str: A thank-you message.
    """
    try:
        # Split input into up to three parts (feedback is optional)
        parts = input_str.split(';', 2)
        conversation_id = parts[0]
        rating = float(parts[1])
        feedback = parts[2] if len(parts) > 2 else ''
    except (IndexError, ValueError):
        return "Invalid input format. Use: 'conversation_id;rating;feedback' with rating as a float."

    conn = sqlite3.connect("db/vehicledb.sqlite")  # Replace with actual path
    cursor = conn.cursor()

    # Create table if not exists
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

    # Insert feedback
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

def read_sql_query(sql,db):
    conn=sqlite3.connect(db)
    cur=conn.cursor()
    cur.execute(sql)
    rows=cur.fetchall()
    conn.commit()
    conn.close()
    for row in rows:
        print(row)
    return rows


@tool
def query_database(input_str: str) -> str:
    """
    Converts natural language questions to SQL queries and executes them.
    Input format: "question;start_date;end_date" (dates as YYYY-MM-DD)
    Example: "Show available bikes;2025-06-11;2025-06-21"
    """
    # Parse input
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

    # invoke the LLM with the prompt string
    sql_query = chat_llm.invoke(prompt)
    print(f"Generated SQL: {sql_query}")
    print(sql_query.content)
    
    return sql_query

MINOR_ISSUES = ["flat tire", "won't start", "headlight", "oil change", "brake squeak"]

def classify_issue(description: str) -> str:
    for issue in MINOR_ISSUES:
        if issue in description.lower():
            return "minor"
    return "major"

@tool
def vehicle_repair_assist(input_str: str) -> str:
    """
    Processes vehicle repair requests in format: "issue;destination;vehicle_name".
    Example input: "Engine knocking;Agra;Royal Enfield Classic 350"
    """
    # Parse input with safety checks
    try:
        description, destination, vehicle_name = input_str.split(";")
    except ValueError:
        return "Invalid format. Use: 'issue;destination;vehicle_name'"
    
    issue_type = classify_issue(description)


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



def generate_updated_review(previous_review : str, new_review : str):
    prompt = ChatPromptTemplate.from_template(
    """
        You are a helpful assistant that combines and refines customer reviews. 
        The goal is to generate a new, well-written review that reflects both the previous and the updated feedback. 
        Preserve the tone, address both positive and negative points, and make it sound natural and honest.Try to fit in maximum 2 lines.

        Previous review: "{previous_review}"
        New review: "{new_review}"
        """
    )
    chain = prompt | chat_llm
    return chain.invoke({
        "previous_review": previous_review,
        "new_review": new_review
    }).content

@tool
def give_feedback(input_str: str):
    """
    Updates the rating and review of a vehicle based on vehicleId.
    Input format: "rating;review;vehicleId"
    Example: "5;Great mileage and comfort;V12345"
    """
    try:
        rating_str, review, vehicleId = input_str.split(";", 2)
        rating = int(rating_str)
    except ValueError:
        return "Invalid input format. Use: 'rating;review;vehicleId'"
    conn = sqlite3.connect(local_file)
    cursor = conn.cursor()

    
    cursor.execute(
    "SELECT vehicle_user_rating, vehicle_usage,vehicle_user_feedback FROM vehicles WHERE vehicle_id = ?",
        (vehicleId,)
    )
    previous_data = cursor.fetchone()
    old_rating, n, old_feedback = previous_data
    
    new_rating = (rating + old_rating*n)/(n+1);
    new_feedback = generate_updated_review(old_feedback,review)
    

    cursor.execute(
        """
        UPDATE vehicles
        SET vehicle_user_rating = ?, vehicle_usage = ?,vehicle_user_feedback = ?
        WHERE vehicle_id = ?
        """,
        (new_rating, n + 1,new_feedback, vehicleId)
    )

    conn.commit()
    conn.close()

tools = [
    lookup_recommendations,
    query_database,
    vehicle_repair_assist,
    give_feedback,
    store_chatbot_feedback
]
chat_llm = llm_chat.bind_tools(tools)

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Agent:    
    prompt = ChatPromptTemplate.from_messages([
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


    Current conversation:"""),
    MessagesPlaceholder(variable_name="messages"),
    ])

    def __init__(self):
        pass

    def filter_chat_messages(messages):
        """
        Filters out only HumanMessage and AIMessage objects from the full message list.
        Useful for feeding into prompt templates that expect a chat history.
        """
        return [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]
    @staticmethod
    def agent_node(state: State):
        """The main agent that decides what to do next"""
        print(" Thinking...")
        
        # Format messages with prompt
        print("Tools available - using normal prompt")
        
        # Ensure messages alternate between user and assistant
        formatted_messages = []
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        response = chat_llm.invoke(formatted_messages)
        
        print(f"AGENT RESPONSE: {response.content[:100]}...")

        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"Tool call detected! ")
        
        return {
            "messages": [response],
        }

    def filter_chat_messages(messages):
        """
        Filters out only HumanMessage and AIMessage objects from the full message list.
        Useful for feeding into prompt templates that expect a chat history.
        """
        return [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]

    @staticmethod
    def should_continue(state: State) -> Literal["tools", "end"]:
        """Determine if we should use tools or end the conversation"""
        last_message = state["messages"][-1]
        current_count = state.get('tool_call_count', 0)
        
        # If the last message has tool calls, use tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print("ROUTING: Going to tools")
            return "tools"
        else:
            # Add rating request before ending
            state["messages"].append(AIMessage(content="How would you like to rate the chatbot today?"))
            print("ROUTING: Ending conversation")
            return "end"
        


    def create_custom_agent():
        tool_node = ToolNode(tools)
        # Initialize the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("agent", Agent.agent_node)
        workflow.add_node("tools", tool_node)
        
        # Add edges
        workflow.add_edge(START, "agent")  # Start with agent
        
        # Add conditional edge from agent
        workflow.add_conditional_edges(
            "agent",
            Agent.should_continue,
            {
                "tools": "tools",
                "end": END
            }
        )
        
        # After tools, always go back to agent
        workflow.add_edge("tools", "agent")
        
        return workflow

class CreateAgent:
    def __init__(self,id):
        workflow = Agent.create_custom_agent()
        memory = MemorySaver()
        self.agent_graph = workflow.compile(checkpointer=memory)
        self.config = {"configurable": {"thread_id": id}}

def test_message_formatting():
    """Test function to debug message formatting and model interaction"""
    # Test case 1: Basic conversation
    test_messages = [
        HumanMessage(content="Hello, I need a bike rental"),
        AIMessage(content="I can help you with that. What type of bike are you looking for?"),
        HumanMessage(content="I want a sports bike")
    ]
    
    print("\nTest Case 1: Basic conversation")
    formatted = []
    for msg in test_messages:
        if isinstance(msg, HumanMessage):
            formatted.append({"role": "user", "content": msg.content})
        elif isinstance(msg, AIMessage):
            formatted.append({"role": "assistant", "content": msg.content})
    print("Formatted messages:", formatted)
    
    # Test case 2: Single message
    print("\nTest Case 2: Single message")
    single_msg = [HumanMessage(content="Hello")]
    formatted_single = [{"role": "user", "content": single_msg[0].content}]
    print("Single message format:", formatted_single)
    
    # Test case 3: Model interaction
    print("\nTest Case 3: Testing model with single message")
    try:
        response = chat_llm.invoke(formatted_single)
        print("Model response:", response)
    except Exception as e:
        print("Model error:", str(e))

def main():
    # Run tests first
    test_message_formatting()
    
    st.title("Vehicle Rental Shop")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Hello Traveller!"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Create new agent instance
        id = str(uuid.uuid4())
        agent, config = CreateAgent(id).agent_graph,CreateAgent(id).config
        
        # Get response from agent
        with st.chat_message("assistant"):
            response = agent.invoke({"messages": [HumanMessage(content=prompt)]}, config=config)
            assistant_response = response["messages"][-1].content
            print(assistant_response)
            st.markdown(assistant_response)
            
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

if __name__ == "__main__":
    main()