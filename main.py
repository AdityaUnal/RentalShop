# %% [markdown]
# # Basic Chatbot
# - This is a simple chatbot that answers basic questions
# - It is also able to do basic tasks
# 

# %%
from dotenv import load_dotenv
import os
load_dotenv()

tavily_api_key = os.getenv('TAVILY_API_KEY')
load_dotenv()
hf_api_key = os.getenv('HUGGINGFACE_API_KEY')

# %%
from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace

from langchain_huggingface import HuggingFaceEmbeddings
import torch

embedding_model_name = 'sentence-transformers/all-MiniLM-L6-v2'
device = "cuda" if torch.cuda.is_available() else "cpu"

embedding_model = HuggingFaceEmbeddings(
model_name=embedding_model_name, model_kwargs={'device': device})
max_length = embedding_model._client.tokenizer.model_max_length - 50

llm_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
llm = HuggingFaceEndpoint(
    repo_id=llm_model_name,
    task="conversational",
    huggingfacehub_api_token = hf_api_key
)
chat_llm = ChatHuggingFace(llm=llm)

# %%
from langchain_core.messages import HumanMessage, AIMessage

messages = [
    HumanMessage(content="Hello!"),
    AIMessage(content="Hi! How can I assist you?"),
    HumanMessage(content="Tell me about your services.")
]
print(chat_llm.invoke(messages))


# %% [markdown]
# ## Populating the database
# - The following script populates the sqlite database for the vehicle information
# - I have hard-coded vehicle information but a simple python block statment can allow the user to do the same : 
# ```python
#     while True:
#         again = input("Do you want to add a vehicle? (yes/no): ").strip().lower()
#         if again != "yes":
#             break
#         vehicle_data = get_vehicle_data()
#         add_vehicle(local_file,vehicle_data)
# ```
# - I have also taken the liberty for manually writing the bookings.

# %% [markdown]
# ### Populating Vehicle database

# %%
from datetime import date, datetime, timedelta

import sqlite3

import pandas as pd

# %%
sample_vehicles = [
    {
        'vehicle_id': 'SCORPIO001',
        'vehicle_name': 'Mahindra Scorpio',
        'vehicle_type': 'Car',
        'vehicle_wheel_count': 4,
        'vehicle_gear_type': '6-Speed Manual',
        'vehicle_price_per_day': 5000.0,
        'vehicle_price_per_hour': 500.0,
        'vehicle_description': 'Powerful SUV good for all types of terrain including hills and rough roads',
        'vehicle_mileage': 12.0,
        'vehicle_fuel_type': 'Diesel',
        'vehicle_transmission': 'Manual',
        'vehicle_seating_capacity': 7,
        'vehicle_user_rating': 4.5,
        'vehicle_usage': 2,
        'vehicle_user_feedback': 'Great for family trips and adventure rides'
    },
    {
        'vehicle_id': 'RE_CLASSIC_001',
        'vehicle_name': 'Royal Enfield Classic 350',
        'vehicle_type': 'Bike',
        'vehicle_wheel_count': 2,
        'vehicle_gear_type': '5-Speed Manual',
        'vehicle_price_per_day': 1500.0,
        'vehicle_price_per_hour': 100.0,
        'vehicle_description': 'Classic motorcycle good for long rides and heavy loads',
        'vehicle_mileage': 35.0,
        'vehicle_fuel_type': 'Petrol',
        'vehicle_transmission': 'Manual',
        'vehicle_seating_capacity': 2,
        'vehicle_user_rating': 4.2,
        'vehicle_usage': 8,
        'vehicle_user_feedback': 'Classic thumping sound and comfortable for long rides'
    },
    {
        'vehicle_id': 'ACTIVA_001',
        'vehicle_name': 'Honda Activa 6G',
        'vehicle_type': 'Scooty',
        'vehicle_wheel_count': 2,
        'vehicle_gear_type': 'Non-Gear',
        'vehicle_price_per_day': 800.0,
        'vehicle_price_per_hour': 60.0,
        'vehicle_description': 'Perfect for city rides and nearby locations',
        'vehicle_mileage': 50.0,
        'vehicle_fuel_type': 'Petrol',
        'vehicle_transmission': 'Automatic',
        'vehicle_seating_capacity': 2,
        'vehicle_user_rating': 4.3,
        'vehicle_usage': 90,
        'vehicle_user_feedback': 'Very fuel efficient and easy to ride'
    },
    {
        'vehicle_id': 'SWIFT_001',
        'vehicle_name': 'Maruti Swift',
        'vehicle_type': 'Car',
        'vehicle_wheel_count': 4,
        'vehicle_gear_type': '5-Speed Manual',
        'vehicle_price_per_day': 2500.0,
        'vehicle_price_per_hour': 200.0,
        'vehicle_description': 'Compact car perfect for city driving and small families',
        'vehicle_mileage': 22.0,
        'vehicle_fuel_type': 'Petrol',
        'vehicle_transmission': 'Manual',
        'vehicle_seating_capacity': 5,
        'vehicle_user_rating': 4.1,
        'vehicle_usage': 43,
        'vehicle_user_feedback': 'Good mileage and comfortable for city rides'
    },
    {
        'vehicle_id': 'PULSAR_001',
        'vehicle_name': 'Bajaj Pulsar 150',
        'vehicle_type': 'Bike',
        'vehicle_wheel_count': 2,
        'vehicle_gear_type': '5-Speed Manual',
        'vehicle_price_per_day': 1000.0,
        'vehicle_price_per_hour': 80.0,
        'vehicle_description': 'Sporty bike good for daily commuting and weekend rides',
        'vehicle_mileage': 45.0,
        'vehicle_fuel_type': 'Petrol',
        'vehicle_transmission': 'Manual',
        'vehicle_seating_capacity': 2,
        'vehicle_user_rating': 4.0,
        'vehicle_usage': 40,
        'vehicle_user_feedback': 'Good performance and stylish design'
    },
    {
        'vehicle_id': 'JUPITER_001',
        'vehicle_name': 'TVS Jupiter',
        'vehicle_type': 'Scooty',
        'vehicle_wheel_count': 2,
        'vehicle_gear_type': 'Non-Gear',
        'vehicle_price_per_day': 700.0,
        'vehicle_price_per_hour': 50.0,
        'vehicle_description': 'Reliable scooter with good storage space',
        'vehicle_mileage': 55.0,
        'vehicle_fuel_type': 'Petrol',
        'vehicle_transmission': 'Automatic',
        'vehicle_seating_capacity': 2,
        'vehicle_user_rating': 4.2,
        'vehicle_usage': 10,
        'vehicle_user_feedback': 'Excellent mileage and comfortable seat'
    }
]

# %%
local_file = "./db/vehicledb.sqlite"


def add_vehicle(file, vehicle_data):
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS vehicles (
        vehicle_id TEXT PRIMARY KEY,
        vehicle_name TEXT,
        vehicle_type TEXT,
        vehicle_wheel_count INTEGER,
        vehicle_gear_type TEXT,
        vehicle_price_per_day REAL,
        vehicle_price_per_hour REAL,
        vehicle_description TEXT,
        vehicle_mileage REAL,
        vehicle_fuel_type TEXT,
        vehicle_transmission TEXT,
        vehicle_seating_capacity INTEGER,
        vehicle_user_rating REAL,
        vehicle_usage REAL,
        vehicle_user_feedback TEXT
    )
""")
    columns = ', '.join(vehicle_data.keys())
    placeholders = ', '.join(['?'] * len(vehicle_data))
    values = tuple(vehicle_data.values())
    cursor.execute(
        f"INSERT OR IGNORE INTO vehicles ({columns}) VALUES ({placeholders})",
        values
    )
    conn.commit()
    conn.close()


for vehicle_data in sample_vehicles:
    add_vehicle(local_file, vehicle_data)

# %% [markdown]
# If a vehicle is new then the column for user rating and user feedback will be shown to user as none. This makes sure no misleading values are given keeping the data consistent. The user feedback is updated by sentiment analysis by the LLM.

# %% [markdown]
# Displays the existing vehicles in the db

# %%
from tabulate import tabulate

conn = sqlite3.connect("db/vehicledb.sqlite")
df = pd.read_sql("SELECT * FROM vehicles", conn)
conn.close()

print(tabulate(df, headers='keys', tablefmt='psql'))  # or 'fancy_grid', 'grid'

# %% [markdown]
# ## Populating bookings

# %%
booking_data = [
    {
        "booking_id": "BKG_JSON_001",
        "customer_id": "CUST_J_001",
        "vehicle_id": "SCORPIO001",
        "booking_date": "2025-06-11",
        "start_date": "2025-06-15",
        "end_date": "2025-06-18",
        "booking_type": "Daily",
        "duration_value": 3,
        "duration_unit": "Days",
        "total_cost": 15000.00,
        "status": "Confirmed",
    },
    {
        "booking_id": "BKG_JSON_002",
        "customer_id": "CUST_J_002",
        "vehicle_id": "ACTIVA_001",
        "booking_date": "2025-06-11",
        "start_date": "2025-06-11",
        "end_date": "2025-06-11",
        "booking_type": "Hourly",
        "duration_value": 5,
        "duration_unit": "Hours",
        "total_cost": 300.00,
        "status": "Confirmed",
    }
]

# %%
local_file = "./db/vehicledb.sqlite"


def add_bookings(file, booking_data):
    conn = sqlite3.connect(file)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bookings (
        booking_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        vehicle_id TEXT NOT NULL,
        booking_date TEXT NOT NULL,
        start_date TEXT NOT NULL,
        end_date TEXT NOT NULL,
        booking_type TEXT NOT NULL,
        duration_value REAL NOT NULL,
        duration_unit TEXT NOT NULL,
        total_cost REAL NOT NULL,
        status TEXT NOT NULL,
        FOREIGN KEY (vehicle_id) REFERENCES vehicles(vehicle_id)
    )
    """)
    columns = ', '.join(booking_data.keys())
    placeholders = ', '.join(['?'] * len(booking_data))
    values = tuple(booking_data.values())
    cursor.execute(
        f"INSERT OR IGNORE INTO bookings ({columns}) VALUES ({placeholders})",
        values
    )
    conn.commit()
    conn.close()

for booking in booking_data:
    add_bookings(local_file, booking)

# %%
from tabulate import tabulate

conn = sqlite3.connect("db/vehicledb.sqlite")
df = pd.read_sql("SELECT * FROM bookings", conn)
conn.close()

print(tabulate(df, headers='keys', tablefmt='psql')) 

# %% [markdown]
# ## Tools
# - We define the tools of our chatbot
# - These tools will help in vehicle recommendation, repairs, user feedback of the vehicle

# %% [markdown]
# ### Vehicle Recommendation
# - Based on the user's travel destination, travel budget, weather, the feedback on vehicle.
# - I created a lookup text('.db/lookup_txt') which is stored in the vector store, for the llm to use it.
# - The lookup_txt can be edited to further refine the recommendations.

# %% [markdown]
# Deleting collection named 'lookup_text' if it already exists. This will be the naem of the collection where I am storing embeddings from new collection.

# %%
import chromadb
from langchain_chroma import Chroma


db_path = os.path.join(os.getcwd(), "chroma_db")
try:
    chroma_client = chromadb.PersistentClient(path=db_path)
    chroma_client.delete_collection(name='lookup_ext')
except Exception as e:
    print(f"DB deletion error (ignored): {e}")

# %% [markdown]
# Storing in vector store
# 

# %%
import re

from langchain_core.tools import tool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


faq_text = 'db/vehicle_lookup.txt'
with open(faq_text, "r", encoding="utf-8") as file:
    faq_text = file.read()

# Split on lines that begin with a number followed by a dot (e.g., "1.")
raw_chunks = re.split(r"(?=\n\d+\.\s)", faq_text)
docs = [Document(page_content=txt.strip()) for txt in raw_chunks]


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=100, chunk_overlap=30, length_function=len
)
split_docs = text_splitter.split_documents(docs)


# Create Chroma vector store
vector_store = Chroma.from_documents(
    split_docs,
    embedding=embedding_model,
    persist_directory=db_path,
    collection_name='lookup_text',
)
retriever = vector_store.as_retriever()

# %%
@tool
def lookup_recommendations(query: str) -> str:
    """See some suggestions of how to recommend ."""
    docs = retriever.invoke(query, k = 1)

    return "\n\n".join([doc.page_content for doc in docs])

# %% [markdown]
# #### Sample output for the tool that uses the lookup document

# %%
print(lookup_recommendations("I am travelling on a budget and the place is far"))

# %%
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

# %% [markdown]
# #### Sample of how the sql query tool will function

# %%
result = query_database("Show available bikes;2025-06-11;2025-06-21")


# %% [markdown]
# ### Repair Help

# %%
MINOR_ISSUES = ["flat tire", "won't start", "headlight", "oil change", "brake squeak"]

def classify_issue(description: str) -> str:
    for issue in MINOR_ISSUES:
        if issue in description.lower():
            return "minor"
    return "major"


# %%
from langchain_community.tools.tavily_search import TavilySearchResults


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


# %% [markdown]
# #### Sample output of the tool that provides breakdown assist

# %%
result = vehicle_repair_assist("The tyre has deflated;Jaipir;Scorpio")
print(result)


# %% [markdown]
# ### Updating review and feedback for vehicle

# %%
from langchain_core.prompts import ChatPromptTemplate


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
    

# %%
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
    

# %% [markdown]
# #### Sample for how the feedback tool works

# %%
conn = sqlite3.connect("db/vehicledb.sqlite")
df = pd.read_sql("SELECT * FROM vehicles", conn)
conn.close()

print("Before : ")
print(tabulate(df, headers='keys', tablefmt='psql')) 

# %%
give_feedback("5;The car is very god. I loved the experience;SCORPIO001")

# %%
conn = sqlite3.connect("db/vehicledb.sqlite")
df = pd.read_sql("SELECT * FROM vehicles", conn)
conn.close()

print("After : ")
print(tabulate(df, headers='keys', tablefmt='psql')) 

# %% [markdown]
# ### Updating review and feedback for chabot
# 

# %%
from langchain.tools import tool
import sqlite3
from datetime import datetime

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


# %%
store_chatbot_feedback("131;3")

# %%
conn = sqlite3.connect("db/vehicledb.sqlite")
df = pd.read_sql("SELECT * FROM chatbot_feedback", conn)
conn.close()

print("After : ")
print(tabulate(df, headers='keys', tablefmt='psql')) 

# %%


# %% [markdown]
# ## Agent
# - Next, defining the assistant function. This function takes the graph state, formats it into a prompt, and then calls an LLM for it to predict the best response.
# - The assistant helps call the graph with state & config.

# %%

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages    

MAX_TOOL_CALLS = 3


class State(TypedDict):
    messages: Annotated[list, add_messages]
    tool_call_count: int



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

Always close with:  
“How would you like to rate the chatbot today?”


    
Current conversation:"""),
    MessagesPlaceholder(variable_name="messages"),
])

limited_prompt = ChatPromptTemplate.from_messages([
    ("ai", """You are a helpful AI assistant. You have reached your tool usage limit for this conversation.
    
Please answer the question using your existing knowledge without using any tools.
Be conversational and provide the best answer you can from your training data.

Current conversation:"""),
    MessagesPlaceholder(variable_name="messages"),
])

tools = [
    lookup_recommendations,
    query_database,
    vehicle_repair_assist,
    give_feedback,
    store_chatbot_feedback
]


chat_llm_with_tools = chat_llm.bind_tools(tools)

def filter_chat_messages(messages):
    """
    Filters out only HumanMessage and AIMessage objects from the full message list.
    Useful for feeding into prompt templates that expect a chat history.
    """
    return [m for m in messages if isinstance(m, (HumanMessage, AIMessage))]


def agent_node(state: State):
    """The main agent that decides what to do next"""
    print(" Thinking...")
    
    current_count = state.get('tool_call_count', 0)
    
    # Format messages with prompt
    if current_count >= MAX_TOOL_CALLS:
        print("🚫 Tool limit reached - using limited prompt")
        formatted_messages = limited_prompt.format_messages(messages=state["messages"])
        response = chat_llm.invoke(formatted_messages)  # Use chat_llm without tools
    else:
        print("✅ Tools available - using normal prompt")
        formatted_messages = prompt.format_messages(messages=state["messages"])
        response = chat_llm_with_tools.invoke(formatted_messages)  # Use chat_llm with tools
    
    print(f"🤖 AGENT RESPONSE: {response.content[:100]}...")
    
    new_count = current_count
    if hasattr(response, 'tool_calls') and response.tool_calls:
        new_count = current_count + 1
        print(f"🔧 Tool call detected! Incrementing count to: {new_count}")
    
    return {
        "messages": [response],
        "tool_call_count": new_count
    }


def should_continue(state: State) -> Literal["tools", "end"]:
    """Determine if we should use tools or end the conversation"""
    last_message = state["messages"][-1]
    current_count = state.get('tool_call_count', 0)
    # If the last message has tool calls, use tools
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        if current_count <= MAX_TOOL_CALLS:
            print("🔧 ROUTING: Going to tools")
            return "tools"
        else:
            print("🚫 ROUTING: Tool limit exceeded, ending conversation")
            return "end"
    else:
        print("🏁 ROUTING: Ending conversation")
        return "end"

# %%
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

tool_node = ToolNode(tools)

# Build the custom graph
def create_custom_agent():
    # Initialize the graph
    workflow = StateGraph(State)
    
    # Add nodes
    workflow.add_node("agent", agent_node)
    workflow.add_node("tools", tool_node)
    
    # Add edges
    workflow.add_edge(START, "agent")  # Start with agent
    
    # Add conditional edge from agent
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")
    
    return workflow


workflow = create_custom_agent()
memory = MemorySaver()
agent_graph = workflow.compile(checkpointer=memory)

# Test with tool call limiting
config = {"configurable": {"thread_id": str(uuid.uuid4())}}

print(f"🎯 MAX_TOOL_CALLS set to: {MAX_TOOL_CALLS}")
print("=" * 50)

# %% [markdown]
# Displaying the general flow of the graph. Since all the operations are **read-only**, I did not add any conditions to access the tools.
# 

# %%
from IPython.display import Image, display

try:
    display(Image(agent_graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %% [markdown]
# ## Example Conversations

# %%
import uuid

user_questions = [
    "Hello",
    #  Before Renting - Vehicle Search & Comparison
    "Which cars are automatic and seat at least 5 people?",

    # Before Renting - Availability & Booking
    "Is the Mahindra Scorpio available this weekend?",

    # During Renting - Issues
    "The vehicle is making noise, what should I do?",

    "I want a stylish vehicle for a short city visit."
]

for question in user_questions:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    agent_graph.update_state(config, {"messages": []})
    for step in agent_graph.stream(
       {"messages":[HumanMessage(content=question)]},
        config,
        stream_mode="values",
    ):
        print(f"📝 Current messages count: {len(step['messages'])}")
        step["messages"][-1].pretty_print()
        print("-" * 30)
    agent_graph.update_state(config, {"messages": []})

    
    

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
from IPython.display import Image, display

try:
    display(Image(graph.get_graph(xray=True).draw_mermaid_png()))
except Exception:
    # This requires some extra dependencies and is optional
    pass

# %%
# from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
# from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
# from langgraph.checkpoint.memory import MemorySaver
# from langgraph.graph import StateGraph, MessagesState, START, END
# from langgraph.prebuilt import ToolNode
# import json
# from typing import Literal

# prompt = ChatPromptTemplate.from_messages([
#     ("system", """You are a helpful AI assistant. You have access to search tools to help answer questions.
    
# When you need to search for information, use the available tools.
# Be conversational and remember previous context from the conversation.
    
# Current conversation:"""),
#     MessagesPlaceholder(variable_name="messages"),
# ])
# chat_llm_with_tools = chat_llm.bind_tools(tools)
# # Define the agent node
# def agent_node(state: MessagesState):
#     """The main agent that decides what to do next"""
#     print("🤖 AGENT NODE: Thinking...")
    
#     # Format messages with prompt
#     formatted_messages = prompt.format_messages(messages=state["messages"])
    
#     # Get response from LLM
#     response = chat_llm_with_tools.invoke(formatted_messages)
    
#     print(f"🤖 AGENT RESPONSE: {response.content[:100]}...")
#     return {"messages": [response]}

# # Define conditional edge function
# def should_continue(state: MessagesState) -> Literal["tools", "end"]:
#     """Determine if we should use tools or end the conversation"""
#     last_message = state["messages"][-1]
    
#     # If the last message has tool calls, use tools
#     if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
#         print("🔧 ROUTING: Going to tools")
#         return "tools"
#     else:
#         print("🏁 ROUTING: Ending conversation")
#         return "end"

# # Create tool node
# tool_node = ToolNode(tools)

# # Build the custom graph
# def create_custom_agent():
#     # Initialize the graph
#     workflow = StateGraph(MessagesState)
    
#     # Add nodes
#     workflow.add_node("agent", agent_node)
#     workflow.add_node("tools", tool_node)
    
#     # Add edges
#     workflow.add_edge(START, "agent")  # Start with agent
    
#     # Add conditional edge from agent
#     workflow.add_conditional_edges(
#         "agent",
#         should_continue,
#         {
#             "tools": "tools",
#             "end": END
#         }
#     )
    
#     # After tools, always go back to agent
#     workflow.add_edge("tools", "agent")
    
#     return workflow

# # Create and compile the graph
# workflow = create_custom_agent()
# memory = MemorySaver()
# agent_graph = workflow.compile(checkpointer=memory)

# # Visualize the graph structure
# print("📊 GRAPH STRUCTURE:")
# print("START → AGENT → [TOOLS or END]")
# print("TOOLS → AGENT (loop back)")
# print("=" * 50)

# # Use the custom agent
# config = {"configurable": {"thread_id": "abc123"}}

# print("\n🚀 Starting conversation 1...")
for step in agent_graph.stream(
    {"messages": [HumanMessage(content="hi im bob! and i live in sf")]},
    config,
    stream_mode="values",
):
    print(f"📝 Current messages count: {len(step['messages'])}")
    step["messages"][-1].pretty_print()
    print("-" * 30)

# print("\n🚀 Starting conversation 2...")
# for step in agent_graph.stream(
#     {"messages": [HumanMessage(content="whats the weather where I live?")]},
#     config,
#     stream_mode="values",
# ):
#     print(f"📝 Current messages count: {len(step['messages'])}")
#     step["messages"][-1].pretty_print()
#     print("-" * 30)

# # Optional: Print the graph as mermaid diagram
# print("\n🎨 GRAPH VISUALIZATION (Mermaid):")
# try:
#     mermaid_graph = agent_graph.get_graph().draw_mermaid()
#     print(mermaid_graph)
# except:
#     print("Mermaid visualization not available")

# # Optional: Get graph information
# print("\n📋 GRAPH INFO:")
# graph_info = agent_graph.get_graph()
# print(f"Nodes: {list(graph_info.nodes.keys())}")
# print(f"Edges: {[(edge.source, edge.target) for edge in graph_info.edges]}")


