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
            huggingfacehub_api_token = hf_api_key,
            max_new_tokens= 1000,
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
    """See some suggestions of how to recommend vehicles."""
    try:
        docs = retriever.invoke(query, k = 1)
        if not docs:
            raise Exception("No recommendations found")
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        raise Exception(f"Error in lookup_recommendations: {str(e)}")

@tool
def query_database(input_str: str) -> str:
    """Converts natural language questions to SQL queries and executes them."""
    try:
        # Parse input
        parts = input_str.split(";")
        if len(parts) != 3:
            raise Exception("Invalid input format. Please use: 'question;start_date;end_date'")
        
        question, start_date, end_date = parts

        # Validate dates if provided
        if start_date != "none" and end_date != "none":
            try:
                from datetime import datetime
                datetime.strptime(start_date, "%Y-%m-%d")
                datetime.strptime(end_date, "%Y-%m-%d")
            except ValueError:
                raise Exception("Invalid date format. Please use YYYY-MM-DD format.")

        prompt = f"""
        Given the following question about vehicle rentals, generate a SQL query to answer it.
        Question: {question}
        Start Date: {start_date}
        End Date: {end_date}
        
        Generate a SQL query that:
        1. Uses the vehicles table
        2. Considers the date range if provided
        3. Returns relevant information
        """
        
        sql_query = chat_llm.invoke(prompt)
        if not sql_query or not sql_query.content:
            raise Exception("Could not generate a valid SQL query")
        
        return sql_query.content
    except Exception as e:
        raise Exception(f"Error in query_database: {str(e)}")

@tool
def vehicle_repair_assist(input_str: str) -> str:
    """Processes vehicle repair requests in format: "issue;destination;vehicle_name"."""
    try:
        # Parse input with safety checks
        parts = input_str.split(";")
        if len(parts) != 3:
            raise Exception("Invalid format. Please use: 'issue;destination;vehicle_name'")
        
        description, destination, vehicle_name = parts
        
        if not all([description, destination, vehicle_name]):
            raise Exception("Please provide all required information: issue, destination, and vehicle name")
        
        issue_type = classify_issue(description)

        if issue_type == "minor":
            search_results = TavilySearchResults(k=1).invoke(f'How to fix {description} . The vehicle is {vehicle_name}')
            return (
                f"This looks like a minor issue: {description}.\n"
                "You can try checking this guide:\n"
                f"{search_results}"
            )
        else:
            search_results = TavilySearchResults(k=3).invoke(f'Mechanics between Delhi and {destination} for fix')
            return (
                f"This might be a major issue: {description}.\n"
                "You can call us at +9199988822221, or find help nearby:\n"
                f"{search_results}"
            )
    except Exception as e:
        raise Exception(f"Error in vehicle_repair_assist: {str(e)}")

@tool
def give_feedback(input_str: str) -> str:
    """Updates the rating and review of a vehicle based on vehicleId."""
    try:
        # Parse input
        parts = input_str.split(";", 2)
        if len(parts) != 3:
            raise Exception("Invalid input format. Please use: 'rating;review;vehicleId'")
        
        rating_str, review, vehicleId = parts
        
        # Validate rating
        try:
            rating = int(rating_str)
            if not 1 <= rating <= 5:
                raise Exception("Rating must be between 1 and 5")
        except ValueError:
            raise Exception("Invalid rating. Please provide a number between 1 and 5")

        # Validate vehicleId
        if not vehicleId or not vehicleId.strip():
            raise Exception("Invalid vehicle ID. Please provide a valid vehicle ID")

        conn = sqlite3.connect(local_file)
        cursor = conn.cursor()

        try:
            # Check if vehicle exists
            cursor.execute(
                "SELECT vehicle_user_rating, vehicle_usage, vehicle_user_feedback FROM vehicles WHERE vehicle_id = ?",
                (vehicleId,)
            )
            previous_data = cursor.fetchone()
            
            if not previous_data:
                raise Exception(f"No vehicle found with ID: {vehicleId}")

            old_rating, n, old_feedback = previous_data
            
            # Calculate new rating
            new_rating = (rating + old_rating * n) / (n + 1)
            new_feedback = generate_updated_review(old_feedback, review)
            
            # Update the database
            cursor.execute(
                """
                UPDATE vehicles
                SET vehicle_user_rating = ?, vehicle_usage = ?, vehicle_user_feedback = ?
                WHERE vehicle_id = ?
                """,
                (new_rating, n + 1, new_feedback, vehicleId)
            )
            conn.commit()
            return "Thank you for your feedback! It has been recorded successfully."
        except sqlite3.Error as e:
            raise Exception(f"Database error: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        raise Exception(f"Error in give_feedback: {str(e)}")

@tool
def store_chatbot_feedback(input_str: str) -> str:
    """Stores user feedback about the chatbot's performance."""
    try:
        # Split input into parts
        parts = input_str.split(';', 2)
        if len(parts) < 2:
            raise Exception("Invalid input format. Use: 'conversation_id;rating;feedback' with rating as a float")

        conversation_id = parts[0]
        try:
            rating = float(parts[1])
            if not 1 <= rating <= 5:
                raise Exception("Rating must be between 1 and 5")
        except ValueError:
            raise Exception("Invalid rating. Please provide a number between 1 and 5")

        feedback = parts[2] if len(parts) > 2 else ''

        conn = sqlite3.connect("db/vehicledb.sqlite")
        cursor = conn.cursor()

        try:
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
            return "Thanks for rating our chatbot! Your feedback helps us improve."
        except sqlite3.Error as e:
            raise Exception(f"Database error: {str(e)}")
        finally:
            conn.close()
    except Exception as e:
        raise Exception(f"Error in store_chatbot_feedback: {str(e)}")

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

MINOR_ISSUES = ["flat tire", "won't start", "headlight", "oil change", "brake squeak"]

def classify_issue(description: str) -> str:
    for issue in MINOR_ISSUES:
        if issue in description.lower():
            return "minor"
    return "major"

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
        
        # Get the last message
        last_message = state["messages"][-1]
        
        # If the last message was a tool response, format it properly
        if hasattr(last_message, 'tool_calls'):
            # Only include the original user message
            formatted_messages = [
                {"role": "user", "content": state["messages"][0].content}
            ]
        else:
            # For normal conversation, include all messages except system messages
            formatted_messages = []
            for msg in state["messages"]:
                if isinstance(msg, dict):
                    if msg.get("role") != "system":
                        formatted_messages.append(msg)
                elif isinstance(msg, HumanMessage):
                    formatted_messages.append({"role": "user", "content": msg.content})
                elif isinstance(msg, AIMessage):
                    formatted_messages.append({"role": "assistant", "content": msg.content})
        
        # Add system message at the start
        formatted_messages.insert(0, {
            "role": "system",
            "content": """ are a helpful and conversational customer support assistant for a Vehicle Rental Company located in Kashmiri Gate, Delhi.

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
    - Remember and refer to earlier context naturally in replies."""
        })
        
        try:
            response = chat_llm.invoke(formatted_messages)
            print(f"AGENT RESPONSE: {response.content[:100]}...")

            if hasattr(response, 'tool_calls') and response.tool_calls:
                print(f"Tool call detected! ")
                try:
                    # Store the tool call in state to prevent loops
                    state['last_tool_call'] = response.tool_calls[0].get('id')
                    return {"messages": [response]}
                except Exception as tool_error:
                    print(f"Error in tool execution: {str(tool_error)}")
                    # Get the user's query from the last message
                    user_query = state["messages"][-1].content if isinstance(state["messages"][-1], HumanMessage) else state["messages"][-1]["content"]
                    
                    # Use Tavily search as fallback
                    search_results = TavilySearchResults(k=3).invoke(f"vehicle rental {user_query}")
                    
                    # Create a helpful response with the search results
                    error_response = AIMessage(content=(
                        "I'm having trouble processing your request directly, but I found some relevant information:\n\n"
                        f"{search_results}\n\n"
                        "Would you like to try asking your question in a different way?"
                    ))
                    return {"messages": [error_response]}
            
            return {"messages": [response]}
        except Exception as e:
            print(f"Error in agent_node: {str(e)}")
            try:
                # Get the user's query from the last message
                user_query = state["messages"][-1].content if isinstance(state["messages"][-1], HumanMessage) else state["messages"][-1]["content"]
                
                # Use Tavily search as fallback
                search_results = TavilySearchResults(k=3).invoke(f"vehicle rental {user_query}")
                print("yaay ",type(search_results))
                print("naay",search_results)
                # Create a helpful response with the search results
                error_response = AIMessage(content=(
                    "I'm having trouble processing your request directly, but I found some relevant information:\n\n"
                    f"{search_results}\n\n"
                    "Would you like to try asking your question in a different way?"
                ))
                return {"messages": [error_response]}
            except Exception as search_error:
                print(f"Error in fallback search: {str(search_error)}")
                # If even the fallback fails, return a basic error message
                return {"messages": [AIMessage(content="I apologize, but I'm having trouble processing your request right now. Please try again in a moment.")]}

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
        
        # If the last message has tool calls, use tools
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            # Check if we've already processed this tool call
            if state.get('last_tool_call') == last_message.tool_calls[0].get('id'):
                print("ROUTING: Ending conversation - tool already processed")
                return "end"
            
            # Store the current tool call ID
            state['last_tool_call'] = last_message.tool_calls[0].get('id')
            print("ROUTING: Going to tools")
            return "tools"
        else:
            # If no tool calls, end the conversation
            print("ROUTING: Ending conversation - no tool calls")
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
        {"role": "user", "content": "Hello, I need a bike rental"},
        {"role": "assistant", "content": "I can help you with that. What type of bike are you looking for?"},
        {"role": "user", "content": "I want a sports bike"}
    ]
    
    print("\nTest Case 1: Basic conversation")
    print("Formatted messages:", test_messages)
    
    # Test case 2: Single message
    print("\nTest Case 2: Single message")
    single_msg = [{"role": "user", "content": "Hello"}]
    print("Single message format:", single_msg)
    
    # Test case 3: Model interaction
    print("\nTest Case 3: Testing model with single message")
    try:
        response = chat_llm.invoke(single_msg)
        print("Model response:", response)
        if hasattr(response, 'tool_calls'):
            print("Tool calls:", response.tool_calls)
    except Exception as e:
        print("Model error:", str(e))

def main():
    # Run tests first
    test_message_formatting()
    
    st.title("Vehicle Rental Shop")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history
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
        agent, config = CreateAgent(id).agent_graph, CreateAgent(id).config
        
        # Get response from agent
        with st.chat_message("assistant"):
            try:
                response = agent.invoke({"messages": [{"role": "user", "content": prompt}]}, config=config)
                assistant_response = response["messages"][-1].content
                st.markdown(assistant_response)
                st.session_state.messages.append({"role": "assistant", "content": assistant_response})
            except Exception as e:
                st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()