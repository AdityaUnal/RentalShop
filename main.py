from langchain_huggingface.llms import HuggingFaceEndpoint
from langchain_huggingface import ChatHuggingFace
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import Annotated, Literal, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph.message import add_messages    
import uuid
import streamlit as st
from dotenv import load_dotenv
import os
load_dotenv()

hf_api_key = os.getenv('HUGGINGFACE_API_KEY')
llm_model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"

## Initialize chat llm model
class llm_config:
    def __init__(self,llm_model_name:str):
        llm = HuggingFaceEndpoint(
            repo_id=llm_model_name,
            task="conversational",
            huggingfacehub_api_token = hf_api_key,
            max_new_tokens= 1000,
            temperature=0.5
        )
        self.chat_llm = ChatHuggingFace(llm=llm)

llm_chat = llm_config(llm_model_name=llm_model_name).chat_llm

class State(TypedDict):
    messages: Annotated[list, add_messages]

class Agent:    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
    You are a helpful and conversational customer support assistant for a Vehicle Rental Company located in Kashmiri Gate, Delhi.

    Your job is to assist users with:
    - Recommending rental vehicles.
    - Answering queries about our services.
    - Helping with general inquiries.

    Guidelines:
    - You have Mahindra Scorpio(Car), Royal Enfield Classic 350(Bike), Honda Activa 6G(Activa), Maruti Swift(Car), Bajaj Pulsar 150(Bike),TVS Jupiter(Scooty)
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
        print("Thinking...")
        
        # Format messages for the chat
        formatted_messages = [
            {"role": "system", "content": """You are a helpful and conversational customer support assistant for a Vehicle Rental Company located in Kashmiri Gate, Delhi.

            Your job is to assist users with:
            - Recommending rental vehicles.
            - Answering queries about our services.
            - Helping with general inquiries.

            Guidelines:
            - You have Mahindra Scorpio(Car), Royal Enfield Classic 350(Bike), Honda Activa 6G(Activa), Maruti Swift(Car), Bajaj Pulsar 150(Bike),TVS Jupiter(Scooty)
            - If the user asks to speak to the rental company, give them the number `+9112233445566` and politely end the chat.

            Conversational Style:
            - Be friendly, concise, and human-like.
            - Responses should be around **30–40 tokens**.
            - Remember and refer to earlier context naturally in replies."""}
        ]
        
        for msg in state["messages"]:
            if isinstance(msg, dict):
                if msg.get("role") != "system":
                    formatted_messages.append(msg)
            elif isinstance(msg, HumanMessage):
                formatted_messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted_messages.append({"role": "assistant", "content": msg.content})
        
        try:
            response = llm_chat.invoke(formatted_messages)
            print(f"AGENT RESPONSE: {response.content[:100]}...")
            return {"messages": [response]}
        except Exception as e:
            print(f"Error in agent_node: {str(e)}")
            return {"messages": [AIMessage(content="I apologize, but I'm having trouble processing your request right now. Please try again in a moment.")]}

    @staticmethod
    def should_continue(state: State) -> Literal["end"]:
        """Determine if we should end the conversation"""
        return "end"

    def create_custom_agent():
        # Initialize the graph
        workflow = StateGraph(State)
        
        # Add nodes
        workflow.add_node("agent", Agent.agent_node)
        
        # Add edges
        workflow.add_edge(START, "agent")  # Start with agent
        workflow.add_edge("agent", END)  # End after agent response
        
        return workflow

class CreateAgent:
    def __init__(self,id):
        workflow = Agent.create_custom_agent()
        self.agent_graph = workflow.compile()
        self.config = {"configurable": {"thread_id": id}}

def main():
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