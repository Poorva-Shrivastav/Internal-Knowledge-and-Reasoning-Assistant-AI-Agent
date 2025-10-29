import os 
from dotenv import load_dotenv
from agents.agent_tools import search_hr_docs, search_confluence_docs
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver

import streamlit as st

load_dotenv()

llm = init_chat_model("google_genai:gemini-2.0-flash")

st.title("SearchIQ")

st.markdown("**:rainbow[Smart internal search engine]**")
            
st.caption("Currently I'm fed with data on OAuth2 : [Confluence API](https://poorvashrivastav03.atlassian.net/wiki/spaces/TRD/overview) and HR policy document.")

memory = MemorySaver()

@tool("summarize_docs", return_direct=False)
def summarize_docs(text: str):
    """Summarize retrieved content for quick reference."""    
    return llm.invoke(f"Summarize briefly:\n\n{text}").content


tools = [search_hr_docs, search_confluence_docs]

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages : Annotated[List, add_messages]

def reasoning_agent(state: State):
    """LLM decides whether to answer or call a tool."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": messages + [response]}


builder = StateGraph(State)

builder.add_node("reasoning_agent_node", reasoning_agent)
builder.add_node("tools", ToolNode(tools))

builder.set_entry_point("reasoning_agent_node")
builder.add_conditional_edges("reasoning_agent_node", tools_condition)
builder.add_edge("tools", "reasoning_agent_node")

graph = builder.compile(checkpointer=memory)

config = {'configurable' : {'thread_id' : 1}}

# if "messages" not in st.session_state:
#     st.session_state.messages = []
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "I'm here to help you. Let's begin!"}]

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("What's up?")
# full_response = ""

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

    response = graph.invoke({
    "messages": [HumanMessage(content=prompt)]
    }, config=config)
    
    if response["messages"][-1].content:
        full_response = response["messages"][-1].content        

    message_placeholder.markdown(response["messages"][-1].content)

    st.session_state.messages.append({"role": "assistant", "content": full_response})