import os 
from dotenv import load_dotenv

from rag_chain.rag_chain import llm
from agents.agent_tools import search_hr_docs, search_confluence_docs
from langchain_core.tools import tool
# from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

from typing import TypedDict, Annotated, List, Optional
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import json
from uuid import uuid4

from langgraph.checkpoint.memory import MemorySaver

# load_dotenv()

# # langsmith_api_key = os.environ["LANGSMITH_API_KEY"]

# # client = Client(api_key=langsmith_api_key)

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

graph = builder.compile()
graph = builder.compile(checkpointer=memory)

app = FastAPI()

# Adding cors middleware setting for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials=True,
    allow_methods = ["*"],
    allow_headers = ["*"],
    expose_headers = ["Content-Type"]
    )

# receives user input msg from client
@app.get("/chat/{message}")
async def get_chat_message(message: str, checkpoint_id: Optional[str] = None):
    config = {
        "configurable" : {
            "thread_id" : checkpoint_id
        }}
    return await graph.ainvoke({
    "messages": [HumanMessage(content=message)]
}, config=config)

# print(response["messages"][-1].content)
