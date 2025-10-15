import os 
from dotenv import load_dotenv

from rag_chain.rag_chain import llm
from agents.agent_tools import search_hr_docs, search_confluence_docs
from langchain_core.tools import tool
# from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage

from typing import TypedDict, Annotated, List
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition


# load_dotenv()

# # langsmith_api_key = os.environ["LANGSMITH_API_KEY"]

# # client = Client(api_key=langsmith_api_key)

@tool("summarize_docs", return_direct=False)
def summarize_docs(text: str):
    """Summarize retrieved content for quick reference."""    
    return llm.invoke(f"Summarize briefly:\n\n{text}").content


tools = [search_hr_docs, search_confluence_docs]

llm_with_tools = llm.bind_tools(tools)


class State(TypedDict):
    messages : Annotated[list, add_messages]

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

# response = graph.invoke({
#     "messages": [HumanMessage(content="What is the leave policy?")]
# })

response = graph.invoke({
    "messages": [HumanMessage(content="Summarize the functional requirements of OAuth 2.0?")]
})

print(response["messages"][-1].content)


# if __name__ == "__main__":
#     search_hr_docs
#     search_confluence_docs