import os 
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain import hub

from langchain.agents import tool, create_react_agent, AgentExecutor
from langchain.agents import initialize_agent
from langchain.agents.agent_types import AgentType
from langsmith import Client

load_dotenv()

confluence_api_key = os.environ["CONFLUENCE_TOKEN"]
langsmith_api_key = os.environ["LANGSMITH_API_KEY"]

client = Client(api_key=langsmith_api_key)

llm = OllamaLLM(model="llama3:8b", temperature=0)
# # embedding_fn = OpenAIEmbeddings()
embedding_fn = OllamaEmbeddings(model="llama3:8b")

loader1 = PyPDFLoader("InfoTech_HR_Policy_Manual.pdf")
pdf_docs = loader1.load()

loader2 = ConfluenceLoader(  
url="https://poorvashrivastav03.atlassian.net/wiki",
username="poorvashrivastav03@gmail.com",  
api_key=confluence_api_key,
space_key="TRD",      # Replace with your Confluence space key
limit=5
)
confluence_docs = loader2.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

chunks_hr = splitter.split_documents(pdf_docs)
vector_db_hr = Chroma.from_documents(chunks_hr, embedding=embedding_fn, collection_name="internal_hr_docs")
retriever_hr = vector_db_hr.as_retriever()


chunk_confl = splitter.split_documents(confluence_docs)
vector_db_confl = Chroma.from_documents(chunk_confl, embedding=embedding_fn, collection_name="confluence_docs")
retriever_confl = vector_db_confl.as_retriever()


# --- taken from langchain documentation ---
# Contextualize question
hr_system_prompt = """
You are an AI HR Assistant for Info Tech Pvt. Limited.
Use only the HR Policy Manual to answer employee queries accurately.
If the manual does not contain the answer, reply:
“This information is not explicitly mentioned in the HR Policy Manual. Please contact HR for clarification.”
Always be clear, professional, and concise.
"""

hr_contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", hr_system_prompt),
    # MessagesPlaceholder("chat_history"),
    ("human", "{input}"),    
])


history_aware_retriever_hr = create_history_aware_retriever(llm, retriever_hr, hr_contextualize_prompt)

# ---------------------- generic code --------------------------------

qa_system_prompt = (
    "You are an assistant for question-answering tasks. Use "
    "the following pieces of retrieved context to answer the "
    "question. If you don't know the answer, just say that you "
    "don't know. Use three sentences maximum and keep the answer "
    "concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        # MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Below we use create_stuff_documents_chain to feed all retrieved context into the LLM
qa_chain = create_stuff_documents_chain(llm, qa_prompt)

# ------------------------------------------------------

hr_chain = create_retrieval_chain(
    history_aware_retriever_hr, qa_chain
)

confl_system_prompt = """You are an assistant with access to Confluence documents.
Use the retrieved context strictly to answer the user's question.

If the context is insufficient, say: "The information is not available in the Confluence space."

Do not make up information. 

Be accurate, concise, and include the title if it is in the context.

Question: {input}
Context:
{context}

Answer:"""


confl_contextualize_prompt = ChatPromptTemplate.from_messages([
    ("system", confl_system_prompt),
    # MessagesPlaceholder("chat_history"),
    ("human", "{input}"),    
])

history_aware_retriever_confl = create_history_aware_retriever(llm, retriever_confl, confl_contextualize_prompt)

confluence_chain = create_retrieval_chain(
    history_aware_retriever_confl, qa_chain
)

# chat_history = []``

# ---------------------- Agent -------------------------

@tool("search_hr_docs")
def search_hr_docs(query: str) -> str:
    """Search HR documents for employee policies, payroll, benefits, or leave information.
    Do not use for technical, project, or Confluence-related queries.
    """  
    result = hr_chain.invoke({"input": query})
    return result["answer"]

@tool("search_confluence_docs")
def search_confluence_docs(query: str) -> str:    
    """Search Confluence for technical setup, integration, or project documentation (e.g., OAuth, Jira,APIs).
    Do not use for HR-related queries or employee policies.
    """
    result = confluence_chain.invoke({"input": query})    
    return result["answer"]


tools = [search_hr_docs, search_confluence_docs]

# prompt_template = client.pull_prompt("hwchase17/react", include_model=True)

prompt_template = client.pull_prompt("hwchase17/react").partial(
    format_instructions=(
        "Follow the ReAct schema strictly. "
        "If you already have enough information, "
        "skip 'Action' and directly output a 'Final Answer'. "
        "The Final Answer must be a clear, concise summary of the relevant content."
    )
)

agent = create_react_agent(
    llm=llm,
    tools=tools,
    prompt=prompt_template,  
)

executor = AgentExecutor.from_agent_and_tools(
    agent=agent, tools=tools, verbose=True, max_iterations=3, handle_parsing_errors=True)

result = executor.invoke({"input":"Summarize the functional requirements of OAuth 2.0"})

print("Final Answer:", result["output"])




