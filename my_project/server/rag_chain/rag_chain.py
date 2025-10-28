import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, ConfluenceLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model

load_dotenv()

confluence_api_key = os.environ["CONFLUENCE_TOKEN"]

llm = init_chat_model("google_genai:gemini-2.0-flash")

embedding_fn = OllamaEmbeddings(model="nomic-embed-text")

loader1 = PyPDFLoader("./assets/InfoTech_HR_Policy_Manual.pdf")

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
retriever_hr = vector_db_hr.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
                )


chunk_confl = splitter.split_documents(confluence_docs)
vector_db_confl = Chroma.from_documents(chunk_confl, embedding=embedding_fn, collection_name="confluence_docs")
retriever_confl = vector_db_confl.as_retriever(
                search_type="similarity", 
                search_kwargs={"k": 3}
                )