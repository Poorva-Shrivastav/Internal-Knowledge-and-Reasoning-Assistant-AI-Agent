import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, ConfluenceLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.tools import tool
from langchain_mistralai import MistralAIEmbeddings

load_dotenv()

# os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

confluence_api_key = os.environ["CONFLUENCE_TOKEN"]
os.environ["MISTRAL_API_KEY"] = os.getenv("MISTRAL_API_KEY")

# llm = init_chat_model("google_genai:gemini-2.0-flash")

# embedding_fn = OllamaEmbeddings(model="nomic-embed-text")

embedding_fn = MistralAIEmbeddings(
    model="mistral-embed",
)

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