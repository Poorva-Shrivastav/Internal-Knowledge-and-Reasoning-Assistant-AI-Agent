The bot that not only answers “where is X documented?” but can reason across multiple sources and help make decisions.

Example Use Cases

Engineering:

    “What’s our OAuth strategy and who owns it?”

    Bot retrieves from Confluence → reasons it → gives structured answer.

Cross-functional:

    “What are the main onboarding steps for new engineers, and how long does each take?”

    Bot pulls HR docs + Engineering docs → summarizes into a step-by-step plan.

Tech Stack

LangChain components:

    1. Document Loaders (Confluence Space, HR policy doc)
    2. Embeddings (OllamaEmbeddings for local /OpenAIEmbeddings)
    3. VectorStore (Chroma for local / Pinecone for hosted)
    4. ConversationalRetrievalChain + ReAct AgentExecutor

Models:

    GPT-4, llama-3

Steps:

1. Data Ingestion: Loading internal documents (Confluence, Github md, HR policy pdf), chunk, and store in a vector DB.

2.Retrieval Chain (RAG) : Retrieve relevant context when a user asks a question

last commit :

- Implemented `search_confluence_docs` tool to query internal Confluence pages
- Integrates with LangGraph ToolNode for automatic tool invocation
- Formats retrieved documents as readable strings for LLM summarization
- Supports multi-step reasoning: LLM decides to call tool and generates final answer

Note - when working with Ollama:

1. ollama run llama3
2. ps aux | grep ollama
3. pkill ollama
4. ollama ps - returns list of active llama models
