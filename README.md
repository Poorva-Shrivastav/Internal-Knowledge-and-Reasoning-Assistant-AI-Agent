The bot that not only answers “where is X documented?” but can reason across multiple sources and help make decisions.

Example Use Cases

Engineering:

    “What’s our OAuth strategy?”

    Bot retrieves from Confluence → reasons it → gives structured answer.

Cross-functional:

    “What is our leave policy?”

    Bot pulls HR policy docs → summarizes the answer.

Tech Stack

LangChain components:

    1. Document Loaders (Confluence Space, HR policy doc)
    2. Embeddings (OllamaEmbeddings for local /OpenAIEmbeddings)
    3. VectorStore (Chroma for local / Pinecone for hosted)
    4. ConversationalRetrievalChain + ReAct AgentExecutor

Langgraph components:

    1. State Graph
    2. Adding nodes and Edges
    3. Adding MemorySaver config

Models:

    ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

Note:

- Implemented `search_confluence_docs` tool to query internal Confluence pages
- Integrates with LangGraph ToolNode for automatic tool invocation
- Formats retrieved documents as readable strings for LLM summarization
- Supports multi-step reasoning: LLM decides to call tool and generates final answer
