
from langchain_core.tools import tool
from rag.rag_chain import retriever_hr, retriever_confl

@tool("search_hr_docs", return_direct=False)
def search_hr_docs(query: str) -> str:
    """Returns result from the HR policy document.
    :Search HR policy documents for a given query, it might include employee policies, payroll, benefits, or leave information.
    :Do not use for technical, project, or Confluence-related queries.
    """
       
    result = retriever_hr.invoke(query)    
    if not result:
        return "No relevant HR policy found."
    return "\n\n".join([r.page_content for r in result])

@tool("search_confluence_docs", return_direct=False)
def search_confluence_docs(query: str) -> str:    
    """Search internal Confluence pages for a given query.
    :Search Confluence for technical setup, integration, or project documentation (e.g., OAuth, Jira,APIs).
    :Do not use for HR-related queries or employee policies.
    """
    
    docs = retriever_confl.invoke(query)     #returns a list

    if not docs:
        return "No relevant Confluence documents found."
    
    # Convert document list into plain text
    combined = "\n\n".join(
        [f"Title: {d.metadata.get('title', 'Untitled')}\n{d.page_content}" for d in docs]
    )

    # print("🔍 Confluence search results:", combined[:300], "...")
    return combined



