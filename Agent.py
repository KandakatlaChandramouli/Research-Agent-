import os
import json
import operator
from typing import List, Dict, Any, TypedDict, Annotated
import requests
import arxiv
import lancedb
from sentence_transformers import SentenceTransformer
import ollama

# Initialize Local Embedding Model (HuggingFace Open Source)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Initialize Local LanceDB for HNSW / Vector Deduplication
DB_PATH = "./research_memory"
db = lancedb.connect(DB_PATH)

# Setup LanceDB Table Schema
try:
    table = db.open_table("research_papers")
except Exception:
    sample_data = [{
        "text": "init",
        "vector": embedder.encode("init").tolist(),
        "title": "init",
        "source": "init",
        "url": "init",
        "summary": "init"
    }]
    table = db.create_table("research_papers", data=sample_data)

# ==========================================
# 1. 100% FREE & OPEN SEARCH TOOLS
# ==========================================

def search_arxiv(query: str, max_results=3) -> List[Dict[str, Any]]:
    """Search arXiv (Free, open API)."""
    try:
        search_client = arxiv.Client()
        search = arxiv.Search(query=query, max_results=max_results, sort_by=arxiv.SearchSortBy.Relevance)
        papers = []
        for result in search_client.results(search):
            papers.append({
                "title": result.title,
                "authors": [a.name for a in result.authors],
                "abstract": result.summary,
                "url": result.pdf_url,
                "year": str(result.published.year),
                "source": "arxiv"
            })
        return papers
    except Exception as e:
        print(f"arXiv search error: {e}")
        return []

def search_semantic_scholar(query: str, max_results=3) -> List[Dict[str, Any]]:
    """Search Semantic Scholar (Free public API)."""
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,abstract,authors,year,url"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        papers = []
        for item in data:
            papers.append({
                "title": item.get("title", ""),
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "abstract": item.get("abstract", "") or "No abstract available.",
                "url": item.get("url", "https://semanticscholar.org"),
                "year": str(item.get("year", "N/A")),
                "source": "semantic_scholar"
            })
        return papers
    except Exception as e:
        print(f"Semantic Scholar error: {e}")
        return []

def search_patents_free(query: str, max_results=3) -> List[Dict[str, Any]]:
    """Free alternative for patent data using EuropePMC / Free Patent search indices or arXiv cross-search."""
    # Using a targeted query on Semantic Scholar specifically filtering for patent-like technical disclosures
    try:
        url = f"https://api.semanticscholar.org/graph/v1/paper/search?query=patent+{query}&limit={max_results}&fields=title,abstract,authors,year,url"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        data = r.json().get("data", [])
        patents = []
        for item in data:
            patents.append({
                "title": f"[Patent/Tech Disclosure] {item.get('title', '')}",
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "abstract": item.get("abstract", "") or "No abstract available.",
                "url": item.get("url", "https://patents.google.com"),
                "year": str(item.get("year", "N/A")),
                "source": "patent_open_index"
            })
        return patents
    except Exception as e:
        print(f"Open Patent search error: {e}")
        return []

# ==========================================
# 2. LOCAL LLM BRAIN (OLLAMA)
# ==========================================

def call_local_llm(prompt: str) -> str:
    """Queries local Ollama instance running open-source models like Llama 3."""
    try:
        response = ollama.chat(model='llama3', messages=[
            {
                'role': 'user',
                'content': prompt,
            },
        ])
        return response['message']['content']
    except Exception as e:
        print(f"Ollama execution error (Is ollama running?): {e}")
        return "{}"

# ==========================================
# 3. STATE & GRAPH DEFINITION
# ==========================================

class ResearchState(TypedDict):
    user_query: str
    expanded_queries: List[str]
    raw_items: Annotated[List[Dict[str, Any]], operator.add]
    ranked_results: List[Dict[str, Any]]
    report_path: str

def query_expander_node(state: ResearchState) -> Dict[str, Any]:
    prompt = (
        f"User wants research on: '{state['user_query']}'.\n"
        "Generate 4 precise alternative search queries including technical synonyms and architectural phrases. "
        "Return ONLY a raw JSON list of strings format like: [\"query1\", \"query2\"]. No markdown blocks, just the array."
    )
    content = call_local_llm(prompt)
    try:
        # Clean up output if model injects markdown
        clean_json = content.replace("```json", "").replace("```", "").strip()
        queries = json.loads(clean_json)
        if not isinstance(queries, list):
            queries = [state["user_query"]]
    except Exception:
        queries = [state["user_query"]]
    
    return {"expanded_queries": queries[:4]}

def searcher_node(state: ResearchState) -> Dict[str, Any]:
    collected_items = []
    # Use user query + expanded queries
    queries_to_run = [state["user_query"]] + state.get("expanded_queries", [])
    for q in queries_to_run:
        collected_items.extend(search_arxiv(q, max_results=2))
        collected_items.extend(search_semantic_scholar(q, max_results=2))
        collected_items.extend(search_patents_free(q, max_results=2))
    return {"raw_items": collected_items}

def ranker_summarizer_node(state: ResearchState) -> Dict[str, Any]:
    unique_items = []
    
    # Deduplication via LanceDB HNSW Vector Similarity (Fully Local)
    for item in state["raw_items"]:
        text_to_embed = f"{item['title']} {item['abstract']}"
        vector = embedder.encode(text_to_embed).tolist()
        
        similar = table.search(vector).limit(1).to_list()
        if similar and similar[0]["_distance"] < 0.08:  
            continue
        
        try:
            table.add([{
                "text": text_to_embed[:500],
                "vector": vector,
                "title": item["title"],
                "source": item["source"],
                "url": item["url"],
                "summary": ""
            }])
        except Exception:
            pass
        
        unique_items.append(item)

    # Local LLM Summarization for top items
    top_items = unique_items[:10]
    for item in top_items:
        prompt = (
            f"Summarize this paper/patent in 2 concise sentences focusing on core problem and method.\n\n"
            f"Title: {item['title']}\nAbstract: {item['abstract']}"
        )
        item["summary"] = call_local_llm(prompt)

    return {"ranked_results": top_items}

def report_generator_node(state: ResearchState) -> Dict[str, Any]:
    query = state["user_query"]
    results = state["ranked_results"]
    
    papers = [r for r in results if "patent" not in r["source"]]
    patents = [r for r in results if "patent" in r["source"]]

    md = f"# Local Open-Source Research Report\n\n> **Query:** {query}\n\n---\n\n"
    
    md += f"## Academic & Research Papers ({len(papers)})\n\n"
    for p in papers:
        md += f"### [{p['title']}]({p['url']})\n"
        md += f"- **Year:** {p['year']} | **Source:** {p['source']}\n"
        md += f"- **Authors:** {', '.join(p.get('authors', [])[:3])}\n"
        md += f"- **Summary:** {p['summary']}\n\n"

    md += f"## Patent & Deep Technical Disclosures ({len(patents)})\n\n"
    for pat in patents:
        md += f"### [{pat['title']}]({pat['url']})\n"
        md += f"- **Date:** {pat['year']}\n"
        md += f"- **Summary:** {pat['summary']}\n\n"

    report_file = "open_research_report.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(md)

    return {"report_path": report_file}

# ==========================================
# 4. BUILD GRAPH & EXECUTE
# ==========================================

from langgraph.graph import StateGraph

workflow = StateGraph(ResearchState)
workflow.add_node("expander", query_expander_node)
workflow.add_node("searcher", searcher_node)
workflow.add_node("ranker", ranker_summarizer_node)
workflow.add_node("reporter", report_generator_node)

workflow.add_edge("expander", "searcher")
workflow.add_edge("searcher", "ranker")
workflow.add_edge("ranker", "reporter")

workflow.set_entry_point("expander")
agent = workflow.compile()

if __name__ == "__main__":
    import sys
    query_input = sys.argv[1] if len(sys.argv) > 1 else "Branch-aware copy-on-write memory management for speculative LLM inference"
    
    print(f"\n[🚀] Running 100% Local Open-Source Agent for: '{query_input}'...\n")
    final_state = agent.invoke({"user_query": query_input, "expanded_queries": [], "raw_items": [], "ranked_results": [], "report_path": ""})
    
    print(f"\n[✅] Complete! Local report generated at: {final_state['report_path']}")
