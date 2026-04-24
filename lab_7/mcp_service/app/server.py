"""MCP сервер"""

from pathlib import Path
import requests
from fastmcp import FastMCP

from .config import settings
from .rag_store import RetrieverStore


mcp = FastMCP("petfinder-rag")
store = RetrieverStore()
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


def read_prompt(filename: str) -> str:
    """Стение текста промпта из файла"""
    return (PROMPTS_DIR / filename).read_text(encoding="utf-8").strip()


@mcp.tool()
def index_dataset(csv_path: str) -> dict:
    """Индексация"""
    return store.index_dataset(csv_path)


@mcp.tool()
def retrieve_context(query: str, k: int = 3) -> list[dict]:
    """ретрив"""
    return store.retrieve(query=query, k=k)


@mcp.tool()
def rag_predict(profile: str, k: int = 3) -> dict:
    """Ретрив + вызов ллм"""
    docs = store.retrieve(query=profile, k=k)
    context = "\n".join([f"- {doc['text']} (label={doc['label']})" for doc in docs])
    system_prompt = read_prompt("rag_system_prompt.md")
    user_prompt = read_prompt("rag_user_prompt.md").format(context=context, profile=profile)
    payload = {
        "provider": settings.llm_provider,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    response = requests.post(f"{settings.llm_service_url}/generate", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return {"llm_answer": data["answer"], "used_docs": docs}


if __name__ == "__main__":
    mcp.run(transport="sse", host="0.0.0.0", port=9000)
