"""Адаптеры для LLM провайдеров"""

from dataclasses import dataclass
from openai import OpenAI
import requests

from .config import settings


@dataclass
class ProviderResult:
    """Ответ моделие"""
    answer: str
    provider: str
    model: str


def ask_ollama(user_prompt: str, system_prompt: str | None = None) -> ProviderResult:
    """Отправляет запрос в Ollama"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
    }
    response = requests.post(f"{settings.ollama_url}/api/chat", json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()
    return ProviderResult(
        answer=data["message"]["content"],
        provider="ollama",
        model=settings.ollama_model,
    )


def ask_openai(user_prompt: str, system_prompt: str | None = None) -> ProviderResult:
    """Отправляет запрос в OpenAI"""
    client = OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    completion = client.chat.completions.create(model=settings.openai_model, messages=messages)
    return ProviderResult(
        answer=completion.choices[0].message.content or "",
        provider="openai",
        model=settings.openai_model,
    )


def ask_hf(user_prompt: str, system_prompt: str | None = None) -> ProviderResult:
    """Отправляет запрос в Hugging Face"""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    headers = {
        "Authorization": f"Bearer {settings.hf_token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": settings.hf_model,
        "messages": messages,
        "max_tokens": 64,
        "temperature": 0.1,
    }
    url = "https://router.huggingface.co/v1/chat/completions"
    response = requests.post(url, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    data = response.json()
    completion = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    return ProviderResult(
        answer=completion or "",
        provider="huggingface",
        model=settings.hf_model,
    )


def generate_answer(user_prompt: str, system_prompt: str | None = None, provider: str | None = None) -> ProviderResult:
    """Выбир провайдера"""
    resolved_provider = (provider or settings.llm_provider).lower()
    if resolved_provider == "openai":
        return ask_openai(user_prompt=user_prompt, system_prompt=system_prompt)
    if resolved_provider in {"hf", "huggingface"}:
        return ask_hf(user_prompt=user_prompt, system_prompt=system_prompt)
    return ask_ollama(user_prompt=user_prompt, system_prompt=system_prompt)
