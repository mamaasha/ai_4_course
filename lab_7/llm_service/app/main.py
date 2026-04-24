"""FastAPI сервис для запросов к LLM"""

from fastapi import FastAPI, HTTPException

from .providers import generate_answer
from .schemas import GenerateRequest, GenerateResponse

app = FastAPI(title="Pet Shelter LLM Service", version="1.0.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
def generate(payload: GenerateRequest) -> GenerateResponse:
    """Отправляет запрос в модель"""
    try:
        result = generate_answer(
            user_prompt=payload.user_prompt,
            system_prompt=payload.system_prompt,
            provider=payload.provider,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return GenerateResponse(answer=result.answer, provider=result.provider, model=result.model)
