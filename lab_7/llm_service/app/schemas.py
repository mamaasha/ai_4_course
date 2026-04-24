from pydantic import BaseModel


class GenerateRequest(BaseModel):
    """Данные запроса к модели"""
    user_prompt: str
    system_prompt: str | None = None
    provider: str | None = None


class GenerateResponse(BaseModel):
    """Ответ модели"""
    answer: str
    provider: str
    model: str
