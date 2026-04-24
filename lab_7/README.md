# Lab 6: NLP (мой результат)

В этой лабораторной я сделала простой PoC для приюта животных: LLM-сервис на FastAPI + Ollama (Qwen2.5:0.5B), MCP-сервис с RAG, и сравнение техник промптинга.

## Что сделано
- Подготовила LLM-сервис с эндпоинтом `/generate`
- Подключила Ollama с моделью `qwen2.5:0.5b`
- Сделала MCP-сервис с тулами:
  - `index_dataset(csv_path)`
  - `retrieve_context(query, k)`
  - `rag_predict(profile, k)`
- Добавила ретривер на `sentence_transformers + faiss` и поддержку `qdrant`
- Вынесла промпты в `prompts/*.md`
- Добавила скрипт расчета метрик `scripts/evaluate_prompting.py`

## Почему так
Я вынесла промпты в markdown-файлы, потому что так их проще читать и менять. Я оставила OpenAI-совместимый интерфейс (`OPENAI_BASE_URL`), чтобы при необходимости подключать любого совместимого провайдера. Для векторного поиска я чаще использую Qdrant, потому что удобнее работать с облачным хранилищем и фильтрами по payload.

## Где что лежит
- `llm_service/` — FastAPI + провайдеры (`ollama/openai/hf`)
- `mcp_service/` — FastMCP + RAG
- `prompts/` — все системные и пользовательские промпты в `.md`
- `data/petfinder_val.csv` — исходный датасет
- `data/petfinder_val_prepared.csv` — подготовленный датасет формата `profile,label`
- `scripts/evaluate_prompting.py` — расчет метрик

## Как запустить
1. Скопировать env:
`cp .env.example .env`

2. Поднять Ollama и модель:
`ollama serve`
`ollama pull qwen2.5:0.5b`

3. Поднять сервисы:
`docker compose up --build`

4. Проверить сервис:
`curl http://127.0.0.1:8000/health`

5. Проверить генерацию:
`curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"provider":"ollama","user_prompt":"Оцени профиль: молодая привитая дружелюбная кошка"}'`

## Запуск метрик
1. Убедиться, что `llm_service` запущен и Ollama доступен
2. Запустить:
`python scripts/evaluate_prompting.py --provider ollama --base-url http://127.0.0.1:8000 --dataset ai_4_course/lab_7/data/petfinder_val_prepared.csv`

## Как запускать без Ollama (через Hugging Face)
1. Скопировать env и добавить токен:
`cp .env.example .env`
в `.env` заполнить `HF_TOKEN=...`
и поставить `HF_MODEL=Qwen/Qwen2.5-7B-Instruct` (для router этот вариант у меня рабочий; `0.5B` у меня был недоступен)

2. Поднять LLM сервис:
`cd ai_4_course/lab_7/llm_service && uvicorn app.main:app --host 127.0.0.1 --port 8000`

3. Проверить генерацию:
`curl -X POST http://127.0.0.1:8000/generate -H 'Content-Type: application/json' -d '{"provider":"hf","user_prompt":"Оцени профиль: молодая привитая дружелюбная кошка"}'`

4. Прогнать метрики:
`cd ai_4_course/lab_7 && python scripts/evaluate_prompting.py --provider hf --base-url http://127.0.0.1:8000 --dataset ai_4_course/lab_7/data/petfinder_val_prepared.csv`

## Если работаю через Qdrant
В `.env` заполняю `QDRANT_URL` и `QDRANT_API_KEY`, после этого MCP пишет и читает векторы из Qdrant

## Метрики
Метрики я считаю скриптом `scripts/evaluate_prompting.py`: для каждого профиля модель возвращает `verdict` (0 или 1), затем я сравниваю его с истинной меткой `label`.

`Accuracy` - доля всех правильных ответов 
`Precision` - из всех случаев, где модель сказала «быстро усыновят», сколько реально были такими 
`Recall` - из всех реально «быстро усыновят» сколько модель нашла 
`F1` - баланс между precision и recall 

Для `petfinder_val.csv` я сделала `petfinder_val_prepared.csv` с колонками `profile,label`, где метка: `AdoptionSpeed <= 2 -> 1`, иначе `0`

Итоговые метрики ниже посчитаны на большой модели `Qwen/Qwen2.5-7B-Instruct` (через Hugging Face). На локалке `qwen2.5:0.5b` результаты были хуже

## Последний фактический прогон 
Успешно досчитан прогон на части датасета: **300 строк** (`data/petfinder_val_partial_300.csv`).

Результат на 300 строках:
- `zero-shot`: accuracy `0.5233`, precision `0.5308`, recall `0.7179`, f1 `0.6104`
- `cot`: accuracy `0.49`, precision `0.5517`, recall `0.1026`, f1 `0.173`
- `few-shot`: accuracy `0.5367`, precision `0.5752`, recall `0.4167`, f1 `0.4833`
- `cot+few-shot`: accuracy `0.5033`, precision `0.6`, recall `0.1346`, f1 `0.2199`

Файлы:
- `report_metrics_hf_val_partial_300.json` — успешные метрики на 300 строках
