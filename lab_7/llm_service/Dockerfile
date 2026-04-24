FROM python:3.12.16-ubuntu22.04

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

RUN curl -fsSL https://ollama.com/install.sh | sh

COPY app /app/app

EXPOSE 8000
CMD ["bash", "-lc", "ollama serve & sleep 3 && ollama pull qwen2.5:0.5b && uvicorn app.main:app --host 0.0.0.0 --port 8000"]
