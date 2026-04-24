FROM python:3.12.16-ubuntu22.04

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app
EXPOSE 9000
CMD ["python", "-m", "app.server"]
