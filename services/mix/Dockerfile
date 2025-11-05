FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi uvicorn soundfile numpy

WORKDIR /app
COPY app /app/app

EXPOSE 7865
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7865"]


