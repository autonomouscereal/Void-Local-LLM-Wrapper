FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg sox && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir fastapi uvicorn soundfile numpy

WORKDIR /app
COPY app /app/app

EXPOSE 7867
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7867"]


