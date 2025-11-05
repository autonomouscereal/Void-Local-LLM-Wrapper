FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir fastapi uvicorn

WORKDIR /app
COPY app /app/app

EXPOSE 7868
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7868"]


