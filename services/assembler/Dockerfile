FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY services/assembler/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY services/assembler/app /app/app
COPY void_json /app/void_json

EXPOSE 9095

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9095"]


