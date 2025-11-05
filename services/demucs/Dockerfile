FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y ffmpeg git && rm -rf /var/lib/apt/lists/*

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir torch==2.8.* torchaudio torchvision
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 9101

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9101"]


