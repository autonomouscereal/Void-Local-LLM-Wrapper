FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg git && rm -rf /var/lib/apt/lists/*

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR /app

COPY requirements.txt /app/requirements.txt

# Install Torch stack (cu118) in line with other services, then vision libs.
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8095

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8095", "--log-level", "info"]


