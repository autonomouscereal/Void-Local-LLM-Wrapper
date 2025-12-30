FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y git ffmpeg && rm -rf /var/lib/apt/lists/*
# Compose controls PyTorch wheel channel (e.g., cu118 for Pascal P40)
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

WORKDIR /app

# Repo-root build context: copy the xtts service requirements from its subdir.
COPY services/xtts/requirements.txt /app/requirements.txt
# Pre-install CUDA torch cu118 compatible with P40 and RTX 8000
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 \
    "torch==2.7.1+cu118" "torchvision==0.22.1+cu118" "torchaudio==2.7.1+cu118"
RUN pip install --no-cache-dir -r /app/requirements.txt

# Service code
COPY services/xtts/app /app/app

# Shared repo packages required at runtime (xtts does not mount the repo).
COPY void_json /app/void_json
COPY void_envelopes /app/void_envelopes
COPY void_quality /app/void_quality

EXPOSE 8020

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8020"]


