FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Allow compose to control PyTorch wheel channel (e.g., cu118 for Pascal P40)
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

# System deps required by cv2 (libGL, GLib, etc.)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

# IMPORTANT (policy): Module import path is fixed to 'app.*' within /app.
# Do NOT rely on external workpath or PYTHONPATH overrides from compose.
# Uvicorn must be started with module 'app.main_min:app' (or 'app.main:app').
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


