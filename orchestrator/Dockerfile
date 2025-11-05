FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Allow compose to control PyTorch wheel channel (e.g., cu118 for Pascal P40)
ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]


