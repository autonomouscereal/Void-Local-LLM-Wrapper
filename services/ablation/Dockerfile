FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Repo-root build context: copy the ablation service requirements from its subdir.
COPY services/ablation/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared packages so the service can use JSONParser
COPY void_json /app/void_json

# Service code
COPY services/ablation/app /app/app

EXPOSE 8096

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8096"]


