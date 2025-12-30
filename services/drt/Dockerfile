FROM mcr.microsoft.com/playwright/python:v1.48.0-jammy

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libxml2 libxslt1.1 ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Repo-root build context: copy the drt service requirements from its subdir.
COPY services/drt/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy shared packages so the service can use JSONParser
COPY void_json /app/void_json

# Service code
COPY services/drt/app /app/app

EXPOSE 8086

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8086"]


