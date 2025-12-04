FROM node:20-bullseye AS frontend
WORKDIR /app
COPY services/chatui/frontend/package.json ./
RUN npm install --no-audit --no-fund
COPY services/chatui/frontend ./
RUN npm run build

FROM python:3.11-slim AS backend
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
WORKDIR /app
RUN apt-get update && apt-get install -y libpq-dev build-essential && rm -rf /var/lib/apt/lists/*
COPY services/chatui/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY services/chatui/backend /app/backend
COPY void_json /app/void_json
COPY --from=frontend /app/dist /app/frontend_dist
EXPOSE 3000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3000"]


