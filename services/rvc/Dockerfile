FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

WORKDIR /app
# Repo-root build context: copy the rvc service requirements from its subdir.
COPY services/rvc/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cu118 -r /app/requirements.txt

# Service code
COPY services/rvc/app /app/app
# Provide a local shim for the legacy `hf_rvc` imports, so old code paths
# that still do `from hf_rvc import ...` won't crash on import. The shim
# itself is a no-op and will raise if actually called.
COPY services/rvc/hf_rvc /app/hf_rvc
# Copy shared packages so the service can return canonical ToolEnvelopes
COPY void_envelopes /app/void_envelopes
COPY void_json /app/void_json

EXPOSE 7863
CMD ["uvicorn","app.main:app","--host","0.0.0.0","--port","7863"]


