FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install -U pip

# Repo-root build context: copy the hunyuan_video service requirements from its subdir.
COPY services/hunyuan_video/requirements.txt /app/requirements.txt
# All pins live in requirements.txt (policy). Torch wheels come from the cu118 index.
RUN python -m pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" -r /app/requirements.txt

# Copy shared packages so the service can return canonical ToolEnvelopes + use JSONParser
COPY void_envelopes /app/void_envelopes
COPY void_json /app/void_json

# Service code
COPY services/hunyuan_video/app.py /app/app.py

ENV HYVIDEO_PORT=8094
EXPOSE 8094

CMD ["python","-u","/app/app.py"]


