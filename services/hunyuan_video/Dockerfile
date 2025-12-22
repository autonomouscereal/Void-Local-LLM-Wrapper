FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN python -m pip install -U pip

COPY requirements.txt /app/requirements.txt
# All pins live in requirements.txt (policy). Torch wheels come from the cu118 index.
RUN python -m pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" -r /app/requirements.txt

COPY app.py /app/app.py

ENV HYVIDEO_PORT=8094
EXPOSE 8094

CMD ["python","-u","/app/app.py"]


