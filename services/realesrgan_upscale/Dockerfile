FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip

# Keep CUDA wheel channel consistent (same pattern as other services)
RUN python -m pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" \
  "torch==2.7.1+cu118" \
  "torchaudio==2.7.1+cu118"

RUN python -m pip install \
  "basicsr==1.4.2" \
  "realesrgan" \
  "gfpgan==1.3.8" \
  "facexlib==0.3.0" \
  "opencv-python-headless" \
  "numpy==1.26.4" \
  "Pillow" \
  "fastapi" \
  "uvicorn[standard]"

# Pull Real-ESRGAN code via codeload tarball
RUN mkdir -p /opt/realesrgan && \
  curl -L -o /tmp/realesrgan.tar.gz https://codeload.github.com/xinntao/Real-ESRGAN/tar.gz/master && \
  tar -xzf /tmp/realesrgan.tar.gz -C /opt/realesrgan --strip-components=1 && \
  rm -f /tmp/realesrgan.tar.gz && \
  mkdir -p /opt/realesrgan/weights

# Copy shared packages so the service can return canonical ToolEnvelopes + use JSONParser
COPY void_envelopes /app/void_envelopes
COPY void_json /app/void_json

COPY services/realesrgan_upscale/app.py /app/app.py

ENV UPSCALE_PORT=8099
EXPOSE 8099

CMD ["python","-u","/app/app.py"]


