FROM pytorch/pytorch:2.7.1-cuda11.8-cudnn9-runtime

ARG PIP_EXTRA_INDEX_URL
ENV PIP_EXTRA_INDEX_URL=${PIP_EXTRA_INDEX_URL}
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg curl ca-certificates && rm -rf /var/lib/apt/lists/*

RUN python -m pip install -U pip

# Torch is already present in base, but keep CUDA channel consistent.
RUN python -m pip install --extra-index-url "${PIP_EXTRA_INDEX_URL}" \
  "torch==2.7.1+cu118" \
  "torchaudio==2.7.1+cu118"

RUN python -m pip install \
  "fastapi" \
  "uvicorn[standard]" \
  "numpy==1.26.4" \
  "opencv-python-headless" \
  "tqdm" \
  "Pillow"

# Pull Practical-RIFE code via codeload tarball
RUN mkdir -p /opt/practical_rife && \
  curl -L -o /tmp/practical_rife.tar.gz https://codeload.github.com/hzwer/Practical-RIFE/tar.gz/main && \
  tar -xzf /tmp/practical_rife.tar.gz -C /opt/practical_rife --strip-components=1 && \
  rm -f /tmp/practical_rife.tar.gz

# Copy shared packages so the service can return canonical ToolEnvelopes + use JSONParser
COPY void_envelopes /app/void_envelopes
COPY void_json /app/void_json

COPY services/rife_vfi/app.py /app/app.py

ENV RIFE_VFI_PORT=8098
EXPOSE 8098

CMD ["python","-u","/app/app.py"]


