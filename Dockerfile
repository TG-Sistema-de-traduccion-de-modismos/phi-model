FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04 AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv python3-dev build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel && \
    pip wheel --no-cache-dir -r requirements.txt -w /wheels


FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3.10 python3-pip python3-venv ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
RUN pip install --upgrade pip setuptools wheel

COPY --from=builder /wheels /wheels

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir /wheels/*.whl

COPY app/ app/

EXPOSE 8006

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8006"]
