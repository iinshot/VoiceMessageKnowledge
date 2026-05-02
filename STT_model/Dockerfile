FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    python3.11-venv \
    libsndfile1 \
    libsndfile1-dev \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
RUN sed -i 's/available_backends = torchaudio\.list_audio_backends()/available_backends = torchaudio.list_audio_backends() if hasattr(torchaudio, "list_audio_backends") else []/g' \
    /usr/local/lib/python3.11/dist-packages/speechbrain/utils/torch_audio_backend.py

COPY Speech-to-Text-main/ ./Speech-to-Text-main/
COPY app.py .

EXPOSE 8001

CMD ["python", "app.py"]