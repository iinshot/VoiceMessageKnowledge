FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY main.py .
COPY converter.py .
COPY work_with_files.py .
COPY services_client.py .

CMD ["python", "main.py"]