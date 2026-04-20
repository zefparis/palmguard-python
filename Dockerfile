FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    --no-install-recommends && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV LIBGL_ALWAYS_SOFTWARE=1
ENV MEDIAPIPE_DISABLE_GPU=1
ENV PORT=8000

CMD gunicorn main:app -w 2 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
