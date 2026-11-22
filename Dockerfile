FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('tiny', compute_type='int8')" || true
ENV WHISPER_MODEL=tiny
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-10000}
