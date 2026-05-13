FROM python:3.11-slim

# Install system dependencies for tesseract, playwright, and audio
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libtesseract-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install uv

WORKDIR /app

COPY nexus-agent-main/requirements.txt .
RUN uv pip install --system -r requirements.txt

RUN playwright install chromium --with-deps

COPY nexus-agent-main/ .

EXPOSE 7860

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]