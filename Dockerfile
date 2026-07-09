FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir \
    fastapi uvicorn python-multipart \
    python-dotenv groq soundfile

COPY src/ src/
COPY api/ api/
COPY app/ app/
COPY evaluation/ evaluation/
COPY data/ data/

EXPOSE 8501 8000

CMD ["streamlit", "run", "app/dashboard.py", "--server.port=8501", "--server.address=0.0.0.0"]
