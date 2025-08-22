# syntax=docker/dockerfile:1.4
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure the model exists
RUN test -f models/delay_model.pkl || (echo "Missing models/delay_model.pkl. Run 'make train' first." && false)

RUN useradd -m -u 10001 appuser && chown -R appuser:appuser /app

USER appuser

ENV PYTHONPATH=/app
ENV MODEL_PATH=/app/models/delay_model.pkl
ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn", "challenge.api:app", "--host", "0.0.0.0", "--port", "8000", "--app-dir", "/app"]
