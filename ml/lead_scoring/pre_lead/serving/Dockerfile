# Cloud Run serving image. Build context = repo root.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY serving/requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Shared preprocessing lib (same code as training) + app
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install --no-deps -e /app
COPY serving/app.py /app/app.py

# Cloud Run provides $PORT (default 8080)
ENV PORT=8080
CMD exec uvicorn app:app --host 0.0.0.0 --port ${PORT}
