# Build from the project root:  docker build -f Docker/Dockerfile -t food-ai .
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=10000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code ./code
COPY model ./model

RUN useradd --create-home appuser
USER appuser

EXPOSE 10000
# 1 worker = one copy of the model in RAM (~0.85 GB). Threads let /health answer during a prediction.
CMD ["sh", "-c", "exec gunicorn --chdir /app/code --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 4 --timeout 120 --access-logfile - app:app"]
