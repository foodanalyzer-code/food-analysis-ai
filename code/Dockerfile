FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=10000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app.py final_config_dual.json food_model_dual.tflite gatekeeper_model.tflite imagenet_classes.json final_config.json ./

RUN useradd --create-home appuser
USER appuser

EXPOSE 10000
CMD ["sh", "-c", "exec gunicorn --chdir /app --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 --access-logfile - app:app"]
