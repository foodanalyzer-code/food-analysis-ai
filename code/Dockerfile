# Build from the project root:  docker build -f Docker/Dockerfile -t food-ai .
#
# Two stages:
#   1) convert - loads the notebook's .keras model with TensorFlow, converts it to .tflite,
#                and verifies the result (the build FAILS if it does not match the original).
#   2) serve   - small image with LiteRT + Flask only. TensorFlow is NOT shipped.

# ---------- Stage 1: convert (runs on the build machine, not on the 512 MB server) ----------
FROM python:3.13-slim AS convert
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 TF_CPP_MIN_LOG_LEVEL=2
# Same versions used to train the model in Colab.
RUN pip install tensorflow-cpu==2.21.0 keras==3.13.2 numpy==2.1.3 ai-edge-litert==2.3.0
WORKDIR /convert
COPY model/EfficientNetB0_finetuned.keras model/final_config.json ./model/
COPY code/convert_model.py .
RUN mkdir out && python convert_model.py model out

# ---------- Stage 2: serve (what actually runs on Render) ----------
FROM python:3.13-slim
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MALLOC_ARENA_MAX=2 \
    PORT=10000

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code/app.py ./code/
COPY model/final_config.json ./model/
COPY --from=convert /convert/out/food_model.tflite /convert/out/conversion_report.json ./model/

RUN useradd --create-home appuser
USER appuser

EXPOSE 10000
# 1 worker (one model copy). 2 threads so /health answers while a photo is processed.
CMD ["sh", "-c", "exec gunicorn --chdir /app/code --bind 0.0.0.0:${PORT:-10000} --workers 1 --threads 2 --timeout 120 --access-logfile - app:app"]
