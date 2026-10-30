"""
Food Analysis AI - inference API (Flask).

Endpoints
  GET  /health   -> {"status": "ok", "model_version": "...", "threshold": 0.3}
  POST /predict  -> header: X-API-Key: <key>
                    body:   multipart/form-data, file field "image"
                            (JPEG, PNG, WEBP, HEIC, MPO - max 15 MB)
                 <- {"predicted_class": "Fresh" | "Rotten", "confidence": 0.9864,
                     "rotten_probability": 0.0136, "threshold": 0.3, "is_rotten": false,
                     "model_version": "...", "processing_ms": 210.5}
  Errors         <- {"error": {"code": "...", "message": "..."}}  (400 / 401 / 413 / 415 / 500)

Model contract (from the training notebook - do not change):
  RGB -> resize 224x224 (PIL BILINEAR) -> float32 / 255 -> model -> sigmoid = P(Rotten)
  Rotten if P >= threshold (0.30, from final_config.json); confidence = P if Rotten else 1 - P
"""
import hashlib
import hmac
import io
import json
import logging
import math
import os
import time
from pathlib import Path

import keras
import numpy as np
import pillow_heif
import tensorflow as tf
from flask import Flask, jsonify, request
from PIL import Image, ImageOps
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("food-ai")
pillow_heif.register_heif_opener()  # iPhone HEIC photos

# ---------------------------------------------------------------- settings
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
API_KEY = os.getenv("API_KEY", "").strip()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "50000000"))
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "HEIF"}

if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required.")

CONFIG = json.loads((MODEL_DIR / "final_config.json").read_text(encoding="utf-8"))
if CONFIG.get("label_map") != {"Fresh": 0, "Rotten": 1}:
    raise RuntimeError(f"Unexpected label_map in final_config.json: {CONFIG.get('label_map')}")
if CONFIG.get("preprocessing") != {"color_mode": "RGB", "rescale": "1/255.0", "interpolation": "bilinear"}:
    raise RuntimeError(f"Unexpected preprocessing in final_config.json: {CONFIG.get('preprocessing')}")

THRESHOLD = float(CONFIG["threshold"])
IMAGE_H, IMAGE_W = (int(v) for v in CONFIG["image_size"])
MODEL_PATH = MODEL_DIR / CONFIG["model_file"]
MODEL_VERSION = f"{CONFIG['model_name']}@{hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]}"


# ---------------------------------------------------------------- model
def load_model():
    """Load the model once and verify inference is deterministic.

    The saved model still contains its training augmentation layers (RandomFlip,
    RandomRotation, RandomZoom). Calling model(x) without training=False applies them,
    so the same photo would get a different answer on every request. Inference
    therefore always goes through a tf.function with training=False, and the
    self-check below refuses to start the server if that is ever broken.
    """
    started = time.perf_counter()
    model = keras.models.load_model(MODEL_PATH, compile=False)

    @tf.function(
        input_signature=[tf.TensorSpec([None, IMAGE_H, IMAGE_W, 3], tf.float32)],
        reduce_retracing=True,
    )
    def infer(batch):
        return model(batch, training=False)

    probe = np.random.default_rng(0).random((1, IMAGE_H, IMAGE_W, 3), dtype=np.float32)
    outputs = {float(infer(probe).numpy()[0, 0]) for _ in range(3)}
    reference = float(model.predict(probe, verbose=0)[0, 0])
    if len(outputs) != 1 or abs(outputs.pop() - reference) > 1e-5:
        raise RuntimeError("Startup self-check failed: inference is not deterministic.")

    log.info("Model %s loaded and self-check passed in %.1fs (threshold=%.2f)",
             MODEL_VERSION, time.perf_counter() - started, THRESHOLD)
    return infer


INFER = load_model()  # runs inside the gunicorn worker (never use --preload)


# ---------------------------------------------------------------- inference
class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def preprocess(data: bytes):
    """Exact notebook pipeline. Only addition: EXIF rotation for phone photos."""
    if not data:
        raise APIError(400, "empty_image", "The uploaded image is empty.")
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise APIError(400, "invalid_image", "The file is not a readable image.")
    image_format = img.format
    if image_format not in ALLOWED_FORMATS:
        raise APIError(415, "unsupported_format",
                       f"Unsupported format '{image_format}'. Allowed: {sorted(ALLOWED_FORMATS)}.")
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise APIError(413, "image_too_large", f"Image has more than {MAX_IMAGE_PIXELS:,} pixels.")
    try:
        img.load()
    except Exception:
        raise APIError(400, "invalid_image", "The image data is corrupted or truncated.")
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        log.warning("Ignoring unreadable EXIF orientation")

    img = img.convert("RGB").resize((IMAGE_W, IMAGE_H), Image.Resampling.BILINEAR)
    batch = np.asarray(img, dtype=np.float32)[np.newaxis, ...] / 255.0
    return batch, image_format


def build_result(rotten_probability: float) -> dict:
    """Same decision logic as predict_image() in the notebook (section 6.14)."""
    is_rotten = rotten_probability >= THRESHOLD
    confidence = rotten_probability if is_rotten else 1.0 - rotten_probability
    return {
        "predicted_class": "Rotten" if is_rotten else "Fresh",
        "confidence": round(confidence, 4),
        "rotten_probability": round(rotten_probability, 4),
        "threshold": round(THRESHOLD, 4),
        "is_rotten": bool(is_rotten),
    }


# ---------------------------------------------------------------- HTTP
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(MAX_UPLOAD_MB * 1024 * 1024)
app.json.sort_keys = False


def error(code, message, status):
    return jsonify({"error": {"code": code, "message": message}}), status


@app.errorhandler(APIError)
def handle_api_error(err):
    return error(err.code, err.message, err.status)


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_err):
    return error("file_too_large", f"Upload exceeds {MAX_UPLOAD_MB:g} MB.", 413)


@app.errorhandler(HTTPException)
def handle_http(err):
    return error((err.name or "http_error").lower().replace(" ", "_"), err.description, err.code)


@app.errorhandler(Exception)
def handle_unexpected(_err):
    log.exception("Unhandled error")
    return error("internal_error", "Unexpected server error.", 500)


@app.get("/health")
def health():
    return {"status": "ok", "model_version": MODEL_VERSION, "threshold": THRESHOLD}


@app.post("/predict")
def predict():
    if not hmac.compare_digest(request.headers.get("X-API-Key", "").encode(), API_KEY.encode()):
        raise APIError(401, "unauthorized", "Missing or invalid X-API-Key header.")
    if "image" not in request.files:
        raise APIError(400, "missing_image", "Send the photo as multipart/form-data field 'image'.")

    started = time.perf_counter()
    data = request.files["image"].read()
    batch, image_format = preprocess(data)
    probability = float(INFER(batch).numpy()[0, 0])
    if not math.isfinite(probability):
        raise APIError(500, "invalid_model_output", "Model returned an invalid value.")

    result = build_result(probability)
    result["model_version"] = MODEL_VERSION
    result["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
    log.info("predict format=%s bytes=%d class=%s p_rotten=%.4f ms=%.1f", image_format,
             len(data), result["predicted_class"], probability, result["processing_ms"])
    return result


if __name__ == "__main__":  # local debugging only; production runs gunicorn (see Docker/Dockerfile)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
