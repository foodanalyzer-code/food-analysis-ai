"""
Food Analysis AI - inference API (Flask + LiteRT). Built to fit Render's Free instance (512 MB).

Endpoints
  GET  /health   -> {"status": "ok", "model_version": "...", "threshold": 0.3, ...}
  POST /predict  -> header: X-API-Key: <key>
                    body:   multipart/form-data, ONE file in field "image"
                            (JPEG, PNG, WEBP, HEIC, MPO - max 15 MB, 32x32 to 24 MP)
                 <- {"predicted_class": "Fresh" | "Rotten", "confidence": 0.9864,
                     "rotten_probability": 0.0136, "threshold": 0.3, "is_rotten": false,
                     "model_version": "...", "processing_ms": 210.5, "request_id": "..."}
  Errors         <- {"error": {"code": "...", "message": "..."}, "request_id": "..."}
                    400 / 401 / 404 / 405 / 413 / 415 / 422 / 500 - always JSON, never a crash.
  Every response carries an X-Request-ID header (sent by the client or generated here).

Model (step 2 of deployment - step 1 is code/convert_model.py)
  The Docker build converts the notebook's EfficientNetB0_finetuned.keras into food_model.tflite
  (same float32 weights, training=False) and verifies it against the original. This server
  only loads that verified file plus conversion_report.json, and runs it with LiteRT
  (~0.12 GB RAM instead of ~0.8 GB for TensorFlow).

Contract (from the training notebook - do not change)
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
import re
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pillow_heif
from ai_edge_litert.interpreter import Interpreter
from flask import Flask, g, jsonify, request
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("food-ai")
pillow_heif.register_heif_opener()  # iPhone HEIC photos

# ---------------------------------------------------------------- settings
MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent.parent / "model"))
API_KEY = os.getenv("API_KEY", "").strip()
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "15"))
# 24 MP covers normal phone photos (12 MP / 24 MP) and keeps peak RAM well under 512 MB.
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "24000000"))
MIN_IMAGE_SIDE = 32
IMAGE_FIELD = "image"
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "HEIF"}
EXIF_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM, 5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE, 8: Image.Transpose.ROTATE_90,
}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

if not API_KEY:
    raise RuntimeError("API_KEY environment variable is required.")

CONFIG = json.loads((MODEL_DIR / "final_config.json").read_text(encoding="utf-8"))
if CONFIG.get("label_map") != {"Fresh": 0, "Rotten": 1}:
    raise RuntimeError(f"Unexpected label_map in final_config.json: {CONFIG.get('label_map')}")
if CONFIG.get("preprocessing") != {"color_mode": "RGB", "rescale": "1/255.0", "interpolation": "bilinear"}:
    raise RuntimeError(f"Unexpected preprocessing in final_config.json: {CONFIG.get('preprocessing')}")

THRESHOLD = float(CONFIG["threshold"])
IMAGE_H, IMAGE_W = (int(v) for v in CONFIG["image_size"])

# Written by convert_model.py during the Docker build.
REPORT = json.loads((MODEL_DIR / "conversion_report.json").read_text(encoding="utf-8"))
MODEL_PATH = MODEL_DIR / REPORT["tflite_model"]
if hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest() != REPORT["tflite_sha256"]:
    raise RuntimeError("food_model.tflite does not match conversion_report.json.")
# Version = the ORIGINAL notebook model's hash, so every result traces back to it.
MODEL_VERSION = f"{CONFIG['model_name']}@{REPORT['source_sha256'][:12]}"


# ---------------------------------------------------------------- model
class Model:
    """LiteRT interpreter + a lock (an interpreter must not be used by two threads at once)."""

    def __init__(self, path: Path):
        started = time.perf_counter()
        self._interpreter = Interpreter(model_path=str(path), num_threads=1)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]["index"]
        self._output = self._interpreter.get_output_details()[0]["index"]
        self.lock = threading.Lock()

        # Startup self-check: the probe from convert_model.py, run 3x, must give one value
        # equal to what the ORIGINAL Keras model produced for it.
        probe = np.random.default_rng(REPORT["probe_seed"]).random((1, IMAGE_H, IMAGE_W, 3), dtype=np.float32)
        outputs = {self.rotten_probability(probe) for _ in range(3)}
        if len(outputs) != 1 or abs(outputs.pop() - REPORT["probe_expected_output"]) > 1e-4:
            raise RuntimeError("Startup self-check failed: model output does not match the original model.")
        log.info("Model %s loaded and self-check passed in %.1fs (threshold=%.2f)",
                 MODEL_VERSION, time.perf_counter() - started, THRESHOLD)

    def rotten_probability(self, batch: np.ndarray) -> float:
        self._interpreter.set_tensor(self._input, batch)
        self._interpreter.invoke()
        return float(self._interpreter.get_tensor(self._output)[0, 0])


MODEL = Model(MODEL_PATH)


# ---------------------------------------------------------------- validation + preprocessing
class APIError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


def read_upload_stream():
    """Validate the multipart request and return the single uploaded file (not read yet)."""
    if request.mimetype != "multipart/form-data":
        raise APIError(400, "not_multipart",
                       "Send the request as multipart/form-data (Postman: Body -> form-data) "
                       f"with one File field named '{IMAGE_FIELD}'.")
    files = request.files.getlist(IMAGE_FIELD)
    if not files:
        if IMAGE_FIELD in request.form:
            raise APIError(400, "image_sent_as_text",
                           f"Field '{IMAGE_FIELD}' was sent as text. In Postman set its type to File.")
        received = sorted(set(request.files) | set(request.form))
        if received:
            raise APIError(400, "wrong_field_name",
                           f"No field named '{IMAGE_FIELD}'. Received: {received}. Rename the file field to '{IMAGE_FIELD}'.")
        raise APIError(400, "missing_image", f"No image received. Add one File field named '{IMAGE_FIELD}'.")
    if len(files) > 1:
        raise APIError(400, "multiple_images", f"Send exactly one image per request (received {len(files)}).")
    return files[0]


def to_rgb(img: Image.Image) -> Image.Image:
    """Any supported colour mode -> 8-bit RGB. Plain RGB images are left untouched (notebook path)."""
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):  # 16-bit grayscale PNG: scale, don't clip
        gray = np.clip(np.asarray(img, dtype=np.float32) / 257.0, 0, 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")
    return img.convert("RGB")  # RGBA, P, L, LA, CMYK, 1, ...


def preprocess(data: bytes):
    """Exact notebook pipeline for normal photos. Only addition: EXIF rotation for phone photos."""
    try:
        img = Image.open(io.BytesIO(data))
    except Image.DecompressionBombError:
        raise APIError(413, "image_too_large", f"Image has more than {MAX_IMAGE_PIXELS:,} pixels.")
    except Exception:
        raise APIError(400, "invalid_image", "The file is not a readable image.")

    image_format = img.format
    if image_format not in ALLOWED_FORMATS:
        raise APIError(415, "unsupported_format",
                       f"Unsupported format '{image_format}'. Allowed: JPEG, PNG, WEBP, HEIC.")
    if min(img.size) < MIN_IMAGE_SIDE:
        raise APIError(400, "image_too_small",
                       f"Image is {img.width}x{img.height}; minimum is {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE}.")
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise APIError(413, "image_too_large",
                       f"Image is {img.width}x{img.height}; maximum is {MAX_IMAGE_PIXELS:,} pixels.")
    try:
        img.load()
    except Image.DecompressionBombError:
        raise APIError(413, "image_too_large", f"Image has more than {MAX_IMAGE_PIXELS:,} pixels.")
    except Exception:
        raise APIError(400, "invalid_image", "The image data is corrupted or truncated.")
    try:
        orientation = img.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1

    try:
        img = to_rgb(img)
        img = img.resize((IMAGE_W, IMAGE_H), Image.Resampling.BILINEAR)
        # Phone EXIF rotation, applied to the small 224x224 image (no full-size copy).
        # Same mapping as PIL.ImageOps.exif_transpose; the output is square, so the result
        # matches rotating first. Photos without an EXIF orientation are untouched.
        if orientation in EXIF_TRANSPOSE:
            img = img.transpose(EXIF_TRANSPOSE[orientation])
        batch = np.asarray(img, dtype=np.float32)[np.newaxis, ...] / 255.0
    except Exception:
        log.exception("Image processing failed")
        raise APIError(422, "unprocessable_image", "The image could not be converted for the model.")
    if batch.shape != (1, IMAGE_H, IMAGE_W, 3):
        raise APIError(422, "unprocessable_image", "The image could not be converted for the model.")
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


@app.before_request
def assign_request_id():
    incoming = request.headers.get("X-Request-ID", "")
    g.request_id = incoming if REQUEST_ID_PATTERN.match(incoming) else uuid.uuid4().hex


@app.after_request
def add_request_id_header(response):
    response.headers["X-Request-ID"] = getattr(g, "request_id", "")
    return response


def error(code, message, status):
    request_id = getattr(g, "request_id", None)
    log.warning("rejected request_id=%s status=%d code=%s", request_id, status, code)
    return jsonify({"error": {"code": code, "message": message}, "request_id": request_id}), status


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
    log.exception("Unhandled error request_id=%s", getattr(g, "request_id", None))
    return error("internal_error", "Unexpected server error. Quote the request_id when reporting it.", 500)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "threshold": THRESHOLD,
        "conversion_max_difference": REPORT["max_abs_difference"],
    }


@app.post("/predict")
def predict():
    if not hmac.compare_digest(request.headers.get("X-API-Key", "").encode(), API_KEY.encode()):
        raise APIError(401, "unauthorized", "Missing or invalid X-API-Key header.")

    started = time.perf_counter()
    upload = read_upload_stream()
    # One photo at a time: keeps peak memory low on a 512 MB instance.
    with MODEL.lock:
        data = upload.read()
        if not data:
            raise APIError(400, "empty_image", "The uploaded file is empty.")
        batch, image_format = preprocess(data)
        probability = MODEL.rotten_probability(batch)
    if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise APIError(500, "invalid_model_output", "Model returned an invalid value.")

    result = build_result(probability)
    result["model_version"] = MODEL_VERSION
    result["processing_ms"] = round((time.perf_counter() - started) * 1000, 1)
    result["request_id"] = g.request_id
    log.info("predict request_id=%s format=%s bytes=%d class=%s p_rotten=%.4f ms=%.1f", g.request_id,
             image_format, len(data), result["predicted_class"], probability, result["processing_ms"])
    return result


if __name__ == "__main__":  # local debugging only; production runs gunicorn (see Docker/Dockerfile)
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
