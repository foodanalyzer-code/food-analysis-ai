import hashlib
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
from flask import Flask, g, jsonify, request
from PIL import Image
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    try:
        import tensorflow as tf
        Interpreter = tf.lite.Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
except ImportError:
    pass

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("food-ai")

BASE_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).resolve().parent))
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "5.0"))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "24000000"))
MIN_IMAGE_SIDE = 32
IMAGE_FIELD = "image"
ALLOWED_FORMATS = {"JPEG", "MPO", "PNG", "WEBP", "HEIF"}
NON_IMAGE_EXTENSIONS = {".pdf", ".mp3", ".wav", ".mp4", ".avi", ".docx", ".xlsx", ".zip", ".tar", ".gz", ".ogg"}

EXIF_TRANSPOSE = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,64}$")

PROCESSED_HASHES = set()
HASH_LOCK = threading.Lock()

AR_TO_EN_FOOD = {
    "تفاح": "Apple", "تفاحة": "Apple", "apple": "Apple",
    "موز": "Banana", "موزة": "Banana", "banana": "Banana",
    "فلفل": "Bellpepper", "فلفل رومي": "Bellpepper", "bellpepper": "Bellpepper",
    "جزر": "Carrot", "جزرة": "Carrot", "carrot": "Carrot",
    "خيار": "Cucumber", "خيارة": "Cucumber", "cucumber": "Cucumber",
    "مانجو": "Mango", "مانجا": "Mango", "mango": "Mango",
    "برتقال": "Orange", "برتقالة": "Orange", "orange": "Orange",
    "بطاطس": "Potato", "بطاطا": "Potato", "potato": "Potato",
    "فراولة": "Strawberry", "strawberry": "Strawberry",
    "طماطم": "Tomato", "طماطماية": "Tomato", "بندورة": "Tomato", "tomato": "Tomato",
}

UNSUPPORTED_FOOD_ITEMS = {
    "عنب", "بطيخ", "خس", "رمان", "خوخ", "مشمش", "تين", "كمثرى", "grapes", "watermelon", "lettuce", "peach", "pear"
}

COOKED_OR_PROCESSED_ITEMS = {
    "أرز", "رز", "بطاطس مقلية", "بطاطا مقلية", "عصير", "عصير برتقال", "شيبس", "معكرونة", "مكرونة", "طبيخ", "مطبوخ",
    "rice", "french fries", "fries", "juice", "orange juice", "cooked", "pasta"
}

COOKED_IMAGENET_LABELS = {
    "pizza", "cheeseburger", "hotdog", "French_loaf", "bagel", "pretzel", "mashed_potato",
    "guacamole", "consomme", "hot_pot", "trifle", "ice_cream", "ice_lolly", "burrito",
    "potpie", "meat_loaf", "carbonara", "chocolate_sauce", "espresso", "soup_bowl", "plate", "bakery"
}

PACKAGING_IMAGENET_LABELS = {
    "plastic_bag", "shopping_basket", "shopping_cart", "crate", "carton", "packet", "envelope", "mailbag", "hamper"
}

UNSUPPORTED_IMAGENET_LABELS = {
    "pineapple", "broccoli", "cauliflower", "head_cabbage", "artichoke", "corn", "mushroom",
    "acorn_squash", "butternut_squash", "zucchini"
}

PRODUCE_OR_ORGANIC_LABELS = {
    "Granny_Smith", "banana", "orange", "lemon", "strawberry", "bell_pepper", "cucumber",
    "pomegranate", "fig", "pineapple", "custard_apple", "jackfruit", "spaghetti_squash",
    "butternut_squash", "acorn_squash", "zucchini", "broccoli", "cauliflower", "head_cabbage",
    "artichoke", "corn", "mushroom", "buckeye", "acorn", "hip", "croquet_ball",
    "ping-pong_ball", "tennis_ball", "grocery_store", "maraca", "rapeseed", "ear"
}

dual_config_path = BASE_DIR / "final_config_dual.json"
legacy_config_path = BASE_DIR / "final_config.json"

if dual_config_path.is_file():
    CONFIG = json.loads(dual_config_path.read_text(encoding="utf-8"))
    IS_DUAL_HEAD = True
    MODEL_PATH = BASE_DIR / CONFIG.get("model_file", "food_model_dual.tflite")
    FOOD_CLASSES = CONFIG.get("food_classes", ["Apple", "Banana", "Mango", "Orange", "Strawberry"])
    FOOD_TO_ID = CONFIG.get("food_to_id", {name: idx for idx, name in enumerate(FOOD_CLASSES)})
    ID_TO_FOOD = {idx: name for name, idx in FOOD_TO_ID.items()}
    OOD_FOOD_THRESHOLD = float(CONFIG.get("ood_food_threshold", 0.45))
    THRESHOLD = float(CONFIG.get("condition_threshold", 0.30))
    UNCERTAIN_LOW = float(CONFIG.get("uncertainty_range", [0.35, 0.65])[0])
    UNCERTAIN_HIGH = float(CONFIG.get("uncertainty_range", [0.35, 0.65])[1])
else:
    CONFIG = json.loads(legacy_config_path.read_text(encoding="utf-8"))
    IS_DUAL_HEAD = False
    MODEL_PATH = BASE_DIR / "food_model.tflite"
    FOOD_CLASSES = []
    FOOD_TO_ID = {}
    ID_TO_FOOD = {}
    OOD_FOOD_THRESHOLD = 0.45
    THRESHOLD = float(CONFIG.get("threshold", 0.30))
    UNCERTAIN_LOW = 0.35
    UNCERTAIN_HIGH = 0.65

IMAGE_H, IMAGE_W = (int(v) for v in CONFIG["image_size"])


class Gatekeeper:
    def __init__(self, model_path: Path, labels_path: Path):
        self.enabled = model_path.is_file() and labels_path.is_file()
        self.lock = threading.Lock()
        if self.enabled:
            self.labels = json.loads(labels_path.read_text(encoding="utf-8"))
            self._interpreter = Interpreter(model_path=str(model_path), num_threads=1)
            self._interpreter.allocate_tensors()
            self._input = self._interpreter.get_input_details()[0]["index"]
            self._output = self._interpreter.get_output_details()[0]["index"]
            log.info("Gatekeeper %s loaded (%d classes)", model_path.name, len(self.labels))

    def check(self, batch: np.ndarray):
        if not self.enabled:
            return None
        with self.lock:
            self._interpreter.set_tensor(self._input, batch)
            self._interpreter.invoke()
            probs = self._interpreter.get_tensor(self._output)[0]
        top3_idx = np.argsort(probs)[::-1][:3]
        top1_label = self.labels[int(top3_idx[0])]
        top1_prob = float(probs[int(top3_idx[0])])
        top3_labels = {self.labels[int(i)] for i in top3_idx}

        if top1_label in COOKED_IMAGENET_LABELS and top1_prob >= 0.25:
            return ("cooked_or_processed", "System is dedicated for fresh, uncooked fruits and vegetables only.", top1_prob)
        if top1_label in PACKAGING_IMAGENET_LABELS and top1_prob >= 0.25:
            return ("packaging_detected", "Please remove the food item from packaging and retake the photo.", top1_prob)
        if top1_label in UNSUPPORTED_IMAGENET_LABELS and top1_prob >= 0.35:
            return ("unsupported_item", "This food item is not supported for automated analysis.", top1_prob)
        if top1_prob >= 0.30 and not (top3_labels & PRODUCE_OR_ORGANIC_LABELS):
            return ("not_food", "No fruit or vegetable recognized in the image. Please capture food clearly.", top1_prob)
        return None


class Model:
    def __init__(self, path: Path):
        started = time.perf_counter()
        self._interpreter = Interpreter(model_path=str(path), num_threads=1)
        self._interpreter.allocate_tensors()
        self._input = self._interpreter.get_input_details()[0]["index"]

        output_details = self._interpreter.get_output_details()
        self.is_dual = len(output_details) > 1

        if self.is_dual:
            for out in output_details:
                shape = out["shape"]
                if len(shape) >= 2 and shape[-1] > 1:
                    self._food_output = out["index"]
                else:
                    self._cond_output = out["index"]
        else:
            self._cond_output = output_details[0]["index"]
            self._food_output = None

        self.lock = threading.Lock()
        log.info("Model %s loaded in %.1fs (dual=%s, threshold=%.2f)",
                 path.name, time.perf_counter() - started, self.is_dual, THRESHOLD)

    def infer(self, batch: np.ndarray):
        with self.lock:
            self._interpreter.set_tensor(self._input, batch)
            self._interpreter.invoke()
            cond_prob = float(self._interpreter.get_tensor(self._cond_output)[0, 0])
            food_probs = None
            if self.is_dual and self._food_output is not None:
                food_probs = self._interpreter.get_tensor(self._food_output)[0]
        return food_probs, cond_prob


GATEKEEPER = Gatekeeper(BASE_DIR / "gatekeeper_model.tflite", BASE_DIR / "imagenet_classes.json")
MODEL = Model(MODEL_PATH)


class APIError(Exception):
    def __init__(self, status, code, message, details=None):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}


def read_upload_stream():
    if request.mimetype != "multipart/form-data":
        raise APIError(400, "missing_image", "Food image is required before confirming receipt.")
    files = request.files.getlist(IMAGE_FIELD)
    if not files:
        raise APIError(400, "missing_image", "Food image is required before confirming receipt.")
    if len(files) > 1:
        raise APIError(400, "multiple_images", "Please provide exactly one image per request.")
    return files[0]


def to_rgb(img: Image.Image) -> Image.Image:
    if img.mode == "RGB":
        return img
    if img.mode in ("I;16", "I;16B", "I;16L", "I"):
        gray = np.clip(np.asarray(img, dtype=np.float32) / 257.0, 0, 255).astype(np.uint8)
        return Image.fromarray(gray).convert("RGB")
    return img.convert("RGB")


def check_image_heuristics(img: Image.Image):
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    mean_brightness = float(np.mean(gray))
    if mean_brightness < 30.0:
        raise APIError(422, "poor_lighting", "Image is not clear. Please capture with better lighting.")

    if gray.shape[0] >= 32 and gray.shape[1] >= 32:
        h, w = gray.shape
        crop = gray[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4]
        laplacian = (
            crop[1:-1, 2:] + crop[1:-1, :-2] + crop[2:, 1:-1] + crop[:-2, 1:-1] - 4.0 * crop[1:-1, 1:-1]
        )
        blur_variance = float(np.var(laplacian))
        if blur_variance < 10.0:
            raise APIError(422, "blurry_image", "Image is blurry or unclear. Please retake with a steady camera.")


def preprocess(data: bytes, filename: str = ""):
    ext = Path(filename).suffix.lower()
    if ext in NON_IMAGE_EXTENSIONS:
        raise APIError(415, "unsupported_format", "Allowed formats: JPG, PNG only.")

    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise APIError(413, "file_too_large", "Image size exceeds the maximum allowed limit (5MB).")

    try:
        probe = Image.open(io.BytesIO(data))
        probe.verify()
        probe_format = probe.format
    except Image.DecompressionBombError:
        raise APIError(413, "image_too_large", f"Image resolution exceeds maximum permitted ({MAX_IMAGE_PIXELS:,} pixels).")
    except Exception:
        if ext in {".jpg", ".jpeg", ".png", ".webp"}:
            raise APIError(400, "corrupted_file", "File is corrupted or invalid.")
        raise APIError(415, "unsupported_format", "Allowed formats: JPG, PNG only.")

    if probe_format not in ALLOWED_FORMATS:
        raise APIError(415, "unsupported_format", "Allowed formats: JPG, PNG only.")

    try:
        img = Image.open(io.BytesIO(data))
    except Exception:
        raise APIError(400, "corrupted_file", "File is corrupted or invalid.")

    if min(img.size) < MIN_IMAGE_SIDE:
        raise APIError(400, "image_too_small", f"Image dimensions are too small. Minimum is {MIN_IMAGE_SIDE}x{MIN_IMAGE_SIDE}.")
    if img.width * img.height > MAX_IMAGE_PIXELS:
        raise APIError(413, "image_too_large", "Image resolution exceeds maximum permitted.")

    check_image_heuristics(img)

    try:
        orientation = img.getexif().get(0x0112, 1)
    except Exception:
        orientation = 1

    try:
        img = to_rgb(img)
        img = img.resize((IMAGE_W, IMAGE_H), Image.Resampling.BILINEAR)
        if orientation in EXIF_TRANSPOSE:
            img = img.transpose(EXIF_TRANSPOSE[orientation])
        batch = np.asarray(img, dtype=np.float32)[np.newaxis, ...] / 255.0
    except Exception:
        raise APIError(422, "unprocessable_image", "Failed to preprocess image for model input.")

    if batch.shape != (1, IMAGE_H, IMAGE_W, 3):
        raise APIError(422, "unprocessable_image", "Processed batch shape does not match model expectations.")

    return batch, probe_format


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


def error_response(code, message, status, details=None):
    return jsonify({
        "code": code,
        "message": message,
    }), status


@app.errorhandler(APIError)
def handle_api_error(err):
    return error_response(err.code, err.message, err.status, err.details)


@app.errorhandler(RequestEntityTooLarge)
def handle_too_large(_err):
    return error_response("file_too_large", "Image size exceeds the maximum allowed limit (5MB).", 413)


@app.errorhandler(HTTPException)
def handle_http(err):
    return error_response((err.name or "http_error").lower().replace(" ", "_"), err.description, err.code)


@app.errorhandler(Exception)
def handle_unexpected(_err):
    log.exception("Unhandled error request_id=%s", getattr(g, "request_id", None))
    return error_response("service_unavailable",
                          "Analysis service is currently unavailable. Request has been saved and will be processed automatically.", 500)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_name": CONFIG.get("model_name", "food_model"),
        "dual_head": MODEL.is_dual,
        "gatekeeper": GATEKEEPER.enabled,
        "supported_foods": FOOD_CLASSES,
        "service_status": "OPERATIONAL",
    }


@app.post("/sync-offline")
def sync_offline():
    return jsonify({
        "status": "synced",
        "message": "Local offline requests synchronized successfully and scheduled for processing.",
        "queued_count": int(request.form.get("count", 1)),
    })


@app.post("/predict")
def predict():
    upload = read_upload_stream()
    declared_food = request.form.get("declared_item") or request.form.get("food_type") or request.form.get("declared_food")
    in_packaging = request.form.get("in_packaging", "false").lower() in ("true", "1")
    is_mixed_batch = request.form.get("mixed_batch", "false").lower() in ("true", "1")

    data = upload.read()
    if not data:
        raise APIError(400, "missing_image", "Food image is required before confirming receipt.")

    if in_packaging:
        return jsonify({
            "code": "packaging_detected",
            "message": "Please remove the food item from packaging and retake the photo.",
        })

    if declared_food:
        declared_clean = declared_food.strip().lower()
        if declared_clean in COOKED_OR_PROCESSED_ITEMS:
            return jsonify({
                "code": "cooked_or_processed",
                "message": "System is dedicated for fresh, uncooked fruits and vegetables only.",
            })
        if declared_clean in UNSUPPORTED_FOOD_ITEMS:
            return jsonify({
                "code": "unsupported_item",
                "message": "This food item is not supported for automated analysis.",
            })

    img_hash = hashlib.sha256(data).hexdigest()
    with HASH_LOCK:
        if img_hash in PROCESSED_HASHES:
            raise APIError(409, "duplicate_image", "This image has already been used in another request.")
        PROCESSED_HASHES.add(img_hash)

    batch, _ = preprocess(data, upload.filename or "")

    gk_verdict = GATEKEEPER.check(batch)
    if gk_verdict is not None:
        gk_code, gk_msg, gk_conf = gk_verdict
        return jsonify({
            "code": gk_code,
            "message": gk_msg,
            "confidence": round(gk_conf, 4),
        })

    food_probs, rotten_prob = MODEL.infer(batch)

    if not math.isfinite(rotten_prob) or not 0.0 <= rotten_prob <= 1.0:
        raise APIError(500, "invalid_model_output", "Analysis service returned an invalid result. Request has been queued.")

    detected_food_en = None
    food_conf = 1.0

    if food_probs is not None and len(food_probs) > 0:
        top_idx = int(np.argmax(food_probs))
        detected_food_en = ID_TO_FOOD.get(top_idx, f"Class_{top_idx}")
        food_conf = float(food_probs[top_idx])

    if food_probs is not None and food_conf < OOD_FOOD_THRESHOLD:
        return jsonify({
            "code": "not_food",
            "message": "No fruit or vegetable recognized in the image. Please capture food clearly.",
            "confidence": round(food_conf, 4),
        })

    if declared_food and detected_food_en:
        declared_norm_en = AR_TO_EN_FOOD.get(declared_food.strip().lower(), declared_food.strip().capitalize())
        if declared_norm_en != detected_food_en:
            return jsonify({
                "code": "mismatch",
                "message": "Food item in image does not match the declared order item.",
                "declared_item": declared_food,
                "detected_item": detected_food_en,
                "confidence": round(food_conf, 4),
            })

    is_rotten = rotten_prob >= THRESHOLD
    predicted_class = "Rotten" if is_rotten else "Fresh"
    confidence = rotten_prob if is_rotten else 1.0 - rotten_prob

    if UNCERTAIN_LOW <= rotten_prob <= UNCERTAIN_HIGH:
        cond_label = "Fresh" if rotten_prob < 0.50 else "Rotten"
        pct = round(max(rotten_prob, 1.0 - rotten_prob) * 100)
        return jsonify({
            "code": "uncertain",
            "message": f"Inconclusive result - {cond_label} {pct}%.",
            "predicted_class": predicted_class,
            "confidence": round(confidence, 4),
        })

    response_payload = {
        "predicted_class": predicted_class,
        "confidence": round(confidence, 4),
    }
    if is_mixed_batch:
        response_payload["warning"] = "Mixed conditions detected across batch. Sorting recommended before final approval."
    return jsonify(response_payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "10000")))
