# Build & Deploy Prompt (V2, simple) — Food Analysis AI

> Save this file as `BUILD_PROMPT_V2.md` in `D:\Happy Software\Food Analysis\Food_Analysis_AI_Week_9\`, start a **new** Claude Code conversation, and send:
> `Read BUILD_PROMPT_V2.md and execute it step by step.`

---

## 0. Role

You are a senior MLOps engineer. Put an already-trained image model behind a small, simple API, test it in Docker, and guide the user to deploy it on Render.

- Talk to the user in **Egyptian Arabic**.
- Code, comments, and commit messages stay in **English**.
- Work step by step. After each step, report what you did and what you verified.
- If something fails, stop, show the exact error, and propose a fix.

**Environment:** Windows 10 with PowerShell and Docker Desktop. Use `curl.exe` (not `curl`, which is an alias in PowerShell).

## 1. Context

A platform where citizens donate surplus fruit and vegetables. A pickup agent uploads a photo of the food. This API returns **Fresh** or **Rotten** plus a confidence score. A Ministry of Health employee always reviews the result; the AI is never the final decision. The main backend calls this API over HTTP.

**Model:** `EfficientNetB0_finetuned`, trained in Google Colab. It is final.

**Test-set results:**
- Accuracy 98.22%
- Macro F1 98.21%
- Rotten recall 97.89%
- Decision threshold 0.30

## 2. Rules

1. **Do not modify the model.**
   - Never modify, re-save, or convert `EfficientNetB0_finetuned.keras`.
   - Never edit `final_config.json`.
2. **Write the 3 code files in section 6 exactly as given.** If you think something must change, explain why and ask the user first. Do not add files beyond the structure in section 4.
3. **Keep the pinned versions.** Do not upgrade any version in `requirements.txt`. The model was saved with Keras 3.13.2 and TensorFlow 2.21.0.
4. **Never handle secrets or credentials.**
   - Never see, print, or commit the production API key.
   - Never create accounts or type passwords.
   - The user does logins, payments, and key generation; you give exact instructions and wait.
5. **Ask for confirmation before** `git push`, deleting anything, or anything that costs money.

## 3. Step 1 — Clean up the previous attempt

A previous, more complex version of this project was started in this folder.

1. Stop and remove any containers created from the old image:
   ```powershell
   docker ps -a --filter "ancestor=food-ai"
   docker rm -f <id>
   ```
2. **After the user confirms**, delete the old folder `food-analysis-ai-service\` and the old `BUILD_PROMPT.md`.
3. Keep these files in `Food_Analysis_AI_Week_9\`:
   - `EfficientNetB0_finetuned.keras`
   - `final_config.json`
   - `test_food.jpg`
   - `BUILD_PROMPT_V2.md`

## 4. Step 2 — Project structure

Create `food-analysis-ai-service\` inside `Food_Analysis_AI_Week_9\`, with **only** this content:

```text
food-analysis-ai-service/
├── code/
│   └── app.py                 # the whole API (Flask)
├── model/
│   ├── EfficientNetB0_finetuned.keras
│   └── final_config.json
├── Docker/
│   └── Dockerfile
└── requirements.txt
```

## 5. Step 3 — Model files

Copy the two model files from `Food_Analysis_AI_Week_9\` into `food-analysis-ai-service\model\` with `Copy-Item`. Do not open them or re-save them.

Verify both hashes with `Get-FileHash -Algorithm SHA256`:

| File | Size (bytes) | SHA-256 |
|---|---|---|
| `EfficientNetB0_finetuned.keras` | 35,572,449 | `0d8628ec8e8175f1176e95dd0c5d92cf177d06db94e1b0cecf3eb7af83895c05` |
| `final_config.json` | 724 | `fbac93e72fb5763c2ca2fcedd104a54ca61e5804582a3260f806c902474ebe0f` |

If either hash differs, **stop** and tell the user.

## 6. Step 4 — Write the files (verbatim)

### `code/app.py`
```python
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
```

### `requirements.txt`
```text
# Pinned to the exact versions used to train the model in Colab. Do not upgrade.
tensorflow-cpu==2.21.0
keras==3.13.2
numpy==2.1.3
pillow==11.3.0
pillow-heif==1.8.1
flask==3.1.3
gunicorn==26.2.0
```

### `Docker/Dockerfile`
```dockerfile
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
```

## 7. Step 5 — Build and test in Docker

Run everything from `food-analysis-ai-service\`. The first build downloads TensorFlow (~260 MB) and takes a few minutes.

1. **Build the image:**
   ```powershell
   docker build -f Docker/Dockerfile -t food-ai .
   ```
2. **Start the container:**
   ```powershell
   docker run -d --name food-ai-test -p 10000:10000 -e API_KEY=dev-key food-ai
   ```
3. **Wait for startup.** Run `docker logs food-ai-test` until it shows `Model EfficientNetB0_finetuned@0d8628ec8e81 loaded and self-check passed`. If it shows `Startup self-check failed`, stop: `app.py` was changed.
4. **Run the checks.** Show each result to the user:

   | Check | Command | Expected |
   |---|---|---|
   | Health | `curl.exe -s http://localhost:10000/health` | `"status":"ok"`, `model_version` = `EfficientNetB0_finetuned@0d8628ec8e81`, `threshold` = 0.3 |
   | Prediction | `curl.exe -s -X POST http://localhost:10000/predict -H "X-API-Key: dev-key" -F "image=@D:\Happy Software\Food Analysis\Food_Analysis_AI_Week_9\test_food.jpg"` | HTTP 200 with `predicted_class`, `confidence`, `rotten_probability`, `threshold`, `is_rotten`, `model_version`, `processing_ms` |
   | Same photo again | Same command as the prediction | Identical `rotten_probability` |
   | No key | The prediction command without `-H "X-API-Key: dev-key"` | 401 `unauthorized` |
   | Not an image | `-F "image=@...\BUILD_PROMPT_V2.md"` with the key | 400 `invalid_image` |

5. **Interpret the prediction.** Explain the result to the user:
   - `confidence` is the confidence in `predicted_class`.
   - With threshold 0.30, a Rotten result can have confidence as low as 0.30.
6. **Clean up:**
   ```powershell
   docker rm -f food-ai-test
   ```

**Measured on 1 CPU:**
- About 0.7–0.85 GB of RAM.
- 25–50 ms per photo for the model, up to ~0.3 s total for a 12 MP phone photo.
- 8 concurrent 10 MB uploads all succeeded.

## 8. Step 6 — Git and GitHub

1. **Check Git.** Make sure Git is installed (`git --version`). If it is missing, tell the user how to install it.
2. **Commit locally.** In `food-analysis-ai-service\`, run `git init -b main`, then `git add .`.
   - `git status` must show exactly 5 files:
     - `code/app.py`
     - `requirements.txt`
     - `Docker/Dockerfile`
     - `model/EfficientNetB0_finetuned.keras`
     - `model/final_config.json`
   - Commit: `Food Analysis AI inference API`.
   - Do **not** use Git LFS. The model is ~35.6 MB, which is fine for GitHub.
3. **Create the GitHub repository.** Ask the user to create an **empty private** repository on GitHub named `food-analysis-ai-service`, with no README, and to give you its URL.
4. **Push.** After confirmation, run `git remote add origin <url>` and `git push -u origin main`. If Git asks for login, the user completes it in the browser window.
5. **Verify.** Ask the user to open the repo and confirm the model file shows ~35.6 MB.

## 9. Step 7 — Deploy on Render

The user performs the account, login, and payment steps. Guide them one step at a time and wait after each.

**Before you start, explain the cost.**
- The service needs Render's **Standard** instance: 1 CPU / 2 GB RAM, about **$25/month**. Ask the user to confirm the price on `render.com/pricing`.
- **Free** and **Starter** have only 512 MB of RAM and will crash with out-of-memory.
- Free also sleeps after 15 minutes of inactivity.

1. **Generate the API key.** The user runs this **in their own PowerShell window, not through you**, and saves the output privately:
   ```powershell
   docker run --rm python:3.13-slim python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```
2. **Sign up.** The user signs up at `render.com` with GitHub and adds a payment method when asked.
3. **Create the service.** **New → Web Service**, then select the `food-analysis-ai-service` repository. If it is not listed, the user grants Render access to it.
4. **Fill the form:**

   | Field | Value |
   |---|---|
   | Name | `food-analysis-ai` |
   | Language / Runtime | `Docker` |
   | Branch | `main` |
   | Region | `Frankfurt`. Closest to Egypt, **cannot be changed later**. If the main backend is hosted on Render in another region, use that region instead. |
   | Root Directory | leave empty |
   | Instance Type | **Standard** (1 CPU, 2 GB) |
   | Environment Variable | `API_KEY` = the key from step 1 |
   | Advanced → Dockerfile Path | `./Docker/Dockerfile` |
   | Advanced → Docker Build Context Directory | `.` |
   | Advanced → Health Check Path | `/health` |

5. **Deploy and watch the logs.** The user clicks **Create Web Service**. In **Logs**, these should appear:
   - `Listening at: http://0.0.0.0:10000`
   - `Model EfficientNetB0_finetuned@0d8628ec8e81 loaded and self-check passed`
   - the status becomes **Live**.
6. **Test the live service.** Give the user these two commands to run **in their own PowerShell** (with their real key), and ask them to paste back only the JSON output:
   ```powershell
   curl.exe -s https://<service-url>/health
   curl.exe -s -X POST https://<service-url>/predict -H "X-API-Key: <their key>" -F "image=@D:\Happy Software\Food Analysis\Food_Analysis_AI_Week_9\test_food.jpg"
   ```
   The result must match the local Docker result for the same photo.

## 10. Step 8 — Handoff summary

Give the user a short summary in Egyptian Arabic for the backend team. It must cover:

1. **The service URL.**
2. **Authentication:** the header `X-API-Key`. The key must be shared only through a private channel.
3. **The request:** `POST /predict`, `multipart/form-data`, field name `image`. Accepted formats: JPEG, PNG, WEBP, HEIC. Max 15 MB.
4. **The response fields:**
   - `predicted_class` (`Fresh` / `Rotten`)
   - `confidence`
   - `rotten_probability`
   - `threshold`
   - `is_rotten`
   - `model_version`
   - `processing_ms`
5. **The error format:** `{"error": {"code", "message"}}` with codes 400, 401, 413, 415, 500.
6. **What to save with each donation order:**
   - `predicted_class`
   - `confidence`
   - `rotten_probability`
   - `model_version`
7. **UI rules:**
   - Show confidence ≥ 0.999 as "more than 99.9%", never "100%".
   - The Ministry employee's decision is final.

## 11. Troubleshooting

| Symptom | Fix |
|---|---|
| Render: "Ran out of memory" / restarts | Instance must be **Standard** (2 GB) |
| Render: "Dockerfile not found" | Dockerfile Path `./Docker/Dockerfile`, Build Context `.` |
| Render: "No open ports detected" | Do not change the Dockerfile `CMD`; it binds to `0.0.0.0:$PORT` |
| `Startup self-check failed` | `app.py` was edited; restore it from section 6 |
| Model load error (deserialization) | Versions were changed; restore `requirements.txt` exactly |
| 401 | Header must be exactly `X-API-Key` |
| 413 / 415 | Photo over 15 MB / unsupported format (send JPEG, PNG, WEBP, or HEIC) |

## 12. Done when

- [ ] Both hashes match section 5.
- [ ] The local Docker checks in section 7 all pass.
- [ ] A private GitHub repo contains exactly the 5 files.
- [ ] The Render service is **Live** on **Standard**, and `/health` returns `"status":"ok"`.
- [ ] The live `/predict` test returned a correct result.
- [ ] The handoff summary was given to the user.
