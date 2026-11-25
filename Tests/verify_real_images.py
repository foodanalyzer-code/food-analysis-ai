import io
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT_DIR / "Code"
SAMPLE_DIR = CODE_DIR / "sample_images"
sys.path.insert(0, str(CODE_DIR))

from app import app, PROCESSED_HASHES

client = app.test_client()
image_files = sorted(list(SAMPLE_DIR.glob("*.jpg")) + list(SAMPLE_DIR.glob("*.png")))

print("=" * 105)
print(f"Testing {len(image_files)} REAL images from {SAMPLE_DIR}:")
print("=" * 105)
print(f"{'File Name':<35} | {'Status':<10} | {'Food Detected':<13} | {'Food Conf':<10} | {'Condition':<10} | {'Cond Conf':<10}")
print("-" * 105)

for img_path in image_files:
    PROCESSED_HASHES.clear()
    img_bytes = img_path.read_bytes()

    res = client.post("/predict", data={"image": (io.BytesIO(img_bytes), img_path.name)}, content_type="multipart/form-data")
    data = res.get_json() or {}

    if "code" in data:
        status = data["code"]
        food = "REJECTED" if status == "not_food" else status
        food_conf = f"{data.get('confidence', 0.0):.1%}" if "confidence" in data else "-"
        cond = "-"
        cond_conf = "-"
    else:
        status = "success"
        food = data.get("detected_food", "-")
        food_conf = f"{data.get('food_confidence', 0.0):.1%}" if "food_confidence" in data else "-"
        cond = data.get("condition", "-")
        cond_conf = f"{data.get('condition_confidence', 0.0):.1%}" if "condition_confidence" in data else "-"

    print(f"{img_path.name[:35]:<35} | {status:<10} | {food:<13} | {food_conf:<10} | {cond:<10} | {cond_conf:<10}")
print("=" * 105)
