import io
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
CODE_DIR = ROOT_DIR / "Code"
SAMPLE_DIR = CODE_DIR / "sample_images"
sys.path.insert(0, str(CODE_DIR))

from app import app, PROCESSED_HASHES

client = app.test_client()


def make_dummy_image(color=(128, 128, 128), size=(100, 100), add_noise=True):
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    if add_noise:
        noise = np.random.randint(-20, 20, arr.shape, dtype=np.int16)
        arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    bio = io.BytesIO()
    Image.fromarray(arr).save(bio, format="JPEG")
    bio.seek(0)
    return bio


def load_real_sample(filename="freshApple (145).jpg"):
    sample_path = SAMPLE_DIR / filename
    if sample_path.is_file():
        return io.BytesIO(sample_path.read_bytes())
    return make_dummy_image((200, 50, 50))


test_results = []


def record(tc_id, scenario, priority, actual, passed):
    test_results.append({
        "tc_id": tc_id,
        "scenario": scenario,
        "priority": priority,
        "actual": actual,
        "status": "PASS" if passed else "FAIL",
    })


res_tc12 = client.post("/predict", data={}, content_type="multipart/form-data")
json_12 = res_tc12.get_json() or {}
record("TC-12", "Submit without image", "HIGH", json_12.get("message", ""), res_tc12.status_code == 400 and json_12.get("code") == "missing_image")

res_tc13 = client.post("/predict", data={"image": (io.BytesIO(b"%PDF-1.4"), "doc.pdf")}, content_type="multipart/form-data")
json_13 = res_tc13.get_json() or {}
record("TC-13", "Non-image file", "HIGH", json_13.get("message", ""), res_tc13.status_code == 415 and json_13.get("code") == "unsupported_format")

res_tc14 = client.post("/predict", data={"image": (io.BytesIO(b"0" * (6 * 1024 * 1024)), "big.jpg")}, content_type="multipart/form-data")
json_14 = res_tc14.get_json() or {}
record("TC-14", "File size > 5MB", "MEDIUM", json_14.get("message", ""), res_tc14.status_code == 413 and json_14.get("code") == "file_too_large")

res_tc15 = client.post("/predict", data={"image": (io.BytesIO(b"plain text"), "fake.jpg")}, content_type="multipart/form-data")
json_15 = res_tc15.get_json() or {}
record("TC-15", "Corrupted file", "HIGH", json_15.get("message", ""), res_tc15.status_code == 400 and json_15.get("code") == "corrupted_file")

dark_bio = io.BytesIO()
Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(dark_bio, format="JPEG")
dark_bio.seek(0)
res_tc16 = client.post("/predict", data={"image": (dark_bio, "dark.jpg")}, content_type="multipart/form-data")
json_16 = res_tc16.get_json() or {}
record("TC-16", "Dark image", "HIGH", json_16.get("message", ""), res_tc16.status_code == 422 and json_16.get("code") == "poor_lighting")

PROCESSED_HASHES.clear()
client.post("/predict", data={"image": (load_real_sample("freshApple (145).jpg"), "a1.jpg")}, content_type="multipart/form-data")
res_tc17 = client.post("/predict", data={"image": (load_real_sample("freshApple (145).jpg"), "a2.jpg")}, content_type="multipart/form-data")
json_17 = res_tc17.get_json() or {}
record("TC-17", "Duplicate image", "CRITICAL", json_17.get("message", ""), res_tc17.status_code == 409 and json_17.get("code") == "duplicate_image")

PROCESSED_HASHES.clear()
res_tc07 = client.post("/predict", data={"image": (load_real_sample("freshBanana (6).jpg"), "b.jpg"), "declared_item": "Apple"}, content_type="multipart/form-data")
json_07 = res_tc07.get_json() or {}
record("TC-07", "Item mismatch", "HIGH", f"{json_07.get('predicted_class')} | {json_07.get('alert', '')}", "alert" in json_07 and "predicted_class" in json_07)

print("\n" + "=" * 95)
print(f"{'TC ID':<8} | {'Scenario':<25} | {'Priority':<8} | {'Status':<6} | {'Actual Response'}")
print("=" * 95)
for r in test_results:
    print(f"{r['tc_id']:<8} | {r['scenario']:<25} | {r['priority']:<8} | {r['status']:<6} | {r['actual']}")
print("=" * 95)
print(f"\nPassed: {sum(1 for r in test_results if r['status'] == 'PASS')}/{len(test_results)}")
