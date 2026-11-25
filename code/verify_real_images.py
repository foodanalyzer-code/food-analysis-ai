import sys
sys.path.insert(0, "Code")
from pathlib import Path
import io, json
from app import app, PROCESSED_HASHES

client = app.test_client()
sample_folder = Path("Code/sample_images")
image_files = sorted(list(sample_folder.glob("*.jpg")) + list(sample_folder.glob("*.png")))

print("=" * 105)
print(f"Testing {len(image_files)} REAL images from {sample_folder}:")
print("=" * 105)
print(f"{'File Name':<35} | {'Status':<10} | {'Food Detected':<13} | {'Food Conf':<10} | {'Condition':<10} | {'Cond Conf':<10}")
print("-" * 105)

for img_path in image_files:
    PROCESSED_HASHES.clear()
    img_bytes = img_path.read_bytes()
    
    res = client.post("/predict", data={"image": (io.BytesIO(img_bytes), img_path.name)}, content_type="multipart/form-data")
    data = res.get_json() or {}
    
    status = data.get("status", "error")
    food = data.get("detected_food", "-")
    food_conf = f"{data.get('food_confidence', 0.0):.1%}" if "food_confidence" in data else "-"
    cond = data.get("condition", "-")
    cond_conf = f"{data.get('condition_confidence', 0.0):.1%}" if "condition_confidence" in data else "-"
    
    if status == "not_food":
        food = "NOT_FOOD"
        cond = "REJECTED"
        food_conf = f"{data.get('confidence', 0.0):.1%}"
    elif status == "error":
        food = "ERROR"
        cond = data.get("error", {}).get("code", "-")
        
    print(f"{img_path.name[:35]:<35} | {status:<10} | {food:<13} | {food_conf:<10} | {cond:<10} | {cond_conf:<10}")
print("=" * 105)
