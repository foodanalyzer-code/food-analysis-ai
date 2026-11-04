import json
import sys
from pathlib import Path

import keras
import tensorflow as tf

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else ".")


# convert keras model to tflite
def main() -> None:
    config = json.loads((MODEL_DIR / "final_config.json").read_text(encoding="utf-8"))
    source = MODEL_DIR / config["model_file"]
    height, width = (int(v) for v in config["image_size"])
    print(f"[1/2] Loading {source.name} with TensorFlow {tf.__version__} / Keras {keras.__version__}")
    model = keras.models.load_model(source, compile=False)
    print(f"      parameters: {model.count_params():,}")

    print("[2/2] Converting to TFLite (float32, no quantization, training=False)")

    # prediction only, no training
    @tf.function(input_signature=[tf.TensorSpec([1, height, width, 3], tf.float32, name="normalized_rgb")])
    def serve(x):
        return {"rotten_probability": model(x, training=False)}  # augmentation layers off

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], model)
    converter.optimizations = []  # keep float32 weights exactly
    tflite_path = OUT_DIR / "food_model.tflite"
    tflite_path.write_bytes(converter.convert())
    print(f"      wrote {tflite_path.name}: {tflite_path.stat().st_size / 2**20:.1f} MB "
          f"(the .keras file is {source.stat().st_size / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
