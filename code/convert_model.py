import json
import sys
from pathlib import Path

import keras
import tensorflow as tf

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else ".")


def main() -> None:
    config = json.loads((MODEL_DIR / "final_config.json").read_text(encoding="utf-8"))
    source = MODEL_DIR / config["model_file"]
    height, width = (int(v) for v in config["image_size"])
    model = keras.models.load_model(source, compile=False)

    @tf.function(input_signature=[tf.TensorSpec([1, height, width, 3], tf.float32, name="normalized_rgb")])
    def serve(x):
        return {"rotten_probability": model(x, training=False)}

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], model)
    converter.optimizations = []
    tflite_path = OUT_DIR / "food_model.tflite"
    tflite_path.write_bytes(converter.convert())


if __name__ == "__main__":
    main()
