
import hashlib
import json
import sys
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
from ai_edge_litert.interpreter import Interpreter

MODEL_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else "model")
OUT_DIR = Path(sys.argv[2] if len(sys.argv) > 2 else ".")
MAX_ALLOWED_DIFF = 1e-4      # tolerance on the Rotten probability
N_RANDOM, N_SMOOTH = 40, 40  # verification inputs
PROBE_SEED = 0               # the same probe is re-checked by app.py at every server start


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verification_inputs(height: int, width: int) -> np.ndarray:
    """Deterministic inputs in [0, 1]: noise images and smooth colour images."""
    rng = np.random.default_rng(42)
    noise = rng.random((N_RANDOM, height, width, 3), dtype=np.float32)
    smooth = []
    for _ in range(N_SMOOTH):
        small = rng.random((1, 7, 7, 3), dtype=np.float32)
        smooth.append(tf.image.resize(small, (height, width), method="bilinear").numpy()[0])
    return np.concatenate([noise, np.stack(smooth)]).astype(np.float32)


def main() -> None:
    config = json.loads((MODEL_DIR / "final_config.json").read_text(encoding="utf-8"))
    source = MODEL_DIR / config["model_file"]
    height, width = (int(v) for v in config["image_size"])
    threshold = float(config["threshold"])
    print(f"[1/4] Loading {source.name} (sha256 {sha256(source)[:12]}) "
          f"with TensorFlow {tf.__version__} / Keras {keras.__version__}")
    model = keras.models.load_model(source, compile=False)
    print(f"      parameters: {model.count_params():,}")

    print("[2/4] Converting to TFLite (float32, no quantization, training=False)")

    @tf.function(input_signature=[tf.TensorSpec([1, height, width, 3], tf.float32, name="normalized_rgb")])
    def serve(x):
        return {"rotten_probability": model(x, training=False)}  # augmentation layers off

    converter = tf.lite.TFLiteConverter.from_concrete_functions([serve.get_concrete_function()], model)
    converter.optimizations = []  # keep float32 weights exactly
    tflite_path = OUT_DIR / "food_model.tflite"
    tflite_path.write_bytes(converter.convert())
    print(f"      wrote {tflite_path.name}: {tflite_path.stat().st_size / 2**20:.1f} MB "
          f"(the .keras file is {source.stat().st_size / 2**20:.1f} MB)")

    print(f"[3/4] Verifying: original Keras model vs converted model with LiteRT "
          f"on {N_RANDOM + N_SMOOTH} inputs")
    inputs = verification_inputs(height, width)
    keras_out = model.predict(inputs, batch_size=16, verbose=0)[:, 0]

    interpreter = Interpreter(model_path=str(tflite_path), num_threads=1)
    interpreter.allocate_tensors()
    in_idx = interpreter.get_input_details()[0]["index"]
    out_idx = interpreter.get_output_details()[0]["index"]

    def run_tflite(x: np.ndarray) -> float:
        interpreter.set_tensor(in_idx, x[np.newaxis])
        interpreter.invoke()
        return float(interpreter.get_tensor(out_idx)[0, 0])

    tflite_out = np.array([run_tflite(x) for x in inputs])
    max_diff = float(np.abs(tflite_out - keras_out).max())
    decisions_match = bool(np.array_equal(keras_out >= threshold, tflite_out >= threshold))

    probe = np.random.default_rng(PROBE_SEED).random((1, height, width, 3), dtype=np.float32)
    probe_keras = float(model.predict(probe, verbose=0)[0, 0])
    probe_tflite = run_tflite(probe[0])

    print(f"      max |difference| = {max_diff:.2e} (allowed {MAX_ALLOWED_DIFF:.0e})")
    print(f"      Fresh/Rotten decisions identical at threshold {threshold}: {decisions_match}")
    print(f"      startup probe: keras={probe_keras:.7f} tflite={probe_tflite:.7f}")
    if max_diff > MAX_ALLOWED_DIFF or not decisions_match or abs(probe_tflite - probe_keras) > MAX_ALLOWED_DIFF:
        sys.exit("CONVERSION REJECTED: the converted model does not match the original. Build stopped.")

    report = {
        "source_model": source.name,
        "source_sha256": sha256(source),
        "tflite_model": tflite_path.name,
        "tflite_sha256": sha256(tflite_path),
        "tensorflow_version": tf.__version__,
        "keras_version": keras.__version__,
        "quantization": "none (float32)",
        "parameters": int(model.count_params()),
        "verification_inputs": int(len(inputs)),
        "max_abs_difference": max_diff,
        "decisions_identical": decisions_match,
        "threshold": threshold,
        "probe_seed": PROBE_SEED,
        "probe_expected_output": probe_keras,
    }
    (OUT_DIR / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[4/4] Conversion verified. Wrote conversion_report.json")


if __name__ == "__main__":
    main()
