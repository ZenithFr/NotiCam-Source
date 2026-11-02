import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_mnv2_lrlf import AuxCamPortraitNet
from prepare_dataset import load_tfrecord_dataset

def export_int8_model(weights_path, output_path="assets/models/portrait_v1_int8.tflite", rep_samples=100, rep_pattern=None):
    print("=== Exporting Production INT8 Model for MediaTek MDLA ===")
    
    if not weights_path or not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Trained weights required for export: {weights_path}")
    if not rep_pattern or not tf.io.gfile.glob(rep_pattern):
        raise FileNotFoundError(f"Real calibration TFRecords required: {rep_pattern}")
    if rep_samples <= 0:
        raise ValueError("rep_samples must be positive")
    # Fixed batch_size=1 for hardware APU allocation.
    model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=1)
    model.load_weights(weights_path)
    print(f"Loaded trained weights from {weights_path}")
    print(f"Total Parameters: {model.count_params()}")

    # The model was trained on [0, 1] RGB; calibrating with [0, 255]
    # random pixels produces an incompatible quantization scale.
    def representative_dataset_gen():
        data = load_tfrecord_dataset(rep_pattern, is_training=False).take(rep_samples)
        count = 0
        for image, _ in data:
            count += 1
            yield [tf.expand_dims(image, 0).numpy()]
        if count == 0:
            raise ValueError("Calibration TFRecords contain no examples")

    # Configure TFLite Converter for full INT8 PTQ.
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("Converting model with full integer quantization...")
    tflite_quant_model = converter.convert()

    # Verify before publishing the artifact.
    interp = tf.lite.Interpreter(model_content=tflite_quant_model)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    print(f"Input: {input_details[0]['name']} shape={input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['name']} shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    print(f"Total nodes: {len(ops)}")
    
    if input_details[0]['dtype'] != np.uint8 or output_details[0]['dtype'] != np.uint8:
        raise ValueError("Input and output must be UINT8")
    if len(ops) > 40:
        raise ValueError(f"Node count {len(ops)} exceeds locked threshold of 40")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)
    print(f"Exported {len(tflite_quant_model):,} bytes to {output_path}")
    print("✅ All static export assertions passed!")

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else None
    pattern = sys.argv[2] if len(sys.argv) > 2 else None
    export_int8_model(weights_path=weights, rep_pattern=pattern)
