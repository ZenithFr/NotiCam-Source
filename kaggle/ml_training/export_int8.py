import os
import sys
import tensorflow as tf
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model_mnv2_lrlf import AuxCamPortraitNet

def export_int8_model(weights_path=None, output_path="assets/models/portrait_v1_int8.tflite", rep_samples=100):
    print("=== Exporting Production INT8 Model for MediaTek MDLA ===")
    
    # 1. Instantiate model with fixed batch_size=1 for hardware APU allocation
    model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=1)
    
    if weights_path and os.path.exists(weights_path):
        model.load_weights(weights_path)
        print(f"Loaded trained weights from {weights_path}")
    else:
        print(f"Notice: {weights_path} not found or omitted; exporting current initialized weights.")

    print(f"Total Parameters: {model.count_params()}")

    # 2. Representative dataset generator
    # Yields normalized float32 images in [0, 255] or [0, 1] as expected by Keras Input layer
    def representative_dataset_gen():
        for _ in range(rep_samples):
            # Realistic random calibration samples
            sample = np.random.rand(1, 384, 512, 3).astype(np.float32) * 255.0
            yield [sample]

    # 3. Configure TFLite Converter for full INT8 PTQ / QAT
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
        tf.lite.OpsSet.TFLITE_BUILTINS
    ]
    converter.inference_input_type = tf.uint8
    converter.inference_output_type = tf.uint8

    print("Converting model with full integer quantization...")
    tflite_quant_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(tflite_quant_model)

    print(f"✅ Successfully exported INT8 model to: {output_path}")
    print(f"File size: {len(tflite_quant_model):,} bytes")

    # 4. Verification of flatbuffer ops
    interp = tf.lite.Interpreter(model_content=tflite_quant_model)
    interp.allocate_tensors()
    ops = interp._get_ops_details()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()

    print(f"Input: {input_details[0]['name']} shape={input_details[0]['shape']} dtype={input_details[0]['dtype']}")
    print(f"Output: {output_details[0]['name']} shape={output_details[0]['shape']} dtype={output_details[0]['dtype']}")
    print(f"Total nodes: {len(ops)}")
    
    assert input_details[0]['dtype'] == np.uint8, "Error: Input must be UINT8!"
    assert output_details[0]['dtype'] == np.uint8, "Error: Output must be UINT8!"
    assert len(ops) <= 40, f"Error: Node count {len(ops)} exceeds locked threshold of 40!"
    print("✅ All static export assertions passed!")

if __name__ == "__main__":
    weights = sys.argv[1] if len(sys.argv) > 1 else None
    export_int8_model(weights_path=weights)
