import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from model_mnv2_lrlf import AuxCamPortraitNet
import tensorflow as tf
import numpy as np

model = AuxCamPortraitNet()
print("PARAMS:", model.count_params())
print("LAYERS (approx nodes):", len(model.layers))

def rep():
    for _ in range(100):
        yield [np.random.rand(1, 384, 512, 3).astype(np.float32) * 255.0]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
    tf.lite.OpsSet.TFLITE_BUILTINS
]
conv.inference_input_type  = tf.uint8
conv.inference_output_type = tf.uint8

tflite_bytes = conv.convert()
out_path = "/home/zenith/Work/auxcam/assets/models/portrait_v1_candidate.tflite"
with open(out_path, "wb") as f:
    f.write(tflite_bytes)

print(f"Saved {out_path}! Size: {len(tflite_bytes)} bytes")
