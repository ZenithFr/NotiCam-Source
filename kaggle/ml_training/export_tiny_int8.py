import tensorflow as tf
import numpy as np

# Tiny 3-conv INT8 test — same shape as FP32 test
inputs = tf.keras.Input(shape=(384, 512, 3), batch_size=1)
x = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu6')(inputs)
x = tf.keras.layers.Conv2D(8, 3, padding='same', activation='relu6')(x)
x = tf.keras.layers.Conv2D(2, 1, padding='same')(x)
model = tf.keras.Model(inputs, x)

def repr_gen():
    for _ in range(100):
        yield [np.random.rand(1, 384, 512, 3).astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = repr_gen
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.uint8
conv.inference_output_type = tf.uint8

tflite_model = conv.convert()
output_path = "/home/zenith/Work/auxcam/assets/models/tiny_int8.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"Exported {output_path} successfully! Size: {len(tflite_model)} bytes")
