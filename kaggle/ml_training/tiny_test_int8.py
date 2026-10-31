import tensorflow as tf
import numpy as np

# Exact same 3-node graph from Test 1 (Conv2D -> DepthwiseConv2D -> Conv2D), INT8 quantized
inp = tf.keras.Input(shape=(384, 512, 3), batch_size=1)
x = tf.keras.layers.Conv2D(8, 3, strides=2, padding='same', activation='relu6')(inp)
x = tf.keras.layers.DepthwiseConv2D(3, strides=2, padding='same', activation='relu6')(x)
x = tf.keras.layers.Conv2D(2, 1, padding='same')(x)
model = tf.keras.Model(inp, x)

print("Model output shape:", model.output_shape)

def rep():
    for _ in range(100):
        yield [np.random.rand(1, 384, 512, 3).astype(np.float32)]

conv = tf.lite.TFLiteConverter.from_keras_model(model)
conv.optimizations = [tf.lite.Optimize.DEFAULT]
conv.representative_dataset = rep
conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
conv.inference_input_type  = tf.uint8
conv.inference_output_type = tf.uint8

tflite_model = conv.convert()
output_path = "/home/zenith/Work/auxcam/assets/models/tiny_test_int8.tflite"
with open(output_path, "wb") as f:
    f.write(tflite_model)

print(f"Exported {output_path} successfully! Size: {len(tflite_model)} bytes")
