import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

def dw_sep(x, ch, k=3, s=1):
    x = layers.DepthwiseConv2D(k, strides=s, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    x = layers.Conv2D(ch, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU(6.0)(x)
    return x

inp = layers.Input(shape=(384, 512, 3), batch_size=1)
# Encoder proxy
x = layers.Conv2D(24, 3, strides=2, padding='same', activation='relu6')(inp)   # 192x256
x = dw_sep(x, 32, s=2)    # 96x128  (1/4 tap for decoder)
low_feat = x
x = dw_sep(x, 64, s=2)    # 48x64
x = dw_sep(x, 128, s=2)   # 24x32
x = dw_sep(x, 256, s=2)   # 12x16  (1/16 tap)
high_feat = x

# ASPP-lite at 1/16
b0 = layers.Conv2D(96, 1, padding='same', activation='relu6')(high_feat)
b1 = dw_sep(high_feat, 96, k=3)
b2 = dw_sep(high_feat, 96, k=5)
deep = layers.Conv2D(96, 1, padding='same', activation='relu6')(layers.Concatenate()([b0, b1, b2]))
deep_up = layers.Resizing(96, 128, interpolation='bilinear')(deep)  # 1/4
low_proj = layers.Conv2D(32, 1, padding='same', activation='relu6')(low_feat)
fuse = layers.Concatenate()([deep_up, low_proj])
fuse = dw_sep(fuse, 96)
fuse = dw_sep(fuse, 96)
out = layers.Conv2D(2, 1, padding='same', activation='linear')(fuse)  # 96x128x2
model = tf.keras.Model(inp, out)

# Export FP16 model for GPU delegate
conv_gpu = tf.lite.TFLiteConverter.from_keras_model(model)
conv_gpu.optimizations = [tf.lite.Optimize.DEFAULT]
conv_gpu.target_spec.supported_types = [tf.float16]
tflite_gpu = conv_gpu.convert()
gpu_path = "/home/zenith/Work/auxcam/assets/models/synthetic_full_fp16.tflite"
with open(gpu_path, "wb") as f:
    f.write(tflite_gpu)
print(f"Exported {gpu_path} successfully! Size: {len(tflite_gpu)} bytes")
