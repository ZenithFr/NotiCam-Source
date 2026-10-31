import tensorflow as tf

# Build clean 3-layer network
inp = tf.keras.Input(shape=(384, 512, 3), name="rgb")
x = tf.keras.layers.Conv2D(16, (3, 3), strides=(2, 2), padding="same", use_bias=False, activation="relu")(inp)
x = tf.keras.layers.DepthwiseConv2D((3, 3), strides=(2, 2), padding="same", use_bias=False, activation="relu")(x)
out = tf.keras.layers.Conv2D(2, (1, 1), padding="same", use_bias=False, activation="relu")(x)
tf_model = tf.keras.Model(inputs=inp, outputs=out)

converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
tflite_model = converter.convert()

with open("/home/zenith/Work/auxcam/assets/models/tiny_test.tflite", "wb") as f:
    f.write(tflite_model)

print("Exported tiny_test.tflite successfully! Size:", len(tflite_model))
