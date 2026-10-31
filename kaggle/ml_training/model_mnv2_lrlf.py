import tensorflow as tf
from tensorflow.keras import layers

def dw_sep(x, ch, k=3, s=1, name=""):
    x = layers.DepthwiseConv2D(k, strides=s, padding='same', use_bias=False, name=f"{name}_dw" if name else None)(x)
    x = layers.BatchNormalization(name=f"{name}_bn1" if name else None)(x)
    x = layers.ReLU(6.0, name=f"{name}_relu1" if name else None)(x)
    x = layers.Conv2D(ch, 1, padding='same', use_bias=False, name=f"{name}_pw" if name else None)(x)
    x = layers.BatchNormalization(name=f"{name}_bn2" if name else None)(x)
    x = layers.ReLU(6.0, name=f"{name}_relu2" if name else None)(x)
    return x

def AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=None):
    inp = layers.Input(shape=input_shape, batch_size=batch_size, name="rgb")
    
    # Backbone: Depthwise Separable Conv stages (MobileNet architecture)
    # Stage 0: 384x512 -> 192x256
    x = layers.Conv2D(24, 3, strides=2, padding='same', use_bias=False, activation='relu6', name="stage0_conv")(inp)
    
    # Stage 1: 192x256 -> 96x128 (1/4 scale tap for low-level features)
    x = dw_sep(x, 32, s=2, name="stage1")
    low_feat = x
    
    # Stage 2: 96x128 -> 48x64
    x = dw_sep(x, 64, s=2, name="stage2")
    
    # Stage 3: 48x64 -> 24x32
    x = dw_sep(x, 128, s=2, name="stage3")
    
    # Stage 4: 24x32 -> 12x16 (1/16 scale tap for high-level features)
    x = dw_sep(x, 256, s=2, name="stage4")
    high_feat = x
    
    # ASPP-lite at 1/16 scale
    b0 = layers.Conv2D(decoder_ch, 1, padding='same', activation='relu6', name="aspp_b0")(high_feat)
    b1 = dw_sep(high_feat, decoder_ch, k=3, name="aspp_b1")
    b2 = dw_sep(high_feat, decoder_ch, k=5, name="aspp_b2")
    deep = layers.Conv2D(decoder_ch, 1, padding='same', activation='relu6', name="aspp_deep")(layers.Concatenate(name="aspp_cat")([b0, b1, b2]))
    
    # Bilinear upsample 4x from 12x16 to 96x128 (1/4 scale)
    deep_up = layers.Resizing(input_shape[0] // 4, input_shape[1] // 4, interpolation='bilinear', name="deep_up")(deep)
    
    # Low-level feature projection
    low_proj = layers.Conv2D(32, 1, padding='same', activation='relu6', name="low_proj")(low_feat)
    
    # Fusion & Refinement
    fuse = layers.Concatenate(name="fuse_cat")([deep_up, low_proj])
    fuse = dw_sep(fuse, decoder_ch, name="fuse_dw1")
    fuse = dw_sep(fuse, decoder_ch, name="fuse_dw2")
    
    # Final bilateral slice output: 2 channels [alpha, depth_guide] at 96x128
    out = layers.Conv2D(2, 1, padding='same', activation='linear', name="coeffs")(fuse)
    
    return tf.keras.Model(inp, out, name="AuxCamPortraitNet")

if __name__ == "__main__":
    m = AuxCamPortraitNet()
    print("AuxCamPortraitNet instantiated successfully!")
    print("Params:", m.count_params())
    print("Output shape:", m.output_shape)
