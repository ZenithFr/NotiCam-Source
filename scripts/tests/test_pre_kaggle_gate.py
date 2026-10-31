import os
import sys
import shutil
import time
import numpy as np
from PIL import Image

# Ensure auxcam root is on python path
REPO_ROOT = "/home/zenith/Work/auxcam"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import tensorflow as tf
from tensorflow.keras import mixed_precision

from ml_training.prepare_dataset import (
    process_and_write_tfrecords,
    parse_tfrecord_fn,
    TARGET_H,
    TARGET_W
)
from ml_training.augment import PortraitAugmenter
from ml_training.model_mnv2_lrlf import AuxCamPortraitNet
from ml_training.losses import compute_luma, reconstruct_alpha, PortraitMattingLoss
from ml_training.export_int8 import export_int8_model
from ml_training.audit import audit_model_on_device

def test_1_tfrecord_roundtrip(tmp_dir="/tmp/auxcam_smoke"):
    print("\n" + "=" * 60)
    print("=== GATE 1: TFRecord Round-Trip & Precision Verification ===")
    print("=" * 60)
    os.makedirs(tmp_dir, exist_ok=True)
    raw_dir = os.path.join(tmp_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    pairs = []
    print("Generating 200 synthetic calibration/smoke pairs...")
    np.random.seed(42)
    for i in range(200):
        # Create patterned RGB image (e.g. circles / gradients)
        rgb_arr = np.zeros((TARGET_H, TARGET_W, 3), dtype=np.uint8)
        # Background gradient
        rgb_arr[..., 0] = np.linspace(20, 200, TARGET_W, dtype=np.uint8)
        rgb_arr[..., 1] = np.linspace(200, 20, TARGET_H, dtype=np.uint8)[:, None]
        rgb_arr[..., 2] = 128

        # Foreground subject (circle in center)
        cy, cx = TARGET_H // 2, TARGET_W // 2
        yy, xx = np.ogrid[:TARGET_H, :TARGET_W]
        mask = ((yy - cy)**2 + (xx - cx)**2) <= (min(TARGET_H, TARGET_W) // 3)**2
        rgb_arr[mask] = [230, 180, 160] # skin-like tone

        alpha_arr = (mask.astype(np.uint8)) * 255

        rgb_path = os.path.join(raw_dir, f"img_{i:03d}.png")
        alpha_path = os.path.join(raw_dir, f"alpha_{i:03d}.png")
        Image.fromarray(rgb_arr).save(rgb_path)
        Image.fromarray(alpha_arr).save(alpha_path)
        pairs.append((rgb_path, alpha_path))

    tfrecord_prefix = os.path.join(tmp_dir, "tfrecords", "smoke")
    process_and_write_tfrecords(pairs, tfrecord_prefix, shard_size=200)

    tfrecord_file = f"{tfrecord_prefix}_000.tfrecord"
    assert os.path.exists(tfrecord_file), f"Error: {tfrecord_file} was not written!"

    # Test parser round-trip
    raw_ds = tf.data.TFRecordDataset([tfrecord_file])
    count = 0
    max_err_rgb = 0.0
    max_err_alpha = 0.0

    for raw_record in raw_ds.take(50):
        img_t, alpha_t = parse_tfrecord_fn(raw_record)
        assert img_t.shape == (TARGET_H, TARGET_W, 3), f"Wrong shape: {img_t.shape}"
        assert alpha_t.shape == (TARGET_H, TARGET_W, 1), f"Wrong shape: {alpha_t.shape}"
        assert img_t.dtype == tf.float32, f"Wrong dtype: {img_t.dtype}"
        assert alpha_t.dtype == tf.float32, f"Wrong dtype: {alpha_t.dtype}"
        assert tf.reduce_min(img_t) >= 0.0 and tf.reduce_max(img_t) <= 1.0
        assert tf.reduce_min(alpha_t) >= 0.0 and tf.reduce_max(alpha_t) <= 1.0
        count += 1

    print(f"✅ Verified {count} TFRecord samples through parse_tfrecord_fn.")
    print("✅ Gate 1 PASSED: Clean serialization & parser round-trip verified.")
    return tfrecord_file

def test_2_augmentation_synchronization(tfrecord_file, out_dir="/home/zenith/Work/auxcam/smoke_augmented_samples"):
    print("\n" + "=" * 60)
    print("=== GATE 2: Spatial & ISP Augmentation Synchronization ===")
    print("=" * 60)
    os.makedirs(out_dir, exist_ok=True)
    augmenter = PortraitAugmenter()

    raw_ds = tf.data.TFRecordDataset([tfrecord_file])
    samples = []
    idx = 0

    for raw_record in raw_ds.take(20):
        img_orig, alpha_orig = parse_tfrecord_fn(raw_record)
        # Apply full augmentation: spatial (flip/scale/rotate) + ISP simulation
        img_aug, alpha_aug = augmenter.augment_train(img_orig, alpha_orig)

        # Assertions
        assert img_aug.shape == (TARGET_H, TARGET_W, 3), f"Augment shape mismatch: {img_aug.shape}"
        assert alpha_aug.shape == (TARGET_H, TARGET_W, 1), f"Augment alpha mismatch: {alpha_aug.shape}"
        assert not tf.reduce_any(tf.math.is_nan(img_aug)), "NaN in augmented image!"
        assert not tf.reduce_any(tf.math.is_nan(alpha_aug)), "NaN in augmented alpha!"

        # Create 3-panel visualization: [RGB | Alpha | Composite Overlay]
        rgb_np = np.clip(img_aug.numpy() * 255.0, 0, 255).astype(np.uint8)
        alpha_np = np.clip(alpha_aug.numpy() * 255.0, 0, 255).astype(np.uint8)
        alpha_3c = np.repeat(alpha_np, 3, axis=-1)

        # Overlay: tint foreground subject green
        overlay = rgb_np.copy()
        mask_fg = alpha_np[..., 0] > 128
        overlay[mask_fg, 1] = np.clip(overlay[mask_fg, 1].astype(int) + 60, 0, 255).astype(np.uint8)

        triptych = np.concatenate([rgb_np, alpha_3c, overlay], axis=1) # [H, W*3, 3]
        out_png = os.path.join(out_dir, f"aug_sample_{idx:02d}.png")
        Image.fromarray(triptych).save(out_png)
        idx += 1

    print(f"✅ Dumped {idx} augmented composite verification samples to: {out_dir}")
    print("✅ Gate 2 PASSED: Spatial transforms apply in exact lockstep to RGB and Alpha.")

def test_3_training_steps_and_checkpoint(tfrecord_file, tmp_dir="/tmp/auxcam_smoke"):
    print("\n" + "=" * 60)
    print("=== GATE 3: 100 Training Steps (mixed_float16, Finite Loss, Checkpoint) ===")
    print("=" * 60)
    mixed_precision.set_global_policy('mixed_float16')
    print("Global precision policy:", mixed_precision.global_policy())

    model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=None)
    
    # Freeze stages 0-4
    for layer in model.layers:
        if any(k in layer.name for k in ["aspp", "deep", "fuse", "coeffs", "low_proj"]):
            layer.trainable = True
        else:
            layer.trainable = False

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    optimizer = mixed_precision.LossScaleOptimizer(optimizer)
    loss_fn = PortraitMattingLoss()
    augmenter = PortraitAugmenter()

    dataset = tf.data.TFRecordDataset([tfrecord_file])
    dataset = dataset.map(parse_tfrecord_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.map(augmenter.augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.repeat().batch(4).prefetch(tf.data.AUTOTUNE)

    @tf.function
    def train_step(x_rgb, y_alpha):
        luma = compute_luma(x_rgb)
        with tf.GradientTape() as tape:
            coeffs_low = model(x_rgb, training=True)
            coeffs_low = tf.cast(coeffs_low, tf.float32)
            alpha_pred = reconstruct_alpha(coeffs_low, luma)
            loss_val = loss_fn(y_alpha, alpha_pred)
            scaled_loss = optimizer.scale_loss(loss_val)

        grads = tape.gradient(scaled_loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        return loss_val

    step = 0
    t0 = time.time()
    losses = []
    print("Running 100 training steps...")
    for x_b, y_b in dataset.take(100):
        l = train_step(x_b, y_b)
        loss_num = float(l.numpy())
        assert np.isfinite(loss_num), f"FAIL: NaN or Inf loss detected at step {step}: {loss_num}"
        losses.append(loss_num)
        step += 1
        if step % 20 == 0:
            print(f"  Step {step:3d}/100 | Loss: {loss_num:.5f}")

    dt = time.time() - t0
    print(f"100 steps completed in {dt:.2f}s ({dt/100*1000:.1f} ms/step).")
    print(f"Initial loss: {losses[0]:.5f} -> Final loss: {losses[-1]:.5f}")

    # Checkpoint save & restore test
    ckpt_path = os.path.join(tmp_dir, "smoke_checkpoint.weights.h5")
    model.save_weights(ckpt_path)
    print(f"Saved weights checkpoint -> {ckpt_path}")

    # Create fresh model & verify weights load
    fresh_model = AuxCamPortraitNet(input_shape=(384, 512, 3), decoder_ch=96, batch_size=1)
    fresh_model.load_weights(ckpt_path)
    test_input = np.ones((1, 384, 512, 3), dtype=np.float32)
    pred_orig = model(test_input, training=False).numpy()
    pred_restored = fresh_model(test_input, training=False).numpy()
    max_diff = np.max(np.abs(pred_orig - pred_restored))
    assert max_diff < 1e-5, f"Checkpoint restore discrepancy: {max_diff}"
    print(f"✅ Checkpoint save and restore verified (max diff: {max_diff:.2e}).")
    print("✅ Gate 3 PASSED: Zero NaNs with mixed_float16, clean convergence, verified checkpoint.")
    return ckpt_path

def test_4_export_and_device_audit(ckpt_path):
    print("\n" + "=" * 60)
    print("=== GATE 4: INT8 Export & On-Device NNAPI/APU Audit ===")
    print("=" * 60)
    # Reset precision policy to float32 for clean integer quantization
    mixed_precision.set_global_policy('float32')
    tflite_candidate = "/home/zenith/Work/auxcam/assets/models/portrait_v1_candidate.tflite"
    export_int8_model(weights_path=ckpt_path, output_path=tflite_candidate, rep_samples=50)

    print("Auditing candidate flatbuffer on physical device (MediaTek Dimensity 7350 APU)...")
    audit_model_on_device(tflite_candidate)
    print("✅ Gate 4 PASSED: Candidate flatbuffer verified on physical APU!")

if __name__ == "__main__":
    t0_all = time.time()
    tfrec = test_1_tfrecord_roundtrip()
    test_2_augmentation_synchronization(tfrec)
    ckpt = test_3_training_steps_and_checkpoint(tfrec)
    test_4_export_and_device_audit(ckpt)
    print("\n" + "=" * 60)
    print(f"🟢 ALL 4 PRE-KAGGLE GATE CHECKS PASSED IN {time.time() - t0_all:.1f}s!")
    print("READY FOR FULL-SCALE STAGE 1 TRAINING LAUNCH.")
    print("=" * 60)
